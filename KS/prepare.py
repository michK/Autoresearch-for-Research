"""
Kuramoto-Sivashinsky Auto-Research: Evaluation Harness
=======================================================
Fixed infrastructure for running and evaluating 1D Kuramoto-Sivashinsky
simulations on a periodic domain. NOT modified by the agent.

The KS equation models spatiotemporal chaos in 1D:
    ∂_t u + u ∂_x u + ∂_xx u + ∂_xxxx u = 0

Energy is injected at long wavelengths (k < 1) and dissipated at short
wavelengths (k > 1), producing a chaotic strange attractor. We hunt for
initial conditions that maximize transient amplitude growth — the analog of
vorticity blowup in the NS problem.

Hardware target: any modern laptop (1D is cheap).
Grid: N=512 on [0, L], L=32π, pseudospectral with Lawson-RK4.
"""

import numpy as np
import json
import os
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

# ===========================================================================
# Constants
# ===========================================================================

N = 512                      # Grid points
L = 32 * np.pi               # Domain length (chaotic regime: L >> 2π)
dx = L / N

MAX_WALL_SECONDS = 60        # 1 minute per experiment (1D is fast)
MAX_SIM_TIME = 200.0         # Maximum simulation time (KS is slow to saturate)
DT_SAFETY = 0.4              # CFL safety factor

METRIC_SAMPLE_INTERVAL = 0.5  # Sample metrics every this many time units

PROJECT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
BEST_FILE = PROJECT_DIR / "best.json"
LEADERBOARD_FILE = PROJECT_DIR / "leaderboard.json"

EXPERIMENTS_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Spectral utilities
# ===========================================================================

def get_wavenumbers():
    """Return 1D wavenumber array for N-point periodic domain of length L."""
    k = np.fft.rfftfreq(N, d=L / (2 * np.pi * N))
    return k


def get_dealias_mask(k):
    """2/3 dealiasing mask."""
    kmax = N // 2
    return (np.abs(k) < (2 / 3) * kmax).astype(float)


def ks_nonlinear(u_hat, k, dealias):
    """
    Nonlinear term of KS: -u * ∂_x u in spectral space.
    Uses pseudospectral evaluation (physical-space multiplication).
    """
    u_hat_d = u_hat * dealias
    u = np.fft.irfft(u_hat_d, n=N)
    dudx = np.fft.irfft(1j * k * u_hat_d, n=N)
    nl = u * dudx
    return -np.fft.rfft(nl) * dealias


def ks_step(u_hat, dt, k, dealias):
    """
    One timestep of KS using Lawson-RK4.

    Handles the stiff linear part -(k^2 - k^4) exactly via integrating factor,
    so only the nonlinear CFL constrains the timestep (not diffusive stability).

    Linear eigenvalue per mode: L_k = k^2 - k^4
      - k < 1: L_k > 0 → unstable (energy injection)
      - k > 1: L_k < 0 → stable (energy dissipation)
    """
    Lk = k**2 - k**4          # linear growth rate per mode
    E1 = np.exp(Lk * dt)      # full-step integrating factor
    E2 = np.exp(Lk * dt / 2)  # half-step integrating factor

    # Lawson-RK4 (Hochbruck & Ostermann 2010)
    N1 = ks_nonlinear(u_hat, k, dealias)
    a = E2 * u_hat + (dt / 2) * E2 * N1

    N2 = ks_nonlinear(a, k, dealias)
    b = E2 * u_hat + (dt / 2) * N2

    N3 = ks_nonlinear(b, k, dealias)
    c = E1 * u_hat + dt * E2 * N3

    N4 = ks_nonlinear(c, k, dealias)

    u_new = E1 * u_hat + (dt / 6) * (E1 * N1 + 2 * E2 * N2 + 2 * E2 * N3 + N4)
    return u_new


def adaptive_dt(u_hat, k, dealias):
    """CFL timestep based on max wave speed |u|."""
    u = np.fft.irfft(u_hat * dealias, n=N)
    u_max = max(np.max(np.abs(u)), 1e-10)
    return DT_SAFETY * dx / u_max


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(u_hat, k, dealias):
    """
    Compute KS diagnostics.

    Returns dict with:
      - max_u:        ||u||_inf (THE key observable, analog of max vorticity)
      - energy:       (1/L) ∫ u² dx  (mean square)
      - energy_input: rate of energy input from k<1 modes (growth from instability)
      - n_peaks:      number of local maxima (proxy for spatial complexity)
    """
    u_hat_d = u_hat * dealias
    u = np.fft.irfft(u_hat_d, n=N)

    max_u = float(np.max(np.abs(u)))
    energy = float(np.mean(u**2))

    # Energy injection rate: contribution of unstable modes
    unstable = (np.abs(k) < 1.0) & (np.abs(k) > 0)
    energy_spectrum = 2 * np.abs(u_hat_d)**2 / N**2  # one-sided PSD
    energy_input = float(np.sum(energy_spectrum[unstable] * (k[unstable]**2 - k[unstable]**4)))

    # Count peaks (complexity measure)
    sign_changes = np.sum(np.diff(np.sign(np.diff(u))) < 0)
    n_peaks = int(sign_changes)

    return {
        "max_u": max_u,
        "energy": energy,
        "energy_input": energy_input,
        "n_peaks": n_peaks,
    }


def compute_score(metrics_history):
    """
    Single scalar score for comparing KS experiments.

    Mirrors the NS scoring formula exactly:
      score = (peak_max_u / initial_max_u) × (1 + energy_growth_rate) × [1.5 bonus]

    The bonus applies if max_u is still growing at the wall clock limit
    (analog of the NS 'still-blowing-up' bonus).
    """
    if len(metrics_history) < 3:
        return 0.0

    max_us = [m["max_u"] for m in metrics_history]
    energies = [m["energy"] for m in metrics_history]
    times = [m["time"] for m in metrics_history]

    # Amplitude amplification: peak / initial
    u_initial = max(max_us[0], 1e-10)
    u_peak = max(max_us)
    u_amplification = u_peak / u_initial

    # Energy growth rate: slope of log(energy) vs time
    valid = [(t, e) for t, e in zip(times, energies) if e > 0]
    if len(valid) < 3:
        return u_amplification

    t_arr = np.array([v[0] for v in valid])
    e_arr = np.array([np.log(v[1]) for v in valid])
    if len(t_arr) > 1 and (t_arr[-1] - t_arr[0]) > 0:
        coeffs = np.polyfit(t_arr, e_arr, 1)
        energy_growth_rate = max(coeffs[0], 0.0)
    else:
        energy_growth_rate = 0.0

    score = u_amplification * (1.0 + energy_growth_rate)

    # Bonus: still growing at end of wall clock window
    if len(max_us) >= 5 and max_us[-1] > max_us[-3]:
        score *= 1.5

    return float(score)


# ===========================================================================
# Experiment logging
# ===========================================================================

def generate_experiment_id():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    return f"{ts}_{h}"


def save_experiment(exp_id, config, metrics_history, score, sim_code_hash):
    result = {
        "id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "score": score,
        "sim_code_hash": sim_code_hash,
        "num_timesteps": len(metrics_history),
        "peak_max_u": max(m["max_u"] for m in metrics_history) if metrics_history else 0,
        "final_energy": metrics_history[-1]["energy"] if metrics_history else 0,
        "metrics_history": metrics_history,
    }
    exp_file = EXPERIMENTS_DIR / f"{exp_id}.json"
    with open(exp_file, "w") as f:
        json.dump(result, f, indent=2)
    return result


def load_best():
    if BEST_FILE.exists():
        with open(BEST_FILE) as f:
            return json.load(f)
    return {"score": 0.0, "id": None}


def save_best(result):
    with open(BEST_FILE, "w") as f:
        json.dump({
            "score": result["score"],
            "id": result["id"],
            "peak_max_u": result["peak_max_u"],
            "timestamp": result["timestamp"],
        }, f, indent=2)


def update_leaderboard(result):
    if LEADERBOARD_FILE.exists():
        with open(LEADERBOARD_FILE) as f:
            lb = json.load(f)
    else:
        lb = []

    lb.append({
        "id": result["id"],
        "score": result["score"],
        "peak_max_u": result["peak_max_u"],
        "timestamp": result["timestamp"],
        "config": result["config"],
    })
    lb.sort(key=lambda x: x["score"], reverse=True)
    lb = lb[:20]

    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(lb, f, indent=2)

    return lb


# ===========================================================================
# Runner
# ===========================================================================

def run_simulation(initial_condition_fn, config=None):
    """
    Run a single KS simulation.

    Args:
        initial_condition_fn: callable(x, k) -> u_hat (spectral coefficients)
            The IC should be real-valued with zero mean (u_hat[0] = 0).
        config: dict of metadata for logging

    Returns:
        dict with experiment results
    """
    if config is None:
        config = {}
    config["N"] = N
    config["L"] = L
    config["max_wall_seconds"] = MAX_WALL_SECONDS

    exp_id = generate_experiment_id()
    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}")
    print(f"Config: {json.dumps(config, indent=2, default=str)}")
    print(f"{'='*60}")

    # Setup
    x = np.linspace(0, L, N, endpoint=False)
    k = get_wavenumbers()
    dealias = get_dealias_mask(k)

    # Generate initial conditions
    u_hat = initial_condition_fn(x, k)

    # Enforce zero mean (KS conserves mean; set to 0)
    u_hat[0] = 0.0

    # Apply dealiasing
    u_hat *= dealias

    # Normalize to unit max amplitude so score measures IC SHAPE, not scale.
    # (Prevents trivial exploit: score = attractor_amplitude / tiny_initial.)
    u_phys = np.fft.irfft(u_hat, n=N)
    u_max = np.max(np.abs(u_phys))
    if u_max > 1e-12:
        u_hat = u_hat / u_max

    # Track sim code hash
    sim_path = PROJECT_DIR / "simulate.py"
    sim_code_hash = hashlib.md5(sim_path.read_bytes()).hexdigest()[:8] if sim_path.exists() else "unknown"

    # Time integration
    t = 0.0
    step = 0
    metrics_history = []
    wall_start = time.time()
    last_metric_time = -1.0

    print(f"{'Step':>6} {'Time':>8} {'MaxU':>10} {'Energy':>12} {'nPeaks':>8} {'dt':>10}")
    print("-" * 64)

    while t < MAX_SIM_TIME:
        elapsed = time.time() - wall_start
        if elapsed > MAX_WALL_SECONDS:
            print(f"\nWall clock limit reached ({MAX_WALL_SECONDS}s)")
            break

        dt = adaptive_dt(u_hat, k, dealias)
        if t + dt > MAX_SIM_TIME:
            dt = MAX_SIM_TIME - t

        u_hat = ks_step(u_hat, dt, k, dealias)
        u_hat[0] = 0.0  # enforce zero mean

        t += dt
        step += 1

        # Sample metrics
        if t - last_metric_time >= METRIC_SAMPLE_INTERVAL:
            metrics = compute_metrics(u_hat, k, dealias)
            metrics["time"] = t
            metrics["step"] = step
            metrics["wall_elapsed"] = time.time() - wall_start
            metrics_history.append(metrics)
            last_metric_time = t

            if step % 500 == 0 or step <= 5:
                print(f"{step:6d} {t:8.2f} {metrics['max_u']:10.4f} "
                      f"{metrics['energy']:12.6f} {metrics['n_peaks']:8d} {dt:10.6f}")

        # Numerical divergence check
        if step % 100 == 0:
            check = np.max(np.abs(np.fft.irfft(u_hat, n=N)))
            if check > 1e6 or np.isnan(check):
                print(f"\nNumerical divergence at step {step}, t={t:.2f}")
                break

    total_wall = time.time() - wall_start
    score = compute_score(metrics_history)

    print(f"\n{'='*60}")
    print(f"Done: {step} steps, t={t:.2f}, wall={total_wall:.1f}s")
    print(f"Score: {score:.4f}")
    if metrics_history:
        print(f"Peak max_u: {max(m['max_u'] for m in metrics_history):.4f}")
        print(f"Final energy: {metrics_history[-1]['energy']:.6f}")
    print(f"{'='*60}")

    result = save_experiment(exp_id, config, metrics_history, score, sim_code_hash)

    best = load_best()
    if score > best["score"]:
        print(f"\n*** NEW BEST! Score {score:.4f} > previous best {best['score']:.4f} ***\n")
        save_best(result)
    else:
        print(f"\nScore {score:.4f} vs best {best['score']:.4f}")

    update_leaderboard(result)
    return result


if __name__ == "__main__":
    # Sanity check: single most-unstable mode
    def single_mode(x, k):
        # Mode n=11: k_11 = 11 * (2π/L) = 11/16 ≈ 0.69 ≈ 1/√2 (most unstable)
        u = np.cos(11 * 2 * np.pi * x / L)
        return np.fft.rfft(u)

    result = run_simulation(single_mode, config={"name": "single_mode_k11_sanity"})
    print(f"\nSanity score: {result['score']:.4f}")
