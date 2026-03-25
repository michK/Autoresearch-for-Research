"""
Gray-Scott Auto-Research: Evaluation Harness
=============================================
Fixed infrastructure for running and evaluating 2D Gray-Scott reaction-diffusion
simulations on a periodic domain. NOT modified by the agent.

The Gray-Scott model:
    ∂u/∂t = Du ∇²u - uv² + F(1-u)
    ∂v/∂t = Dv ∇²v + uv² - (F+k)v

Two chemicals u (substrate) and v (activator) interact on a 2D domain. The
interplay of diffusion, reaction, feed rate F, and kill rate k produces an
extraordinary variety of patterns: spots, stripes, spirals, self-replicating
structures, and spatiotemporal chaos.

We hunt for initial conditions and parameters that maximize pattern complexity
— the richest, most intricate spatial structures achievable.

Grid: 128×128 periodic domain, pseudospectral with Strang splitting.
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

N = 128                       # Grid points per side (128×128)
L = 128.0                     # Domain size
dx = L / N

# Standard Gray-Scott diffusion coefficients
DEFAULT_DU = 0.16
DEFAULT_DV = 0.08

# Default parameters (in the "interesting" region of the phase diagram)
DEFAULT_F = 0.035
DEFAULT_K = 0.065

# Simulation budget
MAX_WALL_SECONDS = 120        # 2 minutes per experiment
MAX_SIM_TIME = 50000.0        # Gray-Scott patterns need long integration times
DT = 1.0                      # Fixed timestep (stable for standard parameters)

# Metrics sampling
METRIC_SAMPLE_INTERVAL = 100.0  # Sample every 100 time units

PROJECT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
BEST_FILE = PROJECT_DIR / "best.json"
LEADERBOARD_FILE = PROJECT_DIR / "leaderboard.json"

EXPERIMENTS_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Spectral utilities
# ===========================================================================

def get_wavenumbers_2d():
    """Return 2D wavenumber grids for N×N periodic domain."""
    k1d = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    kx, ky = np.meshgrid(k1d, k1d[:N // 2 + 1], indexing='ij')
    return kx, ky


def get_k_squared(kx, ky):
    """Return k² for 2D wavenumber grid."""
    return kx**2 + ky**2


# ===========================================================================
# Solver
# ===========================================================================

def gs_step(u, v, u_hat, v_hat, dt, F, k, Du, Dv, diff_u, diff_v,
            diff_u_half, diff_v_half):
    """
    One timestep of Gray-Scott using Strang splitting:
        1. Half diffusion step (exact in spectral space)
        2. Full reaction step (RK2 midpoint in physical space)
        3. Half diffusion step

    This gives second-order splitting accuracy.
    """
    # --- Half diffusion step ---
    u_hat *= diff_u_half
    v_hat *= diff_v_half
    u = np.fft.irfft2(u_hat)
    v = np.fft.irfft2(v_hat)

    # --- Full reaction step (RK2 midpoint) ---
    uv2 = u * v * v
    du = -uv2 + F * (1.0 - u)
    dv = uv2 - (F + k) * v

    u_mid = u + 0.5 * dt * du
    v_mid = v + 0.5 * dt * dv

    uv2_mid = u_mid * v_mid * v_mid
    du_mid = -uv2_mid + F * (1.0 - u_mid)
    dv_mid = uv2_mid - (F + k) * v_mid

    u = u + dt * du_mid
    v = v + dt * dv_mid

    # Clamp to physical range [0, 1]
    np.clip(u, 0, 1, out=u)
    np.clip(v, 0, 1, out=v)

    # --- Half diffusion step ---
    u_hat = np.fft.rfft2(u)
    v_hat = np.fft.rfft2(v)
    u_hat *= diff_u_half
    v_hat *= diff_v_half

    u = np.fft.irfft2(u_hat)
    v = np.fft.irfft2(v_hat)

    return u, v, u_hat, v_hat


# ===========================================================================
# Metrics
# ===========================================================================

def spectral_entropy(v_hat):
    """
    Shannon entropy of the power spectrum of v (excluding DC component).
    Measures how spread the pattern energy is across spatial frequencies.
    High entropy = complex patterns; low entropy = simple/uniform.
    """
    power = np.abs(v_hat)**2
    power[0, 0] = 0  # exclude mean
    total = np.sum(power)
    if total < 1e-20:
        return 0.0
    p = power / total
    # Avoid log(0)
    p_nz = p[p > 1e-30]
    return float(-np.sum(p_nz * np.log(p_nz)))


def compute_metrics(u, v, v_hat):
    """
    Compute Gray-Scott pattern diagnostics.

    Returns dict with:
      - spectral_entropy:  Shannon entropy of v power spectrum (complexity)
      - pattern_energy:    total energy in non-DC modes of v (pattern strength)
      - pattern_contrast:  (max(v) - min(v)) (pattern visibility)
      - mean_v:            spatial mean of v (how much activator exists)
      - n_spots:           approximate number of distinct spots/features
    """
    se = spectral_entropy(v_hat)

    # Pattern energy (excluding DC)
    power = np.abs(v_hat)**2
    power[0, 0] = 0
    pattern_energy = float(np.sum(power)) / (N * N)

    # Pattern contrast
    v_max = float(np.max(v))
    v_min = float(np.min(v))
    pattern_contrast = v_max - v_min

    # Mean v
    mean_v = float(np.mean(v))

    # Count spots: threshold v at mean + 1 sigma, count connected components
    # (approximate with peak counting)
    v_thresh = v > (mean_v + 0.5 * np.std(v))
    n_spots = int(np.sum(v_thresh) / max(1, N * N * 0.01))  # rough estimate

    return {
        "spectral_entropy": se,
        "pattern_energy": pattern_energy,
        "pattern_contrast": pattern_contrast,
        "mean_v": mean_v,
        "n_spots": n_spots,
    }


def compute_score(metrics_history):
    """
    Single scalar score for comparing Gray-Scott experiments.

    Score = peak_complexity × peak_contrast × (1 + entropy_growth_rate) × [1.5 bonus]

    where:
      - peak_complexity = max spectral entropy over the run
      - peak_contrast = max pattern contrast over the run
      - entropy_growth_rate = slope of spectral_entropy vs time
      - bonus ×1.5 if patterns are still evolving at wall clock end

    This rewards initial conditions and parameters that produce the richest,
    most visible, and still-evolving pattern structures.
    """
    if len(metrics_history) < 3:
        return 0.0

    entropies = [m["spectral_entropy"] for m in metrics_history]
    contrasts = [m["pattern_contrast"] for m in metrics_history]
    times = [m["time"] for m in metrics_history]

    peak_entropy = max(entropies)
    peak_contrast = max(contrasts)

    if peak_entropy < 1e-10 or peak_contrast < 1e-10:
        return 0.0

    # Entropy growth rate over the run
    t_arr = np.array(times)
    e_arr = np.array(entropies)
    if len(t_arr) > 2 and (t_arr[-1] - t_arr[0]) > 0:
        coeffs = np.polyfit(t_arr, e_arr, 1)
        entropy_growth_rate = max(coeffs[0], 0.0)
    else:
        entropy_growth_rate = 0.0

    score = peak_entropy * peak_contrast * (1.0 + entropy_growth_rate)

    # Bonus: patterns still evolving at end
    if len(entropies) >= 5 and entropies[-1] > entropies[-3]:
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
        "peak_spectral_entropy": max(m["spectral_entropy"] for m in metrics_history) if metrics_history else 0,
        "peak_contrast": max(m["pattern_contrast"] for m in metrics_history) if metrics_history else 0,
        "final_mean_v": metrics_history[-1]["mean_v"] if metrics_history else 0,
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
            "peak_spectral_entropy": result["peak_spectral_entropy"],
            "peak_contrast": result["peak_contrast"],
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
        "peak_spectral_entropy": result["peak_spectral_entropy"],
        "peak_contrast": result["peak_contrast"],
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

def run_simulation(initial_condition_fn, F=DEFAULT_F, k=DEFAULT_K,
                   Du=DEFAULT_DU, Dv=DEFAULT_DV, config=None):
    """
    Run a single Gray-Scott simulation.

    Args:
        initial_condition_fn: callable(x, y) -> (u, v)
            Returns 2D arrays (N×N) for initial u and v fields.
        F: feed rate
        k: kill rate
        Du, Dv: diffusion coefficients
        config: dict of metadata for logging

    Returns:
        dict with experiment results
    """
    if config is None:
        config = {}
    config["F"] = F
    config["k"] = k
    config["Du"] = Du
    config["Dv"] = Dv
    config["N"] = N
    config["L"] = L
    config["max_wall_seconds"] = MAX_WALL_SECONDS

    exp_id = generate_experiment_id()
    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}")
    print(f"Config: {json.dumps(config, indent=2, default=str)}")
    print(f"{'='*60}")

    # Setup grid
    x1d = np.linspace(0, L, N, endpoint=False)
    x, y = np.meshgrid(x1d, x1d, indexing='ij')
    kx, ky = get_wavenumbers_2d()
    k_sq = get_k_squared(kx, ky)

    # Precompute diffusion operators
    diff_u = np.exp(-Du * k_sq * DT)
    diff_v = np.exp(-Dv * k_sq * DT)
    diff_u_half = np.exp(-Du * k_sq * DT / 2)
    diff_v_half = np.exp(-Dv * k_sq * DT / 2)

    # Generate initial conditions
    u, v = initial_condition_fn(x, y)

    # Ensure physical range
    np.clip(u, 0, 1, out=u)
    np.clip(v, 0, 1, out=v)

    u_hat = np.fft.rfft2(u)
    v_hat = np.fft.rfft2(v)

    # Track sim code hash
    sim_path = PROJECT_DIR / "simulate.py"
    sim_code_hash = hashlib.md5(sim_path.read_bytes()).hexdigest()[:8] if sim_path.exists() else "unknown"

    # Time integration
    t = 0.0
    step = 0
    metrics_history = []
    wall_start = time.time()
    last_metric_time = -1.0

    print(f"{'Step':>8} {'Time':>8} {'Entropy':>10} {'Contrast':>10} "
          f"{'MeanV':>8} {'Spots':>6}")
    print("-" * 60)

    while t < MAX_SIM_TIME:
        elapsed = time.time() - wall_start
        if elapsed > MAX_WALL_SECONDS:
            print(f"\nWall clock limit reached ({MAX_WALL_SECONDS}s)")
            break

        u, v, u_hat, v_hat = gs_step(
            u, v, u_hat, v_hat, DT, F, k, Du, Dv,
            diff_u, diff_v, diff_u_half, diff_v_half
        )

        t += DT
        step += 1

        # Sample metrics
        if t - last_metric_time >= METRIC_SAMPLE_INTERVAL:
            metrics = compute_metrics(u, v, v_hat)
            metrics["time"] = t
            metrics["step"] = step
            metrics["wall_elapsed"] = time.time() - wall_start
            metrics_history.append(metrics)
            last_metric_time = t

            if step % 2000 == 0 or step <= 5:
                print(f"{step:8d} {t:8.0f} {metrics['spectral_entropy']:10.3f} "
                      f"{metrics['pattern_contrast']:10.4f} "
                      f"{metrics['mean_v']:8.4f} {metrics['n_spots']:6d}")

        # Divergence check
        if step % 1000 == 0:
            if np.any(np.isnan(u)) or np.any(np.isnan(v)):
                print(f"\nNumerical divergence at step {step}, t={t:.0f}")
                break

    total_wall = time.time() - wall_start
    score = compute_score(metrics_history)

    print(f"\n{'='*60}")
    print(f"Done: {step} steps, t={t:.0f}, wall={total_wall:.1f}s")
    print(f"Score: {score:.4f}")
    if metrics_history:
        print(f"Peak spectral entropy: {max(m['spectral_entropy'] for m in metrics_history):.3f}")
        print(f"Peak contrast: {max(m['pattern_contrast'] for m in metrics_history):.4f}")
        print(f"Final mean_v: {metrics_history[-1]['mean_v']:.4f}")
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
    # Sanity check: standard spot-forming parameters with center seed
    def center_seed(x, y):
        u = np.ones((N, N))
        v = np.zeros((N, N))
        cx, cy = L / 2, L / 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = r < 10
        u[mask] = 0.5
        v[mask] = 0.25
        return u, v

    result = run_simulation(center_seed, F=0.035, k=0.065,
                            config={"name": "center_seed_sanity"})
    print(f"\nSanity score: {result['score']:.4f}")
