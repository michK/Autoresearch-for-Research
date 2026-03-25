"""
Navier-Stokes Autoresearch: Evaluation Harness
================================================
Fixed infrastructure for running and evaluating 3D incompressible Navier-Stokes
simulations on a periodic domain (torus T^3). NOT modified by the agent.

The Millennium Prize problem asks: do smooth solutions always exist in 3D,
or can they blow up in finite time? We hunt for initial conditions that
produce extreme vorticity growth, which is the signature of potential blowup
(by the Beale-Kato-Majda criterion).

Hardware target: 2-4 GB RAM, 1-2 CPU cores (Linode VPS).
Grid: 64^3 periodic domain, pseudospectral method.
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

# Grid resolution (64^3 is the sweet spot for 2-4GB RAM)
N = 64
L = 2 * np.pi           # Domain length (standard periodic box)
dx = L / N
DEALIAS_FACTOR = 2/3     # Standard 2/3 dealiasing rule

# Simulation budget
MAX_WALL_SECONDS = 180   # 3 minutes per experiment (keeps iteration fast)
MAX_SIM_TIME = 10.0      # Maximum simulation time in dimensionless units
DT_SAFETY = 0.5          # CFL safety factor

# Viscosity (lower = higher Reynolds number = more interesting dynamics)
# Agent can override this in simulate.py but we set a sane default
DEFAULT_NU = 0.01

# Evaluation
METRIC_SAMPLE_INTERVAL = 0.01  # Sample metrics every this many time units

# Paths
PROJECT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
BEST_FILE = PROJECT_DIR / "best.json"
LEADERBOARD_FILE = PROJECT_DIR / "leaderboard.json"

EXPERIMENTS_DIR.mkdir(exist_ok=True)


# ===========================================================================
# Spectral utilities
# ===========================================================================

def get_wavenumbers(n, L=2*np.pi):
    """Return 3D wavenumber grids for an n^3 periodic domain."""
    k1d = np.fft.fftfreq(n, d=L/(2*np.pi*n))
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d[:n//2+1], indexing='ij')
    return kx, ky, kz


def get_dealias_mask(kx, ky, kz, n, factor=DEALIAS_FACTOR):
    """2/3 dealiasing mask in spectral space."""
    kmax = n // 2
    mask = (
        (np.abs(kx) < factor * kmax) &
        (np.abs(ky) < factor * kmax) &
        (np.abs(kz) < factor * kmax)
    )
    return mask.astype(np.float64)


def project_divergence_free(ux_hat, uy_hat, uz_hat, kx, ky, kz):
    """Leray projection: remove divergent component in spectral space."""
    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1.0  # avoid division by zero
    k_dot_u = kx * ux_hat + ky * uy_hat + kz * uz_hat
    ux_hat -= kx * k_dot_u / k_sq
    uy_hat -= ky * k_dot_u / k_sq
    uz_hat -= kz * k_dot_u / k_sq
    return ux_hat, uy_hat, uz_hat


def compute_rhs(ux_hat, uy_hat, uz_hat, kx, ky, kz, nu, dealias_mask):
    """
    Compute RHS of N-S in spectral space:
      du/dt = -P[(u . grad)u] + nu * laplacian(u)
    where P is the Leray projector.

    Uses pseudospectral method: nonlinear term in physical space, then project.
    """
    # Transform to physical space
    ux = np.fft.irfftn(ux_hat * dealias_mask)
    uy = np.fft.irfftn(uy_hat * dealias_mask)
    uz = np.fft.irfftn(uz_hat * dealias_mask)

    # Compute velocity gradients in spectral space
    duxdx_hat = 1j * kx * ux_hat * dealias_mask
    duxdy_hat = 1j * ky * ux_hat * dealias_mask
    duxdz_hat = 1j * kz * ux_hat * dealias_mask
    duydx_hat = 1j * kx * uy_hat * dealias_mask
    duydy_hat = 1j * ky * uy_hat * dealias_mask
    duydz_hat = 1j * kz * uy_hat * dealias_mask
    duzdx_hat = 1j * kx * uz_hat * dealias_mask
    duzdy_hat = 1j * ky * uz_hat * dealias_mask
    duzdz_hat = 1j * kz * uz_hat * dealias_mask

    # Transform gradients to physical space
    duxdx = np.fft.irfftn(duxdx_hat)
    duxdy = np.fft.irfftn(duxdy_hat)
    duxdz = np.fft.irfftn(duxdz_hat)
    duydx = np.fft.irfftn(duydx_hat)
    duydy = np.fft.irfftn(duydy_hat)
    duydz = np.fft.irfftn(duydz_hat)
    duzdx = np.fft.irfftn(duzdx_hat)
    duzdy = np.fft.irfftn(duzdy_hat)
    duzdz = np.fft.irfftn(duzdz_hat)

    # Nonlinear term: (u . grad)u in physical space
    nlx = ux * duxdx + uy * duxdy + uz * duxdz
    nly = ux * duydx + uy * duydy + uz * duydz
    nlz = ux * duzdx + uy * duzdy + uz * duzdz

    # Back to spectral space
    nlx_hat = np.fft.rfftn(nlx)
    nly_hat = np.fft.rfftn(nly)
    nlz_hat = np.fft.rfftn(nlz)

    # Diffusion term: nu * laplacian(u) = -nu * k^2 * u_hat
    k_sq = kx**2 + ky**2 + kz**2
    diff_x = -nu * k_sq * ux_hat
    diff_y = -nu * k_sq * uy_hat
    diff_z = -nu * k_sq * uz_hat

    # RHS = -nonlinear + diffusion, then project
    rhsx = -nlx_hat + diff_x
    rhsy = -nly_hat + diff_y
    rhsz = -nlz_hat + diff_z

    rhsx, rhsy, rhsz = project_divergence_free(rhsx, rhsy, rhsz, kx, ky, kz)

    return rhsx, rhsy, rhsz


def adaptive_dt(ux_hat, uy_hat, uz_hat, nu, kx, ky, kz, safety=DT_SAFETY):
    """Compute adaptive timestep from CFL and diffusive constraints."""
    ux = np.fft.irfftn(ux_hat)
    uy = np.fft.irfftn(uy_hat)
    uz = np.fft.irfftn(uz_hat)

    u_max = max(np.max(np.abs(ux)), np.max(np.abs(uy)), np.max(np.abs(uz)), 1e-10)
    k_max = np.max(np.abs(kx))  # symmetric grid

    # CFL condition
    dt_cfl = dx / u_max

    # Diffusive stability
    dt_diff = dx**2 / (6 * nu) if nu > 0 else float('inf')

    return safety * min(dt_cfl, dt_diff, 0.05)


# ===========================================================================
# Metrics
# ===========================================================================

def compute_vorticity(ux_hat, uy_hat, uz_hat, kx, ky, kz):
    """Compute vorticity omega = curl(u) in physical space."""
    # omega_x = duz/dy - duy/dz
    # omega_y = dux/dz - duz/dx
    # omega_z = duy/dx - dux/dy
    wx = np.fft.irfftn(1j * ky * uz_hat - 1j * kz * uy_hat)
    wy = np.fft.irfftn(1j * kz * ux_hat - 1j * kx * uz_hat)
    wz = np.fft.irfftn(1j * kx * uy_hat - 1j * ky * ux_hat)
    return wx, wy, wz


def compute_metrics(ux_hat, uy_hat, uz_hat, kx, ky, kz, n=N, L_domain=L):
    """
    Compute all blowup-relevant metrics.

    Returns dict with:
      - max_vorticity: ||omega||_inf (THE key blowup indicator, per BKM theorem)
      - enstrophy: integral of |omega|^2 (should grow before blowup)
      - energy: integral of |u|^2 (should be conserved in Euler, decreasing in N-S)
      - max_velocity: ||u||_inf
      - palinstrophy: integral of |grad omega|^2 (higher-order blowup indicator)
    """
    wx, wy, wz = compute_vorticity(ux_hat, uy_hat, uz_hat, kx, ky, kz)
    vort_mag = np.sqrt(wx**2 + wy**2 + wz**2)

    ux = np.fft.irfftn(ux_hat)
    uy = np.fft.irfftn(uy_hat)
    uz = np.fft.irfftn(uz_hat)

    dV = (L_domain / n) ** 3

    max_vorticity = float(np.max(vort_mag))
    enstrophy = float(np.sum(vort_mag**2) * dV)
    energy = 0.5 * float(np.sum(ux**2 + uy**2 + uz**2) * dV)
    max_velocity = float(np.max(np.sqrt(ux**2 + uy**2 + uz**2)))

    # Palinstrophy: |grad omega|^2 (expensive but informative)
    k_sq = kx**2 + ky**2 + kz**2
    wx_hat = np.fft.rfftn(wx)
    wy_hat = np.fft.rfftn(wy)
    wz_hat = np.fft.rfftn(wz)
    palinstrophy = float(np.sum(
        np.abs(1j * kx * wx_hat)**2 + np.abs(1j * ky * wx_hat)**2 + np.abs(1j * kz * wx_hat)**2 +
        np.abs(1j * kx * wy_hat)**2 + np.abs(1j * ky * wy_hat)**2 + np.abs(1j * kz * wy_hat)**2 +
        np.abs(1j * kx * wz_hat)**2 + np.abs(1j * ky * wz_hat)**2 + np.abs(1j * kz * wz_hat)**2
    ) * dV / n**3)  # Parseval normalization

    return {
        "max_vorticity": max_vorticity,
        "enstrophy": enstrophy,
        "energy": energy,
        "max_velocity": max_velocity,
        "palinstrophy": palinstrophy,
    }


def compute_score(metrics_history):
    """
    Single scalar score for comparing experiments.

    Higher is "more interesting" (closer to potential blowup).

    We use: peak vorticity amplification ratio * enstrophy growth rate.
    This rewards initial conditions that show sustained, accelerating
    vorticity growth rather than just high initial vorticity.
    """
    if len(metrics_history) < 3:
        return 0.0

    vorts = [m["max_vorticity"] for m in metrics_history]
    enstrophies = [m["enstrophy"] for m in metrics_history]
    times = [m["time"] for m in metrics_history]

    # Vorticity amplification: peak / initial
    vort_initial = max(vorts[0], 1e-10)
    vort_peak = max(vorts)
    vort_amplification = vort_peak / vort_initial

    # Enstrophy growth rate: fit log(enstrophy) vs time for positive slope
    valid = [(t, e) for t, e in zip(times, enstrophies) if e > 0]
    if len(valid) < 3:
        return vort_amplification

    t_arr = np.array([v[0] for v in valid])
    e_arr = np.array([np.log(v[1]) for v in valid])
    if len(t_arr) > 1 and (t_arr[-1] - t_arr[0]) > 0:
        # Linear fit to log(enstrophy)
        coeffs = np.polyfit(t_arr, e_arr, 1)
        enstrophy_growth_rate = max(coeffs[0], 0.0)
    else:
        enstrophy_growth_rate = 0.0

    # Combined score
    score = vort_amplification * (1.0 + enstrophy_growth_rate)

    # Bonus: was vorticity still growing at end of simulation?
    # (suggests blowup might continue beyond our time window)
    if len(vorts) >= 5 and vorts[-1] > vorts[-3]:
        score *= 1.5

    return float(score)


# ===========================================================================
# Experiment logging
# ===========================================================================

def generate_experiment_id():
    """Generate a unique experiment ID from timestamp + random hash."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    return f"{ts}_{h}"


def save_experiment(exp_id, config, metrics_history, score, sim_code_hash):
    """Save experiment results to JSON."""
    result = {
        "id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "score": score,
        "sim_code_hash": sim_code_hash,
        "num_timesteps": len(metrics_history),
        "peak_max_vorticity": max(m["max_vorticity"] for m in metrics_history) if metrics_history else 0,
        "final_energy": metrics_history[-1]["energy"] if metrics_history else 0,
        "metrics_history": metrics_history,
    }

    exp_file = EXPERIMENTS_DIR / f"{exp_id}.json"
    with open(exp_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def load_best():
    """Load the current best score."""
    if BEST_FILE.exists():
        with open(BEST_FILE) as f:
            return json.load(f)
    return {"score": 0.0, "id": None}


def save_best(result):
    """Save new best result."""
    with open(BEST_FILE, "w") as f:
        json.dump({
            "score": result["score"],
            "id": result["id"],
            "peak_max_vorticity": result["peak_max_vorticity"],
            "timestamp": result["timestamp"],
        }, f, indent=2)


def update_leaderboard(result):
    """Maintain a top-20 leaderboard."""
    if LEADERBOARD_FILE.exists():
        with open(LEADERBOARD_FILE) as f:
            lb = json.load(f)
    else:
        lb = []

    lb.append({
        "id": result["id"],
        "score": result["score"],
        "peak_max_vorticity": result["peak_max_vorticity"],
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

def run_simulation(initial_condition_fn, nu=DEFAULT_NU, config=None):
    """
    Run a single N-S simulation with the given initial conditions.

    Args:
        initial_condition_fn: callable(x, y, z, kx, ky, kz) -> (ux_hat, uy_hat, uz_hat)
            Must return divergence-free spectral velocity field.
        nu: kinematic viscosity
        config: dict of configuration metadata for logging

    Returns:
        dict with experiment results
    """
    if config is None:
        config = {}
    config["nu"] = nu
    config["N"] = N
    config["max_wall_seconds"] = MAX_WALL_SECONDS

    exp_id = generate_experiment_id()
    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}")
    print(f"Config: {json.dumps(config, indent=2, default=str)}")
    print(f"{'='*60}")

    # Setup grid
    x1d = np.linspace(0, L, N, endpoint=False)
    x, y, z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
    kx, ky, kz = get_wavenumbers(N, L)
    dealias_mask = get_dealias_mask(kx, ky, kz, N)

    # Generate initial conditions
    ux_hat, uy_hat, uz_hat = initial_condition_fn(x, y, z, kx, ky, kz)

    # Ensure divergence-free
    ux_hat, uy_hat, uz_hat = project_divergence_free(ux_hat, uy_hat, uz_hat, kx, ky, kz)

    # Apply dealiasing
    ux_hat *= dealias_mask
    uy_hat *= dealias_mask
    uz_hat *= dealias_mask

    # Compute simulate.py hash for tracking code changes
    sim_path = PROJECT_DIR / "simulate.py"
    if sim_path.exists():
        sim_code_hash = hashlib.md5(sim_path.read_bytes()).hexdigest()[:8]
    else:
        sim_code_hash = "unknown"

    # Time integration (RK4)
    t = 0.0
    step = 0
    metrics_history = []
    wall_start = time.time()
    last_metric_time = -1.0

    print(f"{'Step':>6} {'Time':>8} {'MaxVort':>12} {'Enstrophy':>12} {'Energy':>12} {'dt':>10}")
    print("-" * 72)

    while t < MAX_SIM_TIME:
        # Wall clock check
        elapsed = time.time() - wall_start
        if elapsed > MAX_WALL_SECONDS:
            print(f"\nWall clock limit reached ({MAX_WALL_SECONDS}s)")
            break

        # Adaptive timestep
        dt = adaptive_dt(ux_hat, uy_hat, uz_hat, nu, kx, ky, kz)

        # Don't overshoot
        if t + dt > MAX_SIM_TIME:
            dt = MAX_SIM_TIME - t

        # RK4 integration
        k1x, k1y, k1z = compute_rhs(ux_hat, uy_hat, uz_hat, kx, ky, kz, nu, dealias_mask)

        k2x, k2y, k2z = compute_rhs(
            ux_hat + 0.5*dt*k1x, uy_hat + 0.5*dt*k1y, uz_hat + 0.5*dt*k1z,
            kx, ky, kz, nu, dealias_mask)

        k3x, k3y, k3z = compute_rhs(
            ux_hat + 0.5*dt*k2x, uy_hat + 0.5*dt*k2y, uz_hat + 0.5*dt*k2z,
            kx, ky, kz, nu, dealias_mask)

        k4x, k4y, k4z = compute_rhs(
            ux_hat + dt*k3x, uy_hat + dt*k3y, uz_hat + dt*k3z,
            kx, ky, kz, nu, dealias_mask)

        ux_hat += (dt/6) * (k1x + 2*k2x + 2*k3x + k4x)
        uy_hat += (dt/6) * (k1y + 2*k2y + 2*k3y + k4y)
        uz_hat += (dt/6) * (k1z + 2*k2z + 2*k3z + k4z)

        # Re-project (accumulation of numerical divergence)
        ux_hat, uy_hat, uz_hat = project_divergence_free(ux_hat, uy_hat, uz_hat, kx, ky, kz)

        t += dt
        step += 1

        # Sample metrics
        if t - last_metric_time >= METRIC_SAMPLE_INTERVAL:
            metrics = compute_metrics(ux_hat, uy_hat, uz_hat, kx, ky, kz)
            metrics["time"] = t
            metrics["step"] = step
            metrics["wall_elapsed"] = time.time() - wall_start
            metrics_history.append(metrics)
            last_metric_time = t

            if step % 50 == 0 or step <= 5:
                print(f"{step:6d} {t:8.4f} {metrics['max_vorticity']:12.4f} "
                      f"{metrics['enstrophy']:12.4f} {metrics['energy']:12.6f} {dt:10.6f}")

        # Blowup detection: if max vorticity exceeds a threshold, the
        # simulation is numerically diverging. Record and stop.
        if step % 10 == 0:
            quick_check = np.max(np.abs(np.fft.irfftn(ux_hat)))
            if quick_check > 1e6 or np.isnan(quick_check):
                print(f"\nNumerical blowup detected at step {step}, t={t:.4f}")
                metrics = compute_metrics(ux_hat, uy_hat, uz_hat, kx, ky, kz)
                metrics["time"] = t
                metrics["step"] = step
                metrics["wall_elapsed"] = time.time() - wall_start
                metrics["numerical_blowup"] = True
                metrics_history.append(metrics)
                break

    # Final metrics
    total_wall = time.time() - wall_start
    score = compute_score(metrics_history)

    print(f"\n{'='*60}")
    print(f"Experiment complete: {step} steps, t={t:.4f}, wall={total_wall:.1f}s")
    print(f"Score: {score:.4f}")
    if metrics_history:
        print(f"Peak max vorticity: {max(m['max_vorticity'] for m in metrics_history):.4f}")
        print(f"Final energy: {metrics_history[-1]['energy']:.6f}")
    print(f"{'='*60}")

    # Save
    result = save_experiment(exp_id, config, metrics_history, score, sim_code_hash)

    # Check against best
    best = load_best()
    if score > best["score"]:
        print(f"\n*** NEW BEST! Score {score:.4f} > previous best {best['score']:.4f} ***\n")
        save_best(result)
    else:
        print(f"\nScore {score:.4f} vs best {best['score']:.4f} (no improvement)")

    update_leaderboard(result)

    return result


if __name__ == "__main__":
    # Quick sanity test: Taylor-Green vortex
    def taylor_green(x, y, z, kx, ky, kz):
        ux = np.sin(x) * np.cos(y) * np.cos(z)
        uy = -np.cos(x) * np.sin(y) * np.cos(z)
        uz = np.zeros_like(x)
        return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)

    result = run_simulation(taylor_green, nu=0.01, config={"name": "taylor_green_sanity"})
    print(f"\nSanity test score: {result['score']:.4f}")

