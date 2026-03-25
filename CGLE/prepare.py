"""
CGLE Spiral Hunter — Fixed Harness
====================================
Complex Ginzburg-Landau equation (2D):  ∂ₜA = A + (1+ic₁)∇²A - (1+ic₂)|A|²A

Plane wave: A₀ = exp(iωt),  |A₀| = 1,  ω = -c₂
Benjamin-Feir instability: c₁c₂ > 1  →  plane wave unstable
Eckhaus instability, phase turbulence, defect (amplitude) turbulence

Score = peak_defect_density × (1 + growth_rate) × bonus
  peak_defect_density = max(n_defects(t)) / N  over all t
    (n_defects = topological vortex count; divide by N for natural scaling)
  growth_rate = relative increase in defect count last 20% vs prior 20%
  bonus       = 1.5× if final_defect_density > 0.1 × peak (still active)

DO NOT MODIFY THIS FILE. Agent modifies simulate.py only.
"""

import numpy as np
import os, json, hashlib, time
from pathlib import Path

# ── Grid / solver constants ──────────────────────────────────────────────────
N               = 128
L               = 64.0          # domain size (arbitrary units)
DT              = 0.1
MAX_SIM_TIME    = 300.0
MAX_WALL_SECONDS = 120
RECORD_EVERY    = 20            # steps between metric snapshots (every 2 t-units)

EXP_DIR = Path(__file__).parent / "experiments"
EXP_DIR.mkdir(exist_ok=True)

dx   = L / N
x1d  = np.linspace(0, L, N, endpoint=False)
x, y = np.meshgrid(x1d, x1d)

kx1d = np.fft.fftfreq(N, d=dx) * 2 * np.pi
kx, ky = np.meshgrid(kx1d, kx1d)
k2 = kx**2 + ky**2


# ── Solver ───────────────────────────────────────────────────────────────────

def cgle_step(A_hat, c1, c2, lin_half):
    """
    One Strang-split step for ∂ₜA = A + (1+ic₁)∇²A - (1+ic₂)|A|²A.

    Linear propagator (half-step in k-space):
        Â → Â · exp([1 - (1+ic₁)k²] · dt/2)

    Nonlinear step (exact solution of dA/dt = -(1+ic₂)|A|²A):
        |A(t)|² = |A(0)|² / (1 + 2|A(0)|²·dt)
        A(dt)   = A(0) · (1 + 2|A(0)|²·dt)^(-(1+ic₂)/2)
    """
    A_hat = A_hat * lin_half
    A = np.fft.ifft2(A_hat)
    factor = (1.0 + 2.0 * np.abs(A)**2 * DT) ** (-(1.0 + 1j * c2) / 2.0)
    A_hat = np.fft.fft2(A * factor)
    A_hat = A_hat * lin_half
    return A_hat


def make_lin_half(c1):
    """Precompute the half-step linear propagator for given c1."""
    return np.exp((1.0 - (1.0 + 1j * c1) * k2) * (DT / 2.0))


# ── Defect counting ───────────────────────────────────────────────────────────

def count_defects(A):
    """
    Count topological phase defects (vortex cores) in 2D complex field A.
    A defect at (i,j) corresponds to a ±2π winding of phase around a plaquette.
    """
    phase = np.angle(A)
    dpx = np.angle(np.exp(1j * (np.roll(phase, -1, axis=1) - phase)))
    dpy = np.angle(np.exp(1j * (np.roll(phase, -1, axis=0) - phase)))
    winding = dpx + np.roll(dpy, -1, axis=1) - np.roll(dpx, -1, axis=0) - dpy
    return int(np.sum(np.abs(winding) > np.pi))


# ── Metrics / scoring ─────────────────────────────────────────────────────────

def compute_metrics(A_hat, t, step, wall):
    A = np.fft.ifft2(A_hat)
    nd = count_defects(A)
    amp = np.abs(A)
    return {
        'time':           float(t),
        'step':           int(step),
        'wall_elapsed':   float(wall),
        'n_defects':      nd,
        'defect_density': nd / N,       # scaled by N for natural units
        'mean_amp':       float(amp.mean()),
        'amp_std':        float(amp.std()),
    }


def compute_score(metrics_history):
    if not metrics_history:
        return 0.0, {}

    densities = [m['defect_density'] for m in metrics_history]
    peak_density = max(densities)

    n = len(densities)
    late = densities[max(0, int(0.8 * n)):]
    mid  = densities[max(0, int(0.6 * n)):max(0, int(0.8 * n))]
    egr = 0.0
    if mid and late and np.mean(mid) > 0:
        egr = max(0.0, (np.mean(late) - np.mean(mid)) / np.mean(mid))

    final_density = densities[-1]
    bonus = 1.5 if (peak_density > 0 and final_density > 0.1 * peak_density) else 1.0

    score = peak_density * (1.0 + egr) * bonus
    return score, {
        'peak_defect_density':  peak_density,
        'final_defect_density': final_density,
        'defect_growth_rate':   egr,
        'bonus':                bonus,
    }


# ── Main simulation runner ─────────────────────────────────────────────────────

def run_simulation(initial_condition_fn, c1, c2, config=None):
    """
    Args:
        initial_condition_fn: f(x, y) -> complex array of shape (N, N)
        c1: linear dispersion coefficient
        c2: nonlinear frequency shift
        config: dict of metadata
    Returns:
        result dict with score and metrics
    """
    if config is None:
        config = {}

    ts  = time.strftime("%Y%m%d_%H%M%S")
    uid = hashlib.md5(f"{ts}{str(config)}".encode()).hexdigest()[:6]
    exp_id = f"{ts}_{uid}"
    config.update({'N': N, 'L': L, 'c1': c1, 'c2': c2,
                   'max_wall_seconds': MAX_WALL_SECONDS})

    print("=" * 60)
    print(f"Experiment {exp_id}")
    print(f"Config: {json.dumps({k: v for k, v in config.items()}, indent=2)}")
    print("=" * 60)
    print(f"{'Step':>8}  {'Time':>8}  {'Defects':>8}  {'Density':>10}")
    print("-" * 60)

    lin_half = make_lin_half(c1)
    A = initial_condition_fn(x, y).astype(complex)
    A_hat = np.fft.fft2(A)

    metrics_history = []
    t = 0.0
    wall_start = time.time()
    total_steps = int(MAX_SIM_TIME / DT)

    for step in range(total_steps):
        A_hat = cgle_step(A_hat, c1, c2, lin_half)
        t += DT
        wall = time.time() - wall_start

        if step % RECORD_EVERY == 0:
            m = compute_metrics(A_hat, t, step, wall)
            metrics_history.append(m)
            if step % (RECORD_EVERY * 25) == 0:
                print(f"{step:8d}  {t:8.1f}  {m['n_defects']:8d}  {m['defect_density']:10.4f}")

        if wall > MAX_WALL_SECONDS:
            print(f"\nWall time limit at t={t:.1f}")
            break

    score, score_info = compute_score(metrics_history)

    print("\n" + "=" * 60)
    print(f"Done: {step} steps, t={t:.1f}, wall={time.time()-wall_start:.1f}s")
    print(f"Score: {score:.4f}")
    print(f"Peak defect density: {score_info.get('peak_defect_density', 0):.4f}  "
          f"(= {score_info.get('peak_defect_density', 0)*N:.0f} defects)")
    print(f"Final defect density: {score_info.get('final_defect_density', 0):.4f}")
    print(f"Bonus: {score_info.get('bonus', 1.0):.1f}x")
    print("=" * 60)

    best_file = EXP_DIR.parent / "best.json"
    prev_best = 0.0
    if best_file.exists():
        with open(best_file) as f:
            prev_best = json.load(f).get('score', 0.0)

    if score > prev_best:
        print(f"\n*** NEW BEST! Score {score:.4f} > previous best {prev_best:.4f} ***")
        with open(best_file, 'w') as f:
            json.dump({'score': score, 'exp_id': exp_id, 'config': config}, f)
    else:
        print(f"\nScore {score:.4f} vs best {prev_best:.4f}")

    result = {
        'id':                   exp_id,
        'timestamp':            time.strftime("%Y-%m-%dT%H:%M:%S"),
        'config':               config,
        'score':                score,
        'peak_defect_density':  score_info.get('peak_defect_density', 0),
        'final_defect_density': score_info.get('final_defect_density', 0),
        'defect_growth_rate':   score_info.get('defect_growth_rate', 0),
        'bonus':                score_info.get('bonus', 1.0),
        'metrics_history':      metrics_history,
    }
    with open(EXP_DIR / f"{exp_id}.json", 'w') as f:
        json.dump(result, f)

    return result
