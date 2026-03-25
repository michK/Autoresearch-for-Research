"""
NLS Rogue Wave Hunter — Fixed Harness
======================================
Focusing NLS equation:  i∂ₜψ + ∂ₓₓψ + |ψ|²ψ = 0

Background solution: ψ₀(x,t) = a·exp(ia²t)  (|ψ₀| = a everywhere)

Modulational instability (MI):
  Growth rate: σ(k) = k·√(2a² - k²)  for |k| < a√2
  Most unstable: k* = a,  σ_max = a²
  Peregrine soliton (rational): peak |ψ| = 3a  (3× background)
  Order-n rogue: peak = (2n+1)a

Score = peak_amplification × (1 + growth_rate) × bonus
  peak_amplification = max(|ψ(x,t)|/a) over all x,t
  growth_rate        = relative increase in peak during last 20% of sim
  bonus              = 1.5× if final |ψ|_max/a > 1.5 (sustained focusing)

DO NOT MODIFY THIS FILE. Agent modifies simulate.py only.
"""

import numpy as np
import os, json, hashlib, time
from pathlib import Path

# ── Grid / solver constants ──────────────────────────────────────────────────
N               = 1024
L               = 20 * np.pi      # 10 MI wavelengths for a=1
DT              = 0.01
MAX_SIM_TIME    = 50.0
MAX_WALL_SECONDS = 120
RECORD_EVERY    = 20               # steps between metric snapshots
DEFAULT_A       = 1.0              # background amplitude

EXP_DIR = Path(__file__).parent / "experiments"
EXP_DIR.mkdir(exist_ok=True)

x  = np.linspace(0, L, N, endpoint=False)
kx = np.fft.fftfreq(N, d=L / N) * 2 * np.pi

# Precompute split-step dispersion operators
_disp_half = np.exp(-1j * kx**2 * DT / 2)
_disp_full = np.exp(-1j * kx**2 * DT)


# ── Solver ───────────────────────────────────────────────────────────────────

def nls_step(psi_hat):
    """
    One Strang-split step for i∂ₜψ + ∂ₓₓψ + |ψ|²ψ = 0.
    Dispersion:  ψ̂ → ψ̂ · exp(-ik²·dt/2)
    Nonlinear:   ψ  → ψ · exp(i|ψ|²·dt)   [exact, |ψ| conserved]
    """
    psi_hat = psi_hat * _disp_half
    psi = np.fft.ifft(psi_hat)
    psi = psi * np.exp(1j * np.abs(psi)**2 * DT)
    psi_hat = np.fft.fft(psi)
    psi_hat = psi_hat * _disp_half
    return psi_hat


# ── Metrics / scoring ────────────────────────────────────────────────────────

def compute_metrics(psi_hat, a, t, step, wall):
    psi = np.fft.ifft(psi_hat)
    amp = np.abs(psi)
    return {
        'time':          float(t),
        'step':          int(step),
        'wall_elapsed':  float(wall),
        'peak_amp':      float(amp.max()),
        'amplification': float(amp.max() / a),
        'mean_amp':      float(amp.mean()),
        'energy':        float(np.mean(amp**2)),
    }


def compute_score(metrics_history, a):
    if not metrics_history:
        return 0.0, {}

    amps = [m['amplification'] for m in metrics_history]
    peak_amp = max(amps)

    # Growth rate: change in mean amplification over last 20% vs prior 20%
    n = len(amps)
    late  = amps[max(0, int(0.8 * n)):]
    mid   = amps[max(0, int(0.6 * n)): max(0, int(0.8 * n))]
    egr = 0.0
    if mid and late and np.mean(mid) > 0:
        egr = max(0.0, (np.mean(late) - np.mean(mid)) / np.mean(mid))

    # Bonus: still at elevated amplitude at simulation end
    final_amp = amps[-1]
    bonus = 1.5 if final_amp > 1.5 else 1.0

    score = peak_amp * (1.0 + egr) * bonus
    return score, {
        'peak_amplification':        peak_amp,
        'final_amplification':       final_amp,
        'amplification_growth_rate': egr,
        'bonus':                     bonus,
    }


# ── Main simulation runner ────────────────────────────────────────────────────

def run_simulation(initial_condition_fn, a=DEFAULT_A, config=None):
    """
    Args:
        initial_condition_fn: f(x, a) -> complex array of shape (N,)
        a:    background amplitude
        config: dict of metadata for this experiment
    Returns:
        result dict with score and metrics
    """
    if config is None:
        config = {}

    ts  = time.strftime("%Y%m%d_%H%M%S")
    uid = hashlib.md5(f"{ts}{str(config)}".encode()).hexdigest()[:6]
    exp_id = f"{ts}_{uid}"

    config.update({'N': N, 'L': float(L), 'a': a,
                   'max_wall_seconds': MAX_WALL_SECONDS})

    print("=" * 60)
    print(f"Experiment {exp_id}")
    print(f"Config: {json.dumps({k: v for k, v in config.items()}, indent=2)}")
    print("=" * 60)
    print(f"{'Step':>8}  {'Time':>8}  {'Peak|ψ|':>10}  {'Amplif':>8}")
    print("-" * 60)

    psi      = initial_condition_fn(x, a).astype(complex)
    # Normalize IC to background power: mean(|ψ|²) = a²
    # This ensures score measures true focusing gain, not IC amplitude
    psi_rms  = np.sqrt(np.mean(np.abs(psi)**2))
    psi      = psi * (a / psi_rms)
    config['ic_rms_before_norm'] = float(psi_rms)
    psi_hat  = np.fft.fft(psi)

    metrics_history = []
    t          = 0.0
    wall_start = time.time()
    total_steps = int(MAX_SIM_TIME / DT)

    for step in range(total_steps):
        psi_hat = nls_step(psi_hat)
        t += DT
        wall = time.time() - wall_start

        if step % RECORD_EVERY == 0:
            m = compute_metrics(psi_hat, a, t, step, wall)
            metrics_history.append(m)
            if step % (RECORD_EVERY * 25) == 0:
                print(f"{step:8d}  {t:8.2f}  {m['peak_amp']:10.4f}  {m['amplification']:8.4f}")

        if wall > MAX_WALL_SECONDS:
            print(f"\nWall time limit at t={t:.2f}")
            break

    score, score_info = compute_score(metrics_history, a)

    print("\n" + "=" * 60)
    print(f"Done: {step} steps, t={t:.2f}, wall={time.time()-wall_start:.1f}s")
    print(f"Score: {score:.4f}")
    print(f"Peak amplification: {score_info.get('peak_amplification', 0):.4f}×")
    print(f"Final amplification: {score_info.get('final_amplification', 0):.4f}×")
    print(f"Bonus: {score_info.get('bonus', 1.0):.1f}×")
    print("=" * 60)

    # Best tracking
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
        'id':                        exp_id,
        'timestamp':                 time.strftime("%Y-%m-%dT%H:%M:%S"),
        'config':                    config,
        'score':                     score,
        'peak_amplification':        score_info.get('peak_amplification', 0),
        'final_amplification':       score_info.get('final_amplification', 0),
        'amplification_growth_rate': score_info.get('amplification_growth_rate', 0),
        'bonus':                     score_info.get('bonus', 1.0),
        'metrics_history':           metrics_history,
    }
    with open(EXP_DIR / f"{exp_id}.json", 'w') as f:
        json.dump(result, f)

    return result
