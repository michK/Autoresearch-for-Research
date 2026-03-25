"""
NLS Rogue Wave Hunter — Experiment Configuration
=================================================
Agent modifies this file to run experiments.
DO NOT modify prepare.py.

Physics reference:
  i∂ₜψ + ∂ₓₓψ + |ψ|²ψ = 0,  background ψ₀ = a·exp(ia²t)
  MI band: |k| < a√2,  most unstable k* = a
  Akhmediev breather: periodic in x, peak < 3a
  Peregrine soliton:  rational, peak = 3a  (THEORETICAL MAX for order-1)
  Order-n rogue wave: peak = (2n+1)·a  (5a, 7a, ... for higher orders)
"""

import numpy as np
from prepare import run_simulation, x, L, N, DEFAULT_A

a = DEFAULT_A   # background amplitude = 1.0
# MI band: k ∈ (0, a√2) ≈ (0, 1.414)
# Most unstable wavenumber: k* = a = 1.0
# In terms of mode index n: k_n = n·2π/L, so n* = a·L/(2π) = 10

# ── Initial condition library ────────────────────────────────────────────────

def plane_wave_baseline(x, a):
    """
    Unperturbed plane wave. Should not amplify — score = 1.0.
    Sanity check that the solver conserves |ψ|.
    """
    return a * np.ones(len(x), dtype=complex)


def single_mode_optimal(x, a):
    """
    Plane wave + small perturbation at most-unstable MI wavenumber k*=a.
    HYPOTHESIS: single optimal mode focuses into Akhmediev breather, peak ~2.9a.
    """
    eps = 0.01
    k_opt = a  # most unstable wavenumber
    return a * (1 + eps * np.cos(k_opt * x)).astype(complex)


def single_mode_cos(x, a):
    """
    Cosine perturbation at k* with larger amplitude to test faster focusing.
    HYPOTHESIS: larger eps → faster MI → higher peak within time budget.
    """
    eps = 0.05
    k_opt = a
    return a * (1 + eps * np.cos(k_opt * x)).astype(complex)


def multi_mode_in_band(x, a):
    """
    Perturbation spanning all unstable modes with equal amplitude.
    HYPOTHESIS: many modes competing → chaotic focusing, potential for
    super-Peregrine peaks via random constructive interference.
    """
    eps = 0.005
    psi = np.ones(len(x), dtype=complex) * a
    rng = np.random.RandomState(42)
    for n in range(1, 15):
        k = n * 2 * np.pi / L
        if k < a * np.sqrt(2):
            phi = rng.uniform(0, 2 * np.pi)
            psi = psi + a * eps * np.cos(k * x + phi)
    return psi


def peregrine_ic(x, a):
    """
    Approximate Peregrine soliton initial condition at t = -T (before peak).
    Peregrine: ψ_P(x,t) = a·exp(ia²t)·[1 - 4(1+2ia²t)/(1+4a²x²+4a⁴t²)]
    Peak at t=0: |ψ| = 3a. Start at t=-8 → small perturbation on background.
    HYPOTHESIS: seeding the exact rational solution IC → clean 3a peak.
    """
    t0 = -8.0  # start time (well before the peak at t=0)
    numerator   = 4 * (1 + 2j * a**2 * t0)
    denominator = 1 + 4 * a**2 * x**2 + 4 * a**4 * t0**2
    psi = a * np.exp(1j * a**2 * t0) * (1 - numerator / denominator)
    return psi


def akhmediev_ic(x, a):
    """
    Akhmediev breather initial condition at t=0 (growing phase).
    ψ_AB(x,t) = a·exp(ia²t)·[cos(δ-iφ) - ν·cos(κx)] / [1 - ν·cos(κx)·cos(δ)]
    where ν = κ/(a√2), δ = a²√(1-ν²)·t, etc.
    Using κ = a (most unstable mode, ν = 1/√2):
    HYPOTHESIS: exact breather IC → clean, predictable focusing cycle.
    """
    kappa = a           # spatial frequency
    nu    = kappa / (a * np.sqrt(2))   # = 1/√2
    delta_rate = a**2 * np.sqrt(1 - nu**2)
    t0 = 0.0
    delta = delta_rate * t0

    numerator   = np.cosh(delta - 1j * np.arccos(nu)) - nu * np.cos(kappa * x)
    denominator = np.cosh(delta) - nu * np.cos(kappa * x)
    psi = a * np.exp(1j * a**2 * t0) * (numerator / denominator)
    return psi


def perg_akm_k072_ph135(x, a):
    """
    BEST: Peregrine soliton at t0=-4 + phase-optimised Akhmediev perturbation.
    Peregrine: exact rational solution seeded before its peak (t0=-4).
    Akhmediev: κ=0.72 (sub-MI-band), phase=135° (constructive interference).
    Score=7.71, peak=4.85× background (super-Peregrine via MI + focusing synergy).
    DISCOVERED by auto-research loop (power-normalised metric, round 3).
    """
    t0    = -4.0
    kappa = 0.72
    phase = 135 * np.pi / 180
    # Peregrine rational IC
    num_p   = 4 * (1 + 2j * a**2 * t0)
    den_p   = 1 + 4 * a**2 * x**2 + 4 * a**4 * t0**2
    perg    = a * np.exp(1j * a**2 * t0) * (1 - num_p / den_p)
    # Akhmediev perturbation (background-subtracted)
    nu      = min(kappa / (a * np.sqrt(2)), 0.9999)
    num_a   = np.cosh(-1j * np.arccos(nu)) - nu * np.cos(kappa * x + phase)
    den_a   = np.cosh(0.0)                 - nu * np.cos(kappa * x + phase)
    akm     = a * (num_a / den_a)
    pert    = akm - a * np.ones_like(x, dtype=complex)
    return perg + pert


# ── Current experiment ────────────────────────────────────────────────────────

CURRENT_CONFIG = {
    "name":                "perg_akm_k072_ph135",
    "description":         "BEST: Peregrine t0=-4 + Akhmediev kappa=0.72 phase=135°. "
                           "Score=7.71, peak=4.85× (super-Peregrine, auto-research discovery).",
    "initial_condition":   "perg_akm_k072_ph135",
}

IC_MAP = {
    "plane_wave_baseline":    plane_wave_baseline,
    "single_mode_optimal":    single_mode_optimal,
    "single_mode_cos":        single_mode_cos,
    "multi_mode_in_band":     multi_mode_in_band,
    "peregrine_ic":           peregrine_ic,
    "akhmediev_ic":           akhmediev_ic,
    "perg_akm_k072_ph135":    perg_akm_k072_ph135,
}

if __name__ == "__main__":
    ic_fn  = IC_MAP[CURRENT_CONFIG["initial_condition"]]
    result = run_simulation(ic_fn, a=a, config=CURRENT_CONFIG.copy())
    print(f"\nFinal score: {result['score']:.4f}")
