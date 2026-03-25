"""
Gray-Scott Pattern Hunter — Experiment Configuration
=====================================================
Agent modifies this file to run experiments.
DO NOT modify prepare.py.
"""

import numpy as np
from prepare import run_simulation, N, L

# ===========================================================================
# Initial condition library
# ===========================================================================

def center_seed(x, y):
    """
    Standard baseline: uniform u=1, v=0 with a circular seed of v=0.25
    at the center (radius 10). Classic Gray-Scott starting condition.
    """
    u = np.ones((N, N))
    v = np.zeros((N, N))
    cx, cy = L / 2, L / 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = r < 10
    u[mask] = 0.5
    v[mask] = 0.25
    return u, v


def multi_seed(x, y):
    """
    Four circular seeds at quadrant centers.
    HYPOTHESIS: multiple nucleation sites → pattern competition → higher
    spectral entropy than a single central seed.
    """
    u = np.ones((N, N))
    v = np.zeros((N, N))
    for cx, cy in [(L/4, L/4), (3*L/4, L/4), (L/4, 3*L/4), (3*L/4, 3*L/4)]:
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = r < 8
        u[mask] = 0.5
        v[mask] = 0.25
    return u, v


def random_noise_seed(x, y):
    """
    Uniform u=1, v=0 + low-amplitude random noise on both fields.
    HYPOTHESIS: noise triggers simultaneous nucleation everywhere → patterns
    form at all locations at once → maximum possible spectral coverage.
    Noise amplitude 0.05 on u, 0.05 on v.
    """
    rng = np.random.RandomState(42)
    u = 1.0 - 0.05 * rng.random((N, N))
    v = 0.05 * rng.random((N, N))
    return u, v


def stripe_seed(x, y):
    """
    Horizontal stripe of v across the center (width 10).
    HYPOTHESIS: anisotropic seed biases toward stripe/labyrinthine patterns
    which may have higher spectral entropy than isotropic spots.
    """
    u = np.ones((N, N))
    v = np.zeros((N, N))
    mask = np.abs(x - L / 2) < 5
    u[mask] = 0.5
    v[mask] = 0.25
    return u, v


def large_seed(x, y):
    """
    Large circular seed (radius 25, ~40% of domain width).
    HYPOTHESIS: bigger seed → more surface area for instability to develop →
    faster pattern formation → more time for complexity to grow within budget.
    """
    u = np.ones((N, N))
    v = np.zeros((N, N))
    cx, cy = L / 2, L / 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = r < 25
    u[mask] = 0.5
    v[mask] = 0.25
    return u, v


# ===========================================================================
# Current experiment
# ===========================================================================

CURRENT_CONFIG = {
    "name": "best_stripe_k0578",
    "description": "Best config: center seed at stripe-regime params F=0.025, k=0.0578. "
                   "Earns 1.5x bonus (still-evolving Turing patterns). Score=5.082 (7x baseline).",
    "initial_condition": "center_seed",
    "F": 0.025,
    "k": 0.0578,
}

IC_MAP = {
    "center_seed": center_seed,
    "multi_seed": multi_seed,
    "random_noise_seed": random_noise_seed,
    "stripe_seed": stripe_seed,
    "large_seed": large_seed,
}

if __name__ == "__main__":
    ic_fn = IC_MAP[CURRENT_CONFIG["initial_condition"]]
    result = run_simulation(
        ic_fn,
        F=CURRENT_CONFIG.get("F", 0.035),
        k=CURRENT_CONFIG.get("k", 0.065),
        config=CURRENT_CONFIG.copy(),
    )
    print(f"\nFinal score: {result['score']:.4f}")
