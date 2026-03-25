"""
KS Chaos Hunter — Experiment Configuration
==========================================
Agent modifies this file to run experiments.
DO NOT modify prepare.py.
"""

import numpy as np
from prepare import run_simulation, L, N

# ===========================================================================
# Initial condition library
# ===========================================================================

def single_mode_k11(x, k):
    """
    Single most-unstable mode: k_11 = 11*(2π/L) ≈ 0.688 ≈ k* = 1/√2.
    Linear growth rate σ(k_11) ≈ 0.248 (near maximum 0.25).
    Baseline: expect score ~ 5-15.
    """
    n = 11
    u = np.cos(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def multi_mode_unstable(x, k):
    """
    Sum of all 15 unstable modes (n=1..15) with equal amplitudes.
    Hypothesis: broader energy spread → richer nonlinear interactions →
    faster cascade to high-amplitude chaotic state.
    """
    u = np.zeros(N)
    for n in range(1, 16):
        u += np.cos(n * 2 * np.pi * x / L)
    u /= 15  # normalize so max ~ 1
    return np.fft.rfft(u)


def phase_optimized_k11(x, k):
    """
    Most-unstable mode + second harmonic (k_22, at the nonlinear resonance).
    KS cascade: k* → 2k* is the first nonlinear step. Pre-seeding 2k*
    with phase aligned to the cascade might delay the cascade, extending
    the linear growth phase and improving the bonus.
    eps controls the second harmonic amplitude.
    """
    n1 = 11  # most unstable: k ≈ 0.688
    n2 = 22  # second harmonic: k ≈ 1.375 (stable, so acts as energy sink seed)
    eps = 0.1
    u = np.cos(n1 * 2 * np.pi * x / L) + eps * np.cos(n2 * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def peak_modes_sum(x, k):
    """
    Five modes near k* (n=9,10,11,12,13) with unit amplitude.
    Hypothesis: these modes have growth rates within 5% of maximum.
    Their in-phase sum creates a spatially localized peak that grows
    more coherently than a broad-spectrum IC.
    """
    u = np.zeros(N)
    for n in [9, 10, 11, 12, 13]:
        u += np.cos(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def sawtooth_profile(x, k):
    """
    Sawtooth wave approximated by first 15 Fourier modes.
    Sawtooth = sum_n sin(n*x)/n, which concentrates energy at low n
    (near the unstable band). Physical motivation: shock-like profiles
    are the 'natural' structures of hyperbolic PDEs; maybe KS also
    prefers shock-like transients.
    """
    u = np.zeros(N)
    for n in range(1, 16):
        u += (1.0 / n) * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def gaussian_bump(x, k):
    """
    Localized Gaussian bump: u = A * exp(-(x - L/2)^2 / (2*sigma^2)).
    Localized ICs provide a broad spectrum that includes all unstable modes.
    The bump then 'explodes' into the chaotic attractor, potentially with
    larger transients than a single-mode IC.
    sigma = L/20 gives localization width ~ 5% of domain.
    """
    sigma = L / 20
    u = np.exp(-((x - L / 2) ** 2) / (2 * sigma ** 2))
    return np.fft.rfft(u)


def optimal_linear_growth(x, k):
    """
    IC designed to maximize LINEAR growth rate at short times.
    Each mode is given amplitude proportional to its growth rate σ(k_n) = k_n^2 - k_n^4
    (for unstable modes), which weights the IC towards modes that grow fastest.
    Weight = max(σ(k), 0) for modes n=1..15.
    """
    u = np.zeros(N)
    for n in range(1, 16):
        kn = n * 2 * np.pi / L
        sigma_n = kn**2 - kn**4
        if sigma_n > 0:
            u += sigma_n * np.cos(n * 2 * np.pi * x / L)
    u /= np.max(np.abs(u)) + 1e-10  # normalize to max amplitude 1
    return np.fft.rfft(u)


# ===========================================================================
# Current experiment
# ===========================================================================

CURRENT_CONFIG = {
    "name": "single_mode_k11_baseline",
    "description": "Baseline: single most-unstable mode (n=11, k≈0.688≈k*). "
                   "Establishes baseline score for comparison. Linear growth rate σ≈0.248.",
    "initial_condition": "single_mode_k11",
}

def sharp_spike(x, k):
    """
    Very narrow Gaussian spike sigma=L/80 (much sharper than gaussian_bump L/20).
    HYPOTHESIS: sharper localization → broader unstable-band coverage → larger
    transient before collapse. If gaussian_bump (5.31) benefits from localization,
    a sharper spike should do even better.
    """
    sigma = L / 80
    u = np.exp(-((x - L / 2) ** 2) / (2 * sigma ** 2))
    return np.fft.rfft(u)


def sawtooth_steep(x, k):
    """
    Sawtooth with 1/n^0.5 amplitude decay (less steep than 1/n).
    HYPOTHESIS: sawtooth_profile (1/n) scores 6.07. The 1/n decay emphasizes
    low-k modes. With 1/n^0.5, more energy is in high-k modes (k~1), which
    are near the instability boundary. This might create a steeper shock front
    that triggers larger transient spikes.
    """
    u = np.zeros(N)
    for n in range(1, 16):
        u += (1.0 / np.sqrt(n)) * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def sawtooth_sharp(x, k):
    """
    Sawtooth with 1/n^2 amplitude decay (steeper than 1/n).
    Counter-hypothesis: concentrate energy at the LOWEST k modes (large scale)
    which have the slowest growth but longest coherence time → sustained growth
    throughout the window → high egr + bonus.
    """
    u = np.zeros(N)
    for n in range(1, 16):
        u += (1.0 / n**2) * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def double_bump(x, k):
    """
    Two Gaussian bumps separated by L/2, opposite signs.
    HYPOTHESIS: creates a dipole structure → two interacting shock fronts →
    nonlinear interaction between bumps produces larger amplitude excursions
    than a single bump (gaussian_bump: 5.31).
    sigma = L/20 (same as gaussian_bump).
    """
    sigma = L / 20
    u = (np.exp(-((x - L / 4) ** 2) / (2 * sigma ** 2))
         - np.exp(-((x - 3 * L / 4) ** 2) / (2 * sigma ** 2)))
    return np.fft.rfft(u)


def sawtooth_multiscale(x, k):
    """
    Sawtooth at k=1 PLUS sawtooth at k=2 (period L/2), eps=0.3.
    HYPOTHESIS: mimics the NS multi-scale approach. The k=2 sawtooth seeds
    the secondary cascade. If sawtooth structure is optimal at scale L,
    then seeding at L/2 too might accelerate the cascade non-linearly.
    """
    u = np.zeros(N)
    for n in range(1, 16):
        u += (1.0 / n) * np.sin(n * 2 * np.pi * x / L)
    # Add k=2 sawtooth (period L/2)
    eps = 0.3
    for n in range(1, 16):
        u += eps * (1.0 / n) * np.sin(n * 4 * np.pi * x / L)
    return np.fft.rfft(u)


def step_function(x, k):
    """
    Pure step function: u = +1 for x < L/2, -1 for x > L/2.
    The sawtooth sum_n sin(nx)/n converges to this. Here we directly
    use the step function, which has exact 1/n Fourier spectrum up to
    all k (not truncated at n=15). HYPOTHESIS: full-spectrum step function
    > truncated sawtooth because all unstable modes are seeded simultaneously
    with correct 1/n weights.
    """
    u = np.where(x < L / 2, 1.0, -1.0)
    return np.fft.rfft(u)


def cosine_sawtooth(x, k):
    """
    Cosine sum: u = sum_{n=1}^{15} cos(nx)/n.
    HYPOTHESIS: The sin-sawtooth scores 6.07. Does the phase (sin vs cos) matter?
    A cosine sum is the 'periodic ramp' function — different structure from sawtooth.
    If score is phase-insensitive (as expected from translation invariance), this
    should score similarly. If different, phase alignment matters.
    """
    u = np.zeros(N)
    for n in range(1, 16):
        u += (1.0 / n) * np.cos(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def boosted_sawtooth(x, k):
    """
    Sawtooth 1/n but with extra weight on the most-unstable modes (n=10,11,12).
    HYPOTHESIS: sawtooth success is partly from 1/n spectrum, partly from
    the shock structure. Boosting near-peak-growth modes (n=10..12) should
    accelerate the initial linear phase while preserving the overall structure.
    Boost factor = 2x at n=10,11,12.
    """
    u = np.zeros(N)
    for n in range(1, 16):
        amp = 1.0 / n
        if n in (10, 11, 12):
            amp *= 2.0
        u += amp * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def sawtooth_n20(x, k):
    """
    Sawtooth extended to n=1..20 (includes some stable modes n=16..20, k=1.0..1.25).
    HYPOTHESIS: adding small stable-mode content seeds the energy sinks from t=0,
    which might paradoxically stabilize the large-scale structure longer, giving
    more sustained growth. Counter-intuitive but worth testing.
    n=16..20: k=1.0..1.25, mildly stable.
    """
    u = np.zeros(N)
    for n in range(1, 21):
        u += (1.0 / n) * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def odd_sawtooth(x, k):
    """
    Odd-harmonics-only: u = sum_{n=1,3,5,...,15} sin(nx)/n (square wave approx).
    A square wave is a step-function analog. HYPOTHESIS: odd-only spectrum
    gives a sharper spatial transition than the full sawtooth, potentially
    triggering more intense nonlinear coupling at the jump.
    """
    u = np.zeros(N)
    for n in range(1, 16, 2):  # odd n: 1, 3, 5, ..., 15
        u += (1.0 / n) * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def sawtooth_n8(x, k):
    """
    Sawtooth truncated at n=8 (only modes near and below the most unstable k*).
    HYPOTHESIS: n=1..8 covers k=1/16..0.5, all safely in unstable band.
    Lower-k modes have LONGER coherence time. Maybe concentrating in
    the slowest-growing modes gives more sustained growth → higher egr + bonus.
    """
    u = np.zeros(N)
    for n in range(1, 9):
        u += (1.0 / n) * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def square_wave(x, k):
    """
    Full square wave: u = sign(sin(2π x / L)).
    This is the exact sum of all odd harmonics with 1/n weights (truncated at
    dealiasing cutoff n~170). HYPOTHESIS: if odd-sawtooth (n=1..15 odd) at 6.61
    is better than full sawtooth (n=1..15), then a FULL square wave (more odd
    modes, still all with correct 1/n weights) should be even better.
    Modes in stable band (k>1) will be dealiased out; the unstable-band content
    remains and is perfectly weighted.
    """
    u = np.sign(np.sin(2 * np.pi * x / L))
    return np.fft.rfft(u)


def odd_sawtooth_equal(x, k):
    """
    Odd harmonics n=1,3,...,15 with EQUAL amplitude (not 1/n).
    HYPOTHESIS: maybe the 1/n weighting in odd_sawtooth is not optimal.
    Equal amplitude at all odd modes concentrates more energy near k* (n=11).
    If the growth-rate weighting matters, equal-amplitude might beat 1/n.
    """
    u = np.zeros(N)
    for n in range(1, 16, 2):
        u += np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def odd_sawtooth_sqrtn(x, k):
    """
    Odd harmonics n=1,3,...,15 with 1/sqrt(n) amplitude (intermediate between 1/n and equal).
    HYPOTHESIS: systematically tests whether the amplitude decay exponent matters.
    Current best: 1/n (odd_sawtooth, 6.61). Equal (odd_sawtooth_equal) tests other extreme.
    1/sqrt(n) is in between — may combine the square-wave structure advantage with
    better mode-weighting near k*.
    """
    u = np.zeros(N)
    for n in range(1, 16, 2):
        u += (1.0 / np.sqrt(n)) * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def odd_sawtooth_boosted(x, k):
    """
    Odd-sawtooth 1/n but with 2x boost at n=11 (most unstable odd mode).
    HYPOTHESIS: the most unstable odd mode n=11 (k=0.688 ≈ k*) drives the
    early transient. Boosting it within the square-wave structure might give
    fastest initial growth while preserving the antisymmetric shock structure.
    """
    u = np.zeros(N)
    for n in range(1, 16, 2):
        amp = 1.0 / n
        if n == 11:
            amp *= 2.0
        u += amp * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def double_square_wave(x, k):
    """
    Two square waves at period L and period L/2 combined.
    HYPOTHESIS: analogous to the TG multi-scale approach in NS. The period-L
    square wave gives the base structure; the period-L/2 wave seeds the cascade
    at higher k. Amplitude of second wave eps=0.3.
    """
    u = np.sign(np.sin(2 * np.pi * x / L))
    eps = 0.3
    u += eps * np.sign(np.sin(4 * np.pi * x / L))
    return np.fft.rfft(u)


def odd_fast_only(x, k):
    """
    Odd modes restricted to fast-growing: n=7,9,11,13,15 (all with σ > 0.08).
    Drops slow modes n=1,3,5 (σ < 0.09) which contribute amplitude but little growth.
    HYPOTHESIS: removing slow modes concentrates the IC energy near k* → faster
    initial growth → higher peak before saturation.
    Amplitudes: 1/n (same as odd_sawtooth).
    """
    u = np.zeros(N)
    for n in [7, 9, 11, 13, 15]:
        u += (1.0 / n) * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def odd_growthrate_weighted(x, k):
    """
    Odd modes n=1..15 with amplitude ∝ σ(k_n) = k_n² - k_n⁴ (linear growth rate).
    HYPOTHESIS: natural weighting by growth rate concentrates energy where it
    grows fastest (near k*=1/√2, n≈11). More principled than ad hoc 1/n.
    Growth rates: σ(n=1)≈0.004, σ(n=11)≈0.249, σ(n=13)≈0.224, etc.
    """
    u = np.zeros(N)
    for n in range(1, 16, 2):
        kn = n * 2 * np.pi / L
        sigma = kn**2 - kn**4
        if sigma > 0:
            u += sigma * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def odd_top3(x, k):
    """
    Top-3 fastest odd modes: n=9, 11, 13. Growth rates σ = 0.216, 0.249, 0.224.
    Equal amplitude.
    HYPOTHESIS: three near-optimal modes with equal amplitude should give
    constructive interference at the highest-growth part of the spectrum.
    Tests whether fewer, better-targeted modes beat many modes with 1/n weights.
    """
    u = np.zeros(N)
    for n in [9, 11, 13]:
        u += np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def odd_sawtooth_n9start(x, k):
    """
    Odd sawtooth starting from n=3: n=3,5,7,9,11,13,15 with 1/n.
    Drops n=1 (slowest-growing, σ≈0.004 — basically stable on simulation timescale).
    HYPOTHESIS: n=1 contributes amplitude (large scale) but almost zero growth.
    After normalization, keeping n=1 dilutes the energy in fast modes.
    Dropping it should give a more efficiently growing IC.
    """
    u = np.zeros(N)
    for n in range(3, 16, 2):
        u += (1.0 / n) * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def odd_sawtooth_eps(x, k):
    """
    Odd sawtooth (n=1..15, 1/n) + small eps=0.27 seed at the even modes n=10,12
    (the 'natural daughters' of n=11 in the nonlinear cascade).
    HYPOTHESIS: in NS, seeding the nonlinear daughter modes accelerated the cascade.
    Here: n=11 → n=10+1, n=12-1 via 3-wave resonance. Pre-seeding these even modes
    with the right phase might extend the coherent growth phase.
    eps=0.27 matches the optimal NS seed amplitude (testing cross-domain transfer).
    """
    u = np.zeros(N)
    for n in range(1, 16, 2):
        u += (1.0 / n) * np.sin(n * 2 * np.pi * x / L)
    eps = 0.27
    for n in [10, 12]:
        u += eps * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def two_period_square(x, k):
    """
    2-period square wave: modes n=2,6,10,14 (= 2×{1,3,5,7}) with 1/(n/2) weights.
    Creates TWO positive-negative cycles on [0,L]. Each cycle has width L/4≈25.
    Fundamental k=2/16=0.125 — still unstable (σ≈0.016).
    HYPOTHESIS: smaller spatial scale → faster cascade → higher peak before saturation.
    """
    u = np.zeros(N)
    for j, n in enumerate([2, 6, 10, 14], 1):
        u += (1.0 / (2 * j - 1)) * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def three_period_square(x, k):
    """
    3-period square wave: modes n=3,9,15 (= 3×{1,3,5}) with 1/{1,3,5} weights.
    Creates THREE cycles on [0,L]. Fundamental k=3/16≈0.188.
    HYPOTHESIS: tests whether spatial scale matters within the unstable band.
    n=15 is the highest odd unstable mode — near-boundary effects may be interesting.
    """
    u = np.zeros(N)
    for j, n in enumerate([3, 9, 15], 1):
        u += (1.0 / (2 * j - 1)) * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def odd_sawtooth_n1p5(x, k):
    """
    Odd modes n=1..15 with 1/n^1.5 amplitude decay (between 1/n and 1/n^2).
    HYPOTHESIS: 1/n beats 1/n^2 and 1/sqrt(n). Maybe 1/n^1.5 is better or
    similar — fine-grained scan of the amplitude exponent near the optimum.
    """
    u = np.zeros(N)
    for n in range(1, 16, 2):
        u += (1.0 / n**1.5) * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def odd_sawtooth_n0p75(x, k):
    """
    Odd modes n=1..15 with 1/n^0.75 amplitude decay.
    Fine-grained scan between 1/n (=1/n^1.0) and 1/sqrt(n) (=1/n^0.5).
    """
    u = np.zeros(N)
    for n in range(1, 16, 2):
        u += (1.0 / n**0.75) * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


def odd_sawtooth_anti(x, k):
    """
    Odd sawtooth with alternating signs: sin(x)/1 - sin(3x)/3 + sin(5x)/5 - ...
    This is the Leibniz series for π/4 × arctan(sin(x)) — a different spatial
    structure from the standard square wave. HYPOTHESIS: might create a more
    localized spike structure that triggers larger transients.
    """
    u = np.zeros(N)
    for i, n in enumerate(range(1, 16, 2)):
        sign = (-1) ** i
        u += sign * (1.0 / n) * np.sin(n * 2 * np.pi * x / L)
    return np.fft.rfft(u)


IC_MAP = {
    "single_mode_k11": single_mode_k11,
    "multi_mode_unstable": multi_mode_unstable,
    "phase_optimized_k11": phase_optimized_k11,
    "peak_modes_sum": peak_modes_sum,
    "sawtooth_profile": sawtooth_profile,
    "gaussian_bump": gaussian_bump,
    "optimal_linear_growth": optimal_linear_growth,
    "sharp_spike": sharp_spike,
    "sawtooth_steep": sawtooth_steep,
    "sawtooth_sharp": sawtooth_sharp,
    "double_bump": double_bump,
    "sawtooth_multiscale": sawtooth_multiscale,
    "step_function": step_function,
    "cosine_sawtooth": cosine_sawtooth,
    "boosted_sawtooth": boosted_sawtooth,
    "sawtooth_n20": sawtooth_n20,
    "odd_sawtooth": odd_sawtooth,
    "sawtooth_n8": sawtooth_n8,
    "square_wave": square_wave,
    "odd_sawtooth_equal": odd_sawtooth_equal,
    "odd_sawtooth_sqrtn": odd_sawtooth_sqrtn,
    "odd_sawtooth_boosted": odd_sawtooth_boosted,
    "double_square_wave": double_square_wave,
    "odd_fast_only": odd_fast_only,
    "odd_growthrate_weighted": odd_growthrate_weighted,
    "odd_top3": odd_top3,
    "odd_sawtooth_n9start": odd_sawtooth_n9start,
    "odd_sawtooth_eps": odd_sawtooth_eps,
    "two_period_square": two_period_square,
    "three_period_square": three_period_square,
    "odd_sawtooth_n1p5": odd_sawtooth_n1p5,
    "odd_sawtooth_n0p75": odd_sawtooth_n0p75,
    "odd_sawtooth_anti": odd_sawtooth_anti,
}

CURRENT_CONFIG = {
    "name": "sharp_spike",
    "description": "Narrow Gaussian spike sigma=L/80. HYPOTHESIS: sharper localization "
                   "→ broader unstable-band coverage → larger transient. Best so far: sawtooth 6.07.",
    "initial_condition": "sharp_spike",
}

if __name__ == "__main__":
    ic_fn = IC_MAP[CURRENT_CONFIG["initial_condition"]]
    result = run_simulation(
        ic_fn,
        config=CURRENT_CONFIG.copy(),
    )
    print(f"\nFinal score: {result['score']:.4f}")
