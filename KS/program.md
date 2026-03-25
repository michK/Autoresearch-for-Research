# Kuramoto-Sivashinsky Chaos Hunter: Agent Instructions

You are an autonomous research agent hunting for initial conditions that produce extreme transient amplitude growth in the 1D Kuramoto-Sivashinsky (KS) equation on a periodic domain. This is the second domain in a methodology study showing that low-effort AI-in-the-loop auto-research is effective across different PDE systems.

## Background

The KS equation models spatiotemporal chaos:
    ∂_t u + u ∂_x u + ∂_xx u + ∂_xxxx u = 0

on a periodic domain [0, L] with L = 32π. Unlike Navier-Stokes, KS solutions do NOT blow up — they settle onto a strange chaotic attractor. Our goal is to find initial conditions that maximize **transient amplitude growth** before reaching the attractor. This is the KS analog of the NS vorticity amplification problem.

## Scoring

The score mirrors the NS problem exactly:

    score = (peak_max_u / initial_max_u) × (1 + energy_growth_rate) × [1.5 if bonus]

- **peak_max_u**: maximum of ||u||_∞ over the run
- **initial_max_u**: ||u||_∞ at the first timestep
- **energy_growth_rate**: slope of log(energy) vs time (linear fit; 0 if negative)
- **bonus ×1.5**: if ||u||_∞ is still growing at wall clock end (analogous to NS bonus)

## Key physics for the agent

### Linear instability
- KS linear growth rate per mode: σ(k) = k² - k⁴
- Unstable range: k ∈ (0, 1) — 15 unstable modes for L = 32π
- Most unstable mode: k* = 1/√2 ≈ 0.707 (mode n=11 for this L)
- Stable range: k > 1 (these modes drain energy)

### What drives amplitude growth
- **Seeding the most unstable modes** (k near k* = 0.707) gives fastest linear growth
- **Phase alignment**: modes that constructively interfere at early times can delay the onset of chaos, allowing more transient growth
- **Multi-mode initial conditions**: seeding multiple unstable modes simultaneously can create complex transient dynamics not captured by single-mode analysis
- **Amplitude matters**: too-large ICs immediately go nonlinear (shorter growth phase); too-small ICs grow but the score requires initial_max_u to be non-negligible
- **The bonus condition**: timing the IC so the cascade is still growing at the 60s wall clock is worth ×1.5

### What suppresses growth
- **Non-resonant phases**: modes that destructively interfere at early times reduce amplitude
- **Energy at stable modes** (k > 1): immediately dissipated, wastes the IC "budget"
- **Too many modes**: spreads energy, reduces peak per-mode amplitude
- **Perfect symmetry**: u(-x) = -u(x) type antisymmetric ICs conserve certain invariants that prevent the largest excursions

### Key numbers
- L = 32π ≈ 100.5
- dx = L/512 ≈ 0.196
- Unstable mode wavenumbers: k_n = n*(2π/L) = n/16 for n = 1,...,15
- Most unstable: n = 11, k ≈ 0.688
- Fastest growth rate: σ(k*) = (1/√2)² - (1/√2)⁴ = 0.5 - 0.25 = 0.25

## Your workflow

### Setup (once per session)
1. Read `program.md` (this file)
2. Read `prepare.py` to understand the solver and scoring
3. Read `simulate.py` to see current configuration
4. Check `best.json` and `leaderboard.json`

### Experiment loop
1. **Analyze**: leaderboard + recent results. What scored highest? What patterns?
2. **Hypothesize**: form a specific hypothesis. Write it as a comment.
3. **Implement**: add a new IC function to `simulate.py` and set `CURRENT_CONFIG`
4. **Run**: `python simulate.py`
5. **Evaluate**: did score improve? Check bonus condition, growth rates.
6. **Repeat**.

## Initial condition design

Your IC function signature: `def my_ic(x, k): → u_hat`

You return spectral coefficients `u_hat = np.fft.rfft(u)` where `u` is the physical-space IC. Rules:
- Set `u_hat[0] = 0` (zero mean enforced by harness anyway)
- Concentrate energy at unstable modes (k < 1) for efficiency
- The harness applies 2/3 dealiasing automatically

Examples of what to try:
- **Single mode at k***: `u = A * cos(11 * 2π x / L + phi)`
- **Sum of unstable modes**: seed n=8,9,10,11,12 with carefully chosen phases
- **Sawtooth / shock profiles**: concentrate energy in a shock-like structure
- **Random phases at unstable modes**: explore the attractor basin
- **Spatially localized bumps**: Gaussian or similar localized structures
- **Combinations**: mix a large-amplitude most-unstable mode with small perturbations

## Interpreting results
- **Score < 5**: weak growth, poor IC design
- **Score 5-20**: typical single-mode growth to attractor
- **Score 20-50**: good multi-mode or phase-tuned IC
- **Score > 50**: excellent — sustained transient growth
- **Score > 100**: exceptional — potentially publishable finding for this system

## File structure
```
prepare.py    -- fixed harness (DO NOT MODIFY)
simulate.py   -- your experiment file (MODIFY THIS)
program.md    -- these instructions
best.json     -- current best
leaderboard.json -- top 20
experiments/  -- all logs
```

Good hunting.
