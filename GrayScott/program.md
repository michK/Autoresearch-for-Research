# Gray-Scott Pattern Hunter: Agent Instructions

You are an autonomous research agent hunting for initial conditions and parameters that produce the most complex patterns in the 2D Gray-Scott reaction-diffusion system. This is the third domain in a methodology study showing that low-effort AI-in-the-loop auto-research is effective across different PDE systems.

## Background

The Gray-Scott model describes two chemicals u (substrate) and v (activator):

    ∂u/∂t = Du ∇²u - uv² + F(1-u)
    ∂v/∂t = Dv ∇²v + uv² - (F+k)v

- **F** (feed rate): how fast substrate u is replenished
- **k** (kill rate): how fast activator v decays
- **Du, Dv**: diffusion coefficients (Du > Dv creates Turing instability)

The system produces an extraordinary variety of patterns depending on (F, k):
spots, stripes, spirals, self-replicating structures, coral, fingerprints,
and spatiotemporal chaos. Our goal is to find the IC + parameter combination
that produces the **most complex** pattern structures.

## Scoring

    score = peak_spectral_entropy × peak_contrast × (1 + entropy_growth_rate) × [1.5 bonus]

- **spectral_entropy**: Shannon entropy of the v power spectrum (excluding DC).
  High = energy spread across many spatial frequencies = complex pattern.
- **pattern_contrast**: max(v) - min(v). High = visible, distinct patterns.
- **entropy_growth_rate**: slope of spectral_entropy vs time. Rewards sustained complexification.
- **bonus ×1.5**: if spectral_entropy still increasing at wall clock end (patterns still evolving).

## Key physics for the agent

### The (F, k) phase diagram
This is the MOST IMPORTANT parameter space to explore. Different regions produce:
- **F ≈ 0.01-0.02, k ≈ 0.04-0.05**: traveling waves, spirals
- **F ≈ 0.02-0.04, k ≈ 0.05-0.06**: stripes, labyrinthine patterns
- **F ≈ 0.03-0.04, k ≈ 0.06-0.065**: spots, self-replicating spots ("mitosis")
- **F ≈ 0.04-0.06, k ≈ 0.06-0.065**: sparse spots, decay
- **F ≈ 0.02-0.03, k ≈ 0.05-0.055**: dense worm-like patterns ("coral")
- **Boundaries between regions**: often the most complex, combining pattern types

### What drives pattern complexity
- **Turing instability**: Du > Dv is essential. Standard: Du=0.16, Dv=0.08.
- **Near bifurcation boundaries**: patterns are most complex at transitions between regimes.
- **Multiple pattern types coexisting**: if spots AND stripes form simultaneously, entropy is high.
- **Spatiotemporal chaos**: some (F,k) regions produce chaotic dynamics → maximum spectral entropy.

### What suppresses complexity
- **Too high F**: substrate floods the system, v decays → uniform u=1, v=0. Score ≈ 0.
- **Too low F**: insufficient feed, v dies out. Score ≈ 0.
- **Wrong ratio Du/Dv**: Turing instability requires Du/Dv > 1 (typically ~2). If too close to 1, no patterns.
- **Too small perturbation**: patterns take forever to nucleate → never reach complexity within wall clock.
- **Too symmetric IC**: may lock system into simple modes. Breaking symmetry helps.

### Initial condition design
Standard IC: u=1, v=0 everywhere, plus a local perturbation where u is lowered and v is raised. Variations to try:
- **Single seed**: circle/square of v at center. Size and amplitude matter.
- **Multiple seeds**: several spots at different locations → pattern competition.
- **Random noise**: u=1 + noise, v=noise → simultaneous nucleation everywhere.
- **Stripe seed**: initial stripe of v → biases system toward stripe/labyrinth patterns.
- **Gradient seed**: spatially varying perturbation → pattern wavelength selection.

## Your workflow

### Setup (once per session)
1. Read this file and `prepare.py`
2. Read `simulate.py` for current configuration
3. Check `best.json` and `leaderboard.json`

### Experiment loop
1. **Analyze**: what scored highest? Which (F,k) region? Which IC type?
2. **Hypothesize**: form a specific hypothesis about (F,k) or IC structure.
3. **Implement**: new IC function + parameters in `simulate.py`, set `CURRENT_CONFIG`.
4. **Run**: `python simulate.py`
5. **Evaluate**: check score breakdown (entropy vs contrast vs bonus).
6. **Repeat**.

## Interpreting results
- **Score < 1**: no patterns formed (wrong parameters or IC too weak)
- **Score 1-5**: simple patterns (single mode, low entropy)
- **Score 5-15**: moderate complexity (spots or stripes)
- **Score > 15**: high complexity (mixed patterns or spatiotemporal chaos)
- **Score > 30**: exceptional — near the complexity frontier for this grid/budget

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
