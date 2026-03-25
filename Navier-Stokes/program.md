# Navier-Stokes Blowup Hunter: Agent Instructions

You are an autonomous research agent hunting for initial conditions that produce extreme vorticity growth in 3D incompressible Navier-Stokes equations on a periodic domain (T^3). This is related to the Clay Millennium Prize problem on N-S existence and smoothness.

## Background

The Beale-Kato-Majda theorem says: a smooth solution of 3D Navier-Stokes blows up at time T if and only if the time integral of ||omega||_inf (max vorticity) diverges as t -> T. So our proxy for "getting close to blowup" is finding initial conditions that produce the fastest, most sustained growth of max vorticity within a fixed compute budget.

We work on a 64^3 periodic domain with a pseudospectral solver. Each experiment runs for 3 minutes wall clock. The score combines vorticity amplification ratio with enstrophy growth rate.

## Your workflow

### Setup (once per session)

1. Read this file (`program.md`)
2. Read `prepare.py` to understand the solver, metrics, and scoring
3. Read `simulate.py` to understand the current experiment configuration
4. Check `best.json` and `leaderboard.json` for current best results
5. Look at recent experiment logs in `experiments/` to understand what's been tried

### Experiment loop (repeat)

1. **Analyze**: Look at the leaderboard and recent results. What initial conditions scored highest? What patterns do you see? Are there unexplored regions of the design space?

2. **Hypothesize**: Form a specific hypothesis about what might produce more extreme vorticity growth. Write it down as a comment in `simulate.py`. Examples:
   - "Anti-parallel tubes with tighter separation might increase reconnection rate"
   - "Superimposing high-wavenumber perturbations on ABC flow might trigger faster cascade"
   - "Combining two known good configurations might produce constructive interference of vortex stretching"

3. **Modify `simulate.py`**: Edit the file to implement your hypothesis. You can:
   - Create new initial condition functions
   - Modify parameters of existing ones (tube separation, perturbation strength, etc.)
   - Adjust viscosity (lower nu = higher Re = more interesting but harder to resolve)
   - Combine multiple vortex structures
   - Try completely novel constructions

4. **Run**: Execute `python simulate.py` and wait for it to complete (~3 min)

5. **Evaluate**: Check the output. Did the score improve? Look at the metrics:
   - Was max vorticity still growing at the end? (Good sign)
   - Did enstrophy grow exponentially? (Good sign)
   - Did the simulation hit numerical blowup? (Interesting but might be numerical artifact at 64^3)
   - Did energy conservation hold? (Sanity check)

6. **Record**: The system automatically logs results. If the experiment improved the score, great. If not, analyze why and adjust.

7. **Repeat** from step 1.

## Key physics intuition for the agent

### What drives vorticity growth
- **Vortex stretching**: The omega . grad(u) term is the ONLY mechanism for vorticity amplification in 3D. (This is why 2D N-S doesn't blow up: no vortex stretching.)
- **Vortex reconnection**: When anti-parallel vortex tubes approach each other, they reconnect, creating intense localized vorticity.
- **Cascade to small scales**: Energy transfers from large to small scales, concentrating gradients.

### What suppresses blowup
- **Viscosity (nu)**: Diffuses vorticity. Lower nu lets vorticity grow more, but also makes the simulation harder to resolve numerically.
- **Depletion of nonlinearity**: In many known flows, geometric constraints prevent the vortex stretching term from maintaining its maximum growth rate. This is the "supercriticality barrier" identified by Tao (2016).
- **Numerical resolution**: At 64^3, we can't distinguish true blowup from under-resolution artifacts once gradients get very sharp. This is a fundamental limitation.

### What to try
- **Anti-parallel vortex tubes** are the gold standard. Vary: tube radius (sigma), separation distance, perturbation wavelength and amplitude, circulation strength.
- **Trefoil/linked vortex configurations**: Topological linking forces reconnection events.
- **Multi-scale initial conditions**: Seed energy at multiple wavenumbers simultaneously.
- **Symmetry exploitation**: High-symmetry configurations (like Kida-Pelz) can focus vortex interactions.
- **Low viscosity with carefully chosen scales**: Push nu down to 0.001 or lower, but ensure the initial condition has most energy at large scales so the simulation stays resolved longer.
- **Combining configurations**: Superimpose two known vortex structures that interact.
- **Exotic constructions**: Vortex sheets, concentrated vortex rings colliding, helical vortex filaments.

### What NOT to do
- Don't modify `prepare.py`. It's the fixed evaluation harness.
- Don't set nu = 0 (Euler equations). The solver assumes nonzero viscosity for stability.
- Don't make the initial condition so complex that most energy is at high wavenumbers. It'll just get dealiased away.
- Don't chase numerical artifacts. If a "blowup" happens in the first few timesteps, it's probably just a resolution issue, not real physics.

## Interpreting results

- **Score > 10**: Decent vorticity amplification.
- **Score > 50**: Significant growth, worth investigating further.
- **Score > 100**: Excellent. The initial condition is producing sustained, accelerating vorticity growth.
- **Score > 500**: Exceptional. Potentially publishable finding at this resolution.
- **Numerical blowup with smooth early growth**: Very interesting. Try re-running at the same parameters to confirm, and note that this might be a resolution artifact or might be genuine.

## File structure

```
prepare.py       -- fixed solver, metrics, logging (DO NOT MODIFY)
simulate.py      -- your experiment file (MODIFY THIS)
program.md       -- these instructions (read-only during experiments)
best.json        -- current best score
leaderboard.json -- top 20 experiments
experiments/     -- all experiment logs (JSON)
```

## Starting a session

Say something like: "I've read program.md. Let me check the current state and run a new experiment." Then follow the workflow above.

Good hunting.

