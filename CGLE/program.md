# CGLE Spiral Hunter — Agent Program

## Objective
Maximize score = `peak_defect_density × (1 + growth_rate) × bonus`
where `peak_defect_density = max(n_defects(t)) / N` over the full simulation.

## Physics Reference

**Equation:** ∂ₜA = A + (1+ic₁)∇²A - (1+ic₂)|A|²A  (2D CGLE)
**Domain:** N=128, L=64, periodic BC
**Plane wave solution:** A₀ = exp(iωt),  |A₀| = 1,  ω = −c₂

### Parameter regimes
| Region | c₁ | c₂ | Dynamics |
|--------|-----|-----|----------|
| Benjamin-Feir stable | c₁c₂ < 1 | — | Plane wave survives |
| Benjamin-Feir unstable | c₁c₂ > 1 | — | Phase turbulence |
| Defect (amplitude) chaos | c₂ ≲ −1 | — | Many vortex-antivortex pairs |
| Spiral turbulence | c₁≈1.5, c₂≈−2 | — | High defect count |
| Frozen spirals | c₁≈2, c₂≈−0.5 | — | Low egr, no bonus |

### Benjamin-Feir instability
- Onset: c₁c₂ > 1
- Growth rate of mode k: σ(k) = −k²[(1+c₁²)k² − 2(1+c₁c₂)]/(2a²)
  (where a=1 is plane wave amplitude)
- Most unstable k: k* = √[(1+c₁c₂)/(1+c₁²)]

### Topological defects (vortices)
- A defect is a point where A=0 (phase undefined)
- Winding number ±1 around each defect
- Created/annihilated in pairs (opposite winding)
- High defect count ↔ spatiotemporal chaos

### Scoring bonus
- 1.5× if final_defect_density > 0.1 × peak (still active at t=300)
- 1.0× if defects have all annihilated (frozen spirals)

## Search Strategy

### Round 1 (baselines)
- `uniform_plane_wave`: A=1, should give 0 defects (BF stable) or slow growth
- `noisy_plane_wave`: A=1 + small noise, tests BF instability for given (c₁,c₂)
- `random_phase`: random amplitude/phase, tests which parameters support chaos
- `spiral_seed`: manually seeded spiral(s), tests whether spirals self-sustain

### Hypotheses to explore
1. **Parameter sweep**: grid search over (c₁,c₂) with noise IC to find chaos region
2. **Multi-spiral seed**: seed many spirals → fast transient to turbulence
3. **Sparse vs dense seed**: few large spirals vs many small seeds
4. **Amplitude modulation IC**: non-uniform amplitude to trigger defect nucleation
5. **Near BF boundary**: c₁c₂ just above 1 → phase turbulence without defects?
6. **Deep chaos**: large |c₂| → maximum defect density?

### Key physical insights
- Defect chaos requires sufficient |c₂| (nonlinear frequency shift)
- c₁ > 0 needed for supercritical bifurcation (subcritical c₁<0 may collapse)
- Defects form spontaneously in BF-unstable region from any small perturbation
- Higher defect density = higher score; they should persist for bonus
- The growth_rate bonus rewards systems still nucleating defects at late time

## Implementation Notes
- IC must return complex128 array of shape (N, N) = (128, 128)
- Standard IC: `A = np.ones((N,N), dtype=complex) * np.exp(1j * phi_noise)`
- For spiral seed: analytic vortex profile `A = r/(r+1) * exp(iθ)` (approximate)
- The solver uses Strang splitting; DT=0.1, MAX_SIM_TIME=300
- Wall budget 120s → ~600 time units possible but limited to 300
