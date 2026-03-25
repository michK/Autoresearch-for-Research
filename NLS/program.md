# NLS Rogue Wave Hunter — Agent Program

## Objective
Maximize score = `peak_amplification × (1 + growth_rate) × bonus`
where `peak_amplification = max(|ψ(x,t)|) / a` over the full simulation.

## Physics Reference

**Equation:** i∂ₜψ + ∂ₓₓψ + |ψ|²ψ = 0 (focusing NLS)
**Background:** ψ₀ = a·exp(ia²t),  |ψ₀| = a = 1

### Modulational Instability (MI)
- Unstable wavenumbers: |k| < a√2 ≈ 1.414
- Growth rate: σ(k) = k·√(2a² − k²)
- Most unstable: k* = a = 1.0  (mode index n=10 for our domain L=20π)
- σ_max = a² = 1.0  (e-folding time ~1)
- MI band modes in our domain (Δk = 0.1): n = 1..14

### Known Exact Solutions (amplification targets)
| Solution | Peak amplification | Notes |
|----------|--------------------|-------|
| Unperturbed plane wave | 1.0 | baseline |
| Akhmediev breather | < 3.0 | periodic in x |
| Peregrine soliton | **3.0** | rational, maximum for order-1 |
| Order-2 rogue wave | **5.0** | requires fine-tuned IC |
| Order-3 rogue wave | **7.0** | very sensitive IC |

### Scoring bonus
- 1.5× if `final_amplification > 1.5` (still elevated at t=50)
- 1.0× otherwise
- Bonus likely for solutions that are periodic (Akhmediev) or still evolving

## Search Strategy

### Round 1 (baselines — already coded in simulate.py)
- `plane_wave_baseline`: sanity check, should give ~1.0
- `single_mode_optimal`: MI at k*=1, expect ~2.5–3.0
- `peregrine_ic`: seeded Peregrine, expect ~3.0
- `akhmediev_ic`: seeded Akhmediev breather, may earn bonus (periodic)
- `multi_mode_in_band`: all unstable modes, expect chaotic ~2–3

### Hypotheses to explore
1. **Phase optimization**: cos vs sin perturbation, phase sweep at k*
2. **Amplitude sweep**: eps=0.001, 0.01, 0.1 — how does eps affect timing/peak?
3. **Mode superposition**: seed k* + 2k* constructively — can beat Peregrine?
4. **Higher-order Peregrine IC**: exact order-2 solution seeded at t=-10
5. **Multiple Peregrine seeds**: localized at different x positions — superposition
6. **Background amplitude**: a=2 → MI grows faster, but score normalizes by a
7. **Narrow-band vs broadband**: focused perturbation in k-space vs spread

### Key physical insights
- Peregrine peak = 3a is **NOT** the absolute maximum — higher-order rational
  solutions (order n) give (2n+1)a. These require extremely precise ICs.
- Akhmediev breathers are periodic and may earn the 1.5× bonus (still oscillating)
- Multiple breathers at different k can interfere constructively
- The "noise background" (multi_mode) may give very high but brief spikes

## Implementation Notes
- IC must return complex128 array of shape (N=1024,)
- Background amplitude is a=1; never return all-zeros
- `psi = a * (1 + eps * perturbation)` is the standard form
- For seeded exact solutions, use the analytical formula with t<0 start
- The solver conserves: ∫|ψ|² dx (mass/power), ∫(|∂ₓψ|² - |ψ|⁴/2) dx (Hamiltonian)
