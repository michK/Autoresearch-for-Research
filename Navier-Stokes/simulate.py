"""
Navier-Stokes Autoresearch: Simulation Configuration
=====================================================
THIS IS THE FILE THE AGENT MODIFIES.

The agent's job is to design initial conditions that produce the most
extreme vorticity growth in 3D incompressible Navier-Stokes on T^3.

The key creative levers are:
  1. Initial velocity field construction
  2. Viscosity (nu) - lower = higher Reynolds number = more turbulent
  3. Combinations of known vortex structures
  4. Novel constructions the agent invents

The score metric rewards:
  - High vorticity amplification (peak / initial)
  - Sustained enstrophy growth rate
  - Vorticity still growing at simulation end (bonus)

Each experiment runs for a fixed 3-minute wall clock budget.
"""

import numpy as np
from prepare import run_simulation, N, L


# ===========================================================================
# Initial condition library
# ===========================================================================

def taylor_green(x, y, z, kx, ky, kz):
    """
    Taylor-Green vortex: classic benchmark.
    Produces vortex stretching and a turbulent cascade.
    Known to show significant enstrophy growth before viscous decay.
    """
    A = 1.0
    ux = A * np.sin(x) * np.cos(y) * np.cos(z)
    uy = -A * np.cos(x) * np.sin(y) * np.cos(z)
    uz = np.zeros_like(x)
    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def anti_parallel_vortex_tubes(x, y, z, kx, ky, kz):
    """
    Anti-parallel vortex tubes approaching each other.
    This configuration (studied by Kerr 1993, Hou & Li 2006) is believed
    to produce the most extreme vorticity growth in 3D Euler/N-S.
    The tubes undergo reconnection, which concentrates vorticity.
    """
    # Two vortex tubes along x-axis, offset in y, with opposite circulation
    sigma = 0.3  # tube radius
    separation = 1.5  # y-separation between tubes
    strength = 3.0

    y1 = np.pi - separation / 2
    y2 = np.pi + separation / 2
    z0 = np.pi

    # Gaussian vortex tubes with vorticity along x
    r1_sq = (y - y1)**2 + (z - z0)**2
    r2_sq = (y - y2)**2 + (z - z0)**2

    omega_x1 = strength * np.exp(-r1_sq / (2 * sigma**2))
    omega_x2 = -strength * np.exp(-r2_sq / (2 * sigma**2))

    # Add perturbation along the tubes to break translational symmetry
    # This triggers the instability that leads to reconnection
    perturbation = 0.3 * np.sin(2 * x)
    omega_x = (omega_x1 + omega_x2) * (1 + perturbation)

    # Reconstruct velocity from vorticity via Biot-Savart in spectral space
    # omega = (omega_x, 0, 0) -> solve curl(u) = omega
    omega_x_hat = np.fft.rfftn(omega_x)

    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1.0

    # From curl(u) = omega with div(u) = 0:
    # u_hat = i * (k x omega_hat) / k^2
    # With omega = (omega_x, 0, 0):
    #   uy_hat = i * kz * omega_x_hat / k^2
    #   uz_hat = -i * ky * omega_x_hat / k^2
    #   ux_hat = 0 (from this component alone)
    ux_hat = np.zeros_like(omega_x_hat)
    uy_hat = 1j * kz * omega_x_hat / k_sq
    uz_hat = -1j * ky * omega_x_hat / k_sq

    return ux_hat, uy_hat, uz_hat


def kida_vortex(x, y, z, kx, ky, kz):
    """
    Kida-Pelz high-symmetry initial condition.
    Exploits octahedral symmetry to concentrate vortex interactions.
    """
    A = 2.0
    ux = A * (np.sin(x) * np.cos(y) * np.cos(z)
              + 0.5 * np.sin(2*x) * np.cos(2*z))
    uy = A * (-np.cos(x) * np.sin(y) * np.cos(z)
              + 0.5 * np.cos(2*y) * np.sin(2*z))
    uz = A * (np.cos(x) * np.cos(y) * np.sin(z)
              - 0.5 * np.sin(2*y) * np.cos(2*x))

    # The above is not exactly divergence-free; the Leray projection
    # in prepare.py will fix this.
    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def perturbed_abc_flow(x, y, z, kx, ky, kz):
    """
    Arnold-Beltrami-Childress flow with perturbation.
    ABC flows are exact steady solutions of Euler equations (eigenfunction
    of the curl operator). Adding perturbation can trigger instabilities.
    """
    A, B, C = 1.0, 1.0, 1.0
    eps = 0.2  # perturbation strength

    ux = A * np.sin(z) + C * np.cos(y) + eps * np.sin(2*x) * np.cos(3*z)
    uy = B * np.sin(x) + A * np.cos(z) + eps * np.cos(2*y) * np.sin(3*x)
    uz = C * np.sin(y) + B * np.cos(x) + eps * np.sin(2*z) * np.cos(3*y)

    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def taylor_green_low_nu(x, y, z, kx, ky, kz):
    """
    Hypothesis: Taylor-Green at high amplitude + low viscosity produces
    sustained enstrophy growth via turbulent cascade.
    TG is known to produce enstrophy amplification before viscous decay;
    at lower nu the peak is higher and delayed, maximizing our score window.
    Energy concentrated at k=1 (large scale) ensures good resolution.
    """
    A = 2.0  # higher amplitude → faster cascade
    ux = A * np.sin(x) * np.cos(y) * np.cos(z)
    uy = -A * np.cos(x) * np.sin(y) * np.cos(z)
    uz = np.zeros_like(x)
    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def tg_multiscale(x, y, z, kx, ky, kz):
    """
    Taylor-Green at k=1 (A=2) with k=2 seed (eps).
    Optimal eps=0.27 at nu=0.0001 gives score=565.
    """
    A = 2.0
    eps = 0.27  # optimal value found by experiment
    ux = A * np.sin(x)*np.cos(y)*np.cos(z) + eps * np.sin(2*x)*np.cos(2*y)*np.cos(2*z)
    uy = -A * np.cos(x)*np.sin(y)*np.cos(z) - eps * np.cos(2*x)*np.sin(2*y)*np.cos(2*z)
    uz = np.zeros_like(x)
    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def tg_multiscale_3d(x, y, z, kx, ky, kz):
    """
    Hypothesis: adding uz at k=2 in the TG-permuted orientation seeds the
    z-direction cascade simultaneously. Current tg_multiscale has only xy k=2
    seeding. By adding the (y,z,x) permutation TG at k=2 with uz≠0, the cascade
    becomes more isotropic in 3D. Optimal eps values tuned for current best.

    The uz = eps_z * sin(2z)*cos(2x)*cos(2y) adds the TG structure in the
    zx-plane. Leray projection ensures divergence-free.
    """
    A = 2.0
    eps_xy = 0.27  # k=2 xy-component (from tg_multiscale optimal)
    eps_z = 0.15   # k=2 z-component: small enough to not inflate initial vort much

    # k=1 TG main component
    ux = A * np.sin(x)*np.cos(y)*np.cos(z)
    uy = -A * np.cos(x)*np.sin(y)*np.cos(z)
    uz = np.zeros_like(x)

    # k=2 TG in xy-plane (same as tg_multiscale)
    ux += eps_xy * np.sin(2*x)*np.cos(2*y)*np.cos(2*z)
    uy -= eps_xy * np.cos(2*x)*np.sin(2*y)*np.cos(2*z)

    # k=2 TG in z-direction (add uz to seed z-cascade)
    uz += eps_z * np.sin(2*z)*np.cos(2*x)*np.cos(2*y)

    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def tg_antiphase_k2_low_nu(x, y, z, kx, ky, kz):
    """
    Hypothesis: At nu=0.00005, cascade peaks BEFORE wall clock (no bonus, peak=464).
    A NEGATIVE eps (anti-phase k=2 seed) destructively interferes with the natural
    k=2 cascade daughters, DELAYING the cascade peak to after wall clock → bonus.
    At Re=50000 (vs 20000 at nu=0.0001), if the cascade can be timed to peak
    just after wall clock, the higher Re could produce peak vort > 515 with bonus.

    eps=-0.10: moderate anti-phase interference to delay cascade.
    If peak ~520 with bonus: score = (520/4)*2.93*1.5 = 573 > 565.

    Note: initial max vort = 4.0 (anti-phase seed doesn't affect k=1 maximum).
    """
    A = 2.0
    eps = -0.10   # anti-phase: destructively interferes with natural k=2 daughters

    ux = A * np.sin(x)*np.cos(y)*np.cos(z) + eps * np.sin(2*x)*np.cos(2*y)*np.cos(2*z)
    uy = -A * np.cos(x)*np.sin(y)*np.cos(z) - eps * np.cos(2*x)*np.sin(2*y)*np.cos(2*z)
    uz = np.zeros_like(x)
    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def tg_asymmetric_k2(x, y, z, kx, ky, kz):
    """
    Hypothesis: Break TG symmetry (ux_amp = uy_amp = 2) by adding 0.2 × TG_yzx.
    TG_xyz: ux=2*sin(x)cos(y)cos(z), uy=-2*cos(x)sin(y)cos(z), uz=0
    TG_yzx: ux=0, uy=0.2*sin(y)cos(z)cos(x), uz=-0.2*cos(y)sin(z)cos(x)

    Combined (naturally div-free):
      ux = 2*sin(x)cos(y)cos(z)
      uy = -1.8*cos(x)sin(y)cos(z)    [reduced from 2]
      uz = -0.2*cos(x)cos(y)sin(z)     [nonzero from t=0]

    Key: ω_z = uy_x + ux_y = (1.8+2)*sin(x)sin(y)cos(z) = 3.8*... → initial max vort = 3.8!
    Lower than pure TG (4.0). If peak is similar (~500), vort_amp = 131 vs 128 → score ~577.
    Also: uz≠0 provides z-direction vortex stretching from t=0.
    Plus k=2 TG seed (eps=0.27) for cascade acceleration.
    """
    A = 2.0
    A_prime = 0.2   # yzx TG component (subtracted from uy, adds uz)
    eps = 0.27

    # TG_xyz + TG_yzx: naturally div-free combination
    ux = A * np.sin(x)*np.cos(y)*np.cos(z)
    uy = -(A - A_prime) * np.cos(x)*np.sin(y)*np.cos(z)
    uz = -A_prime * np.cos(x)*np.cos(y)*np.sin(z)

    # k=2 TG seed
    ux += eps * np.sin(2*x)*np.cos(2*y)*np.cos(2*z)
    uy -= eps * np.cos(2*x)*np.sin(2*y)*np.cos(2*z)

    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def tg_direct_stretch(x, y, z, kx, ky, kz):
    """
    Hypothesis: TG's dominant vorticity ω_z=2A*sin(x)*sin(y)*cos(z) has ZERO
    initial vortex stretching because uz=0 → ∂uz/∂z=0. The cascade starts only
    after uz develops nonlinearly (~0.5 time units).

    Adding uz = gamma*sin(x)*sin(y)*sin(z) creates ∂uz/∂z = gamma*sin(x)*sin(y)*cos(z),
    which stretches ω_z at its maximum (x=π/2, y=π/2, z=0) from t=0.
    After Leray projection: 2/3 of uz survives (∂uz_proj/∂z = 2*gamma/3*sin(x)*sin(y)*cos(z)).
    This starts the vortex stretching cascade earlier → higher peak vort.

    gamma=0.1 is small enough to barely change initial max vort (~4.003),
    but provides IMMEDIATE stretching of ω_z. Combined with eps=0.27 k=2 seed.
    """
    A = 2.0
    eps = 0.27
    gamma = 0.1   # uz at k=(1,1,1) sin-sin-sin, targets ω_z stretching

    ux = A * np.sin(x)*np.cos(y)*np.cos(z)
    uy = -A * np.cos(x)*np.sin(y)*np.cos(z)
    uz = gamma * np.sin(x)*np.sin(y)*np.sin(z)   # after Leray: 2/3 survives, stretches ω_z

    # k=2 TG seed
    ux += eps * np.sin(2*x)*np.cos(2*y)*np.cos(2*z)
    uy -= eps * np.cos(2*x)*np.sin(2*y)*np.cos(2*z)

    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def tg_A19_eps030(x, y, z, kx, ky, kz):
    """
    Hypothesis: At A=2.0, eps=0.30 gives peak=533 but NO bonus (cascade too fast).
    At A=2.0, eps=0.27 gives peak=515 WITH bonus (score=565).
    With bonus, eps=0.30 would score: (533/4)*2.94*1.5 = 588.

    Strategy: Lower A=1.9 slows the cascade (lower Re, slower turnover).
    This might restore the bonus at eps=0.30 while keeping high peak vort.
    Lower initial vort (3.8 vs 4.0) also improves vort_amp ratio.
    Expected score if bonus: (519/3.8) * 2.94 * 1.5 ≈ 603.
    """
    A = 1.9
    eps = 0.30

    ux = A * np.sin(x)*np.cos(y)*np.cos(z) + eps * np.sin(2*x)*np.cos(2*y)*np.cos(2*z)
    uy = -A * np.cos(x)*np.sin(y)*np.cos(z) - eps * np.cos(2*x)*np.sin(2*y)*np.cos(2*z)
    uz = np.zeros_like(x)
    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def tg_natural_daughters(x, y, z, kx, ky, kz):
    """
    Hypothesis: The NATURAL k=2 nonlinear daughters of TG k=1 evolution are at
    modes (2,0,2) and (0,2,2), NOT at TG k=2 (2,2,2) which we've been using.
    Seeding the resonant daughter modes directly should trigger a more efficient
    cascade, potentially raising peak vorticity above the eps=0.27 baseline (565).

    From (u.∇u) for TG k=1:
      (u.∇u)_x = (A²/4)*sin(2x)*cos(2z) → solenoidal: ux~sin(2x)cos(2z), uz~-cos(2x)sin(2z)
      (u.∇u)_y = (A²/4)*sin(2y)*cos(2z) → solenoidal: uy~sin(2y)cos(2z), uz~-cos(2y)sin(2z)

    These modes are at |k|=sqrt(8)≈2.83, SMALLER than TG k=2's |k|=sqrt(12)≈3.46.
    Less viscous damping → more effective per unit seeded energy.
    Testing eps=0.27 to compare directly with best TG k=2 result.
    """
    A = 2.0
    eps = 0.27

    # k=1 TG base
    ux = A * np.sin(x)*np.cos(y)*np.cos(z)
    uy = -A * np.cos(x)*np.sin(y)*np.cos(z)
    uz = np.zeros_like(x)

    # Natural k=2 daughter from (2,0,2) mode [from (u.∇u)_x solenoidal projection]
    ux += eps * np.sin(2*x)*np.cos(2*z)
    uz += -eps * np.cos(2*x)*np.sin(2*z)

    # Natural k=2 daughter from (0,2,2) mode [from (u.∇u)_y solenoidal projection]
    uy += eps * np.sin(2*y)*np.cos(2*z)
    uz += -eps * np.cos(2*y)*np.sin(2*z)

    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def shear_k2(x, y, z, kx, ky, kz):
    """
    Hypothesis: Kolmogorov shear uy=A*sin(x) has initial max vorticity = A = 2.0,
    HALF of TG's 4.0 (for same A). If peak vorticity is similar (~400-500),
    vort_amp = peak/initial doubles from ~128 to ~200, potentially doubling score.

    The TG k=2 seed (eps=0.27) serves dual purpose:
    1. Provides 3D perturbation to trigger the KH instability in the shear layer
    2. Seeds the nonlinear cascade (same as in tg_multiscale)

    Key: shear+k2 initial max vort ≈ 2.49 (vs TG's 4.0) because k=2 vorticity maxima
    occur at different spatial locations than the shear vorticity maximum.
    At equal peak vort (~400): vort_amp ≈ 400/2.49 ≈ 160 vs 128 → score ~703 vs 565.
    """
    A = 2.0
    eps = 0.27   # same optimal k=2 seed as best tg_multiscale

    # Kolmogorov shear: only y-velocity, varying in x. Div-free trivially.
    # Max vorticity = omega_z = A*cos(x), max = A = 2.0
    ux = np.zeros_like(x)
    uy = A * np.sin(x)
    uz = np.zeros_like(x)

    # k=2 TG seed: 3D perturbation + cascade seed
    ux += eps * np.sin(2*x)*np.cos(2*y)*np.cos(2*z)
    uy -= eps * np.cos(2*x)*np.sin(2*y)*np.cos(2*z)

    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def tg_A22(x, y, z, kx, ky, kz):
    """
    Taylor-Green at A=2.2 (vs baseline A=2.0) with optimal k=2 seed eps=0.27.
    Higher amplitude → Re increases by 10% → peak vort should increase.
    Initial vort scales as A (~4.4 vs 4.0), peak vort scales faster with Re.
    If vort_amp = peak/initial increases even slightly, score beats 565.
    Risk: cascade peaks earlier (simulation time), potentially before wall clock limit → no bonus.
    """
    A = 2.2
    eps = 0.27   # optimal k=2 seed from tg_multiscale

    ux = A * np.sin(x)*np.cos(y)*np.cos(z)
    uy = -A * np.cos(x)*np.sin(y)*np.cos(z)
    uz = np.zeros_like(x)

    ux += eps * np.sin(2*x)*np.cos(2*y)*np.cos(2*z)
    uy -= eps * np.cos(2*x)*np.sin(2*y)*np.cos(2*z)

    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def tg_k1_3d_k2(x, y, z, kx, ky, kz):
    """
    Hypothesis: Standard TG has uz=0 which limits z-direction vortex stretching.
    Adding a small uz = gamma*cos(x)*cos(y)*sin(z) at k=1 (the Kida-Pelz z-mode)
    enables stretching in all 3 directions from the START, not just after cascade
    populates uz. This should enhance the overall vortex stretching rate.
    Leray projection will keep it divergence-free. Combined with eps=0.27 k=2 seed
    for cascade acceleration.
    gamma=0.2 is small enough to not significantly raise initial vorticity.
    """
    A = 2.0
    eps = 0.27    # optimal k=2 seed
    gamma = 0.2   # small uz at k=1 (Kida-Pelz z-mode)

    ux = A * np.sin(x)*np.cos(y)*np.cos(z)
    uy = -A * np.cos(x)*np.sin(y)*np.cos(z)
    uz = gamma * np.cos(x)*np.cos(y)*np.sin(z)  # Kida-like z-component at k=1

    # k=2 seed
    ux += eps * np.sin(2*x)*np.cos(2*y)*np.cos(2*z)
    uy -= eps * np.cos(2*x)*np.sin(2*y)*np.cos(2*z)

    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def tg_multiscale_k23(x, y, z, kx, ky, kz):
    """
    Taylor-Green at k=1 (A=2) + k=2 seed (eps=0.27, optimal) + k=3 seed (eps_k3).
    Hypothesis: k=3 seeding cascades energy to even smaller scales faster, potentially
    raising peak vorticity without disrupting the k=2-driven timing. k=3 modes
    carry less energy (eps_k3 << eps_k2) so they shouldn't shift the bonus cliff.
    The k=3 modes hit viscous cutoff faster, so they should add a brief burst
    of vorticity amplification on top of the k=2-driven cascade.
    """
    A = 2.0
    eps_k2 = 0.27   # optimal from tg_multiscale (score=565)
    eps_k3 = 0.05   # small k=3 perturbation

    ux = A * np.sin(x)*np.cos(y)*np.cos(z)
    uy = -A * np.cos(x)*np.sin(y)*np.cos(z)
    uz = np.zeros_like(x)

    # k=2 seed (TG structure at wavenumber 2)
    ux += eps_k2 * np.sin(2*x)*np.cos(2*y)*np.cos(2*z)
    uy -= eps_k2 * np.cos(2*x)*np.sin(2*y)*np.cos(2*z)

    # k=3 seed (TG structure at wavenumber 3)
    ux += eps_k3 * np.sin(3*x)*np.cos(3*y)*np.cos(3*z)
    uy -= eps_k3 * np.cos(3*x)*np.sin(3*y)*np.cos(3*z)

    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def tg_A22_eps0245(x, y, z, kx, ky, kz):
    """
    A=2.2, eps=0.245, nu=0.0001. TESTED: score=303 (no bonus). Worse.
    """
    A = 2.2
    eps = 0.245
    ux = A * np.sin(x)*np.cos(y)*np.cos(z)
    uy = -A * np.cos(x)*np.sin(y)*np.cos(z)
    uz = np.zeros_like(x)
    ux += eps * np.sin(2*x)*np.cos(2*y)*np.cos(2*z)
    uy -= eps * np.cos(2*x)*np.sin(2*y)*np.cos(2*z)
    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def tg_tiny_eps(x, y, z, kx, ky, kz):
    """
    TG k=1 (A=2) + tiny k=2 seed (eps=0.05) at nu=0.0001.
    Pure TG (eps=0): egr~3.5, no bonus, score=494. vort_amp=111, (1+egr)=4.5.
    TG+k2 (eps=0.27): egr~2.0, bonus, score=565. vort_amp=127, (1+egr)=3.0.

    HYPOTHESIS: eps=0.05 barely seeds k=2, so:
    - The cascade is only slightly redirected → egr stays near 3.0 (vs 2.0)
    - The tiny seed delays the cascade JUST enough to get the bonus
    - Result: vort_amp=115?, (1+egr)=3.5?, bonus: 115*3.5*1.5 = 604 > 565

    If pure TG peaks at 180s with no bonus: eps=0.05 cascade peaks slightly later → bonus!
    But: pure TG (eps=0) peaks BEFORE 180s (no bonus). Adding any eps>0 accelerates
    the cascade → also peaks before 180s → still no bonus?

    Possible resolution: the k=2 seed actually DELAYS the cascade by channeling energy
    into a different mode structure, slowing the k=1 energy cascade.
    """
    A = 2.0
    eps = 0.05   # tiny seed: minimal disruption to TG cascade dynamics

    ux = A * np.sin(x)*np.cos(y)*np.cos(z)
    uy = -A * np.cos(x)*np.sin(y)*np.cos(z)
    uz = np.zeros_like(x)

    ux += eps * np.sin(2*x)*np.cos(2*y)*np.cos(2*z)
    uy -= eps * np.cos(2*x)*np.sin(2*y)*np.cos(2*z)

    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def tg_eps_cliff_search(x, y, z, kx, ky, kz):
    """
    Hypothesis: The bonus cliff is between eps=0.27 (bonus, score=565) and eps=0.28
    (no bonus, score=305). The score is monotone increasing in eps (within bonus region).
    Therefore the optimal eps is the highest value that still gets the bonus condition.
    Testing eps=0.275 — halfway between 0.27 and 0.28. If it gets the bonus, score > 565.
    If not, we'll try eps=0.272 or 0.271.

    Bonus condition: vorts[-1] > vorts[-3], i.e., vorticity still rising at wall clock end.
    Cascade peaks LATER with LOWER eps (fewer k=2 modes → slower transfer).
    eps=0.275 should still peak after 180s if the cliff is sharp near 0.28.
    """
    A = 2.0
    eps = 0.275   # between known good (0.27) and bad (0.28)

    ux = A * np.sin(x)*np.cos(y)*np.cos(z)
    uy = -A * np.cos(x)*np.sin(y)*np.cos(z)
    uz = np.zeros_like(x)

    ux += eps * np.sin(2*x)*np.cos(2*y)*np.cos(2*z)
    uy -= eps * np.cos(2*x)*np.sin(2*y)*np.cos(2*z)

    return np.fft.rfftn(ux), np.fft.rfftn(uy), np.fft.rfftn(uz)


def colliding_vortex_rings(x, y, z, kx, ky, kz):
    """
    Two vortex rings colliding head-on along the z-axis.
    Ring 1: centered at (π,π,π-d), positive toroidal vorticity → self-induced +z velocity.
    Ring 2: centered at (π,π,π+d), negative toroidal vorticity → self-induced -z velocity.
    They approach and collide at z=π, triggering extreme vortex reconnection.

    This is qualitatively different from TG: instead of distributed vortex stretching,
    we get concentrated azimuthal-to-axial vorticity conversion during reconnection.
    Experimentally (Kida & Takaoka 1994) vortex ring collisions produce the fastest
    known vorticity growth rates in 3D incompressible flow.

    Strategy: strong rings (strength=4) with moderate radius (R=1.2) and small core (σ=0.25),
    separated by d=1.2 so they interact early. Low nu=0.0001 for high Re.
    Adding TG k=2 seed (eps=0.15) to accelerate small-scale cascade after reconnection.
    """
    # Ring parameters
    strength = 4.0
    R = 1.2        # ring radius
    sigma = 0.25   # core size (well-resolved at 64^3)
    d = 1.2        # half-separation: rings at z=π±d

    r = np.sqrt((x - np.pi)**2 + (y - np.pi)**2)
    r = np.where(r < 1e-10, 1e-10, r)  # avoid division by zero

    # Toroidal vorticity magnitude at each point (Gaussian torus)
    z1 = np.pi - d
    z2 = np.pi + d

    omega_phi1 = strength * np.exp(-((r - R)**2 + (z - z1)**2) / (2 * sigma**2))
    omega_phi2 = -strength * np.exp(-((r - R)**2 + (z - z2)**2) / (2 * sigma**2))

    # Convert φ-direction vorticity to (x,y,z) components:
    # ω_x = -sin(φ) × ω_phi = -(y-π)/r × ω_phi
    # ω_y = cos(φ) × ω_phi = (x-π)/r × ω_phi
    # ω_z = 0 (toroidal rings have no axial vorticity initially)
    omega_x1 = -(y - np.pi) / r * omega_phi1
    omega_y1 = (x - np.pi) / r * omega_phi1
    omega_x2 = -(y - np.pi) / r * omega_phi2
    omega_y2 = (x - np.pi) / r * omega_phi2

    omega_x = omega_x1 + omega_x2
    omega_y = omega_y1 + omega_y2
    omega_z = np.zeros_like(x)

    # Convert vorticity → velocity via Biot-Savart: u_hat = i(k × ω_hat) / k²
    ox_hat = np.fft.rfftn(omega_x)
    oy_hat = np.fft.rfftn(omega_y)
    oz_hat = np.fft.rfftn(omega_z)

    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1.0

    # u_hat = i * (k × omega_hat) / k²
    # k × omega = (ky*oz - kz*oy, kz*ox - kx*oz, kx*oy - ky*ox)
    ux_hat = 1j * (ky * oz_hat - kz * oy_hat) / k_sq
    uy_hat = 1j * (kz * ox_hat - kx * oz_hat) / k_sq
    uz_hat = 1j * (kx * oy_hat - ky * ox_hat) / k_sq

    # Zero mean flow
    ux_hat[0, 0, 0] = 0.0
    uy_hat[0, 0, 0] = 0.0
    uz_hat[0, 0, 0] = 0.0

    # Add TG k=2 seed to accelerate small-scale cascade after reconnection
    eps = 0.15
    ux_seed = eps * np.sin(2*x)*np.cos(2*y)*np.cos(2*z)
    uy_seed = -eps * np.cos(2*x)*np.sin(2*y)*np.cos(2*z)
    ux_hat += np.fft.rfftn(ux_seed)
    uy_hat += np.fft.rfftn(uy_seed)

    return ux_hat, uy_hat, uz_hat


def reconnecting_tubes(x, y, z, kx, ky, kz):
    """
    Hypothesis: Anti-parallel tubes with DISPLACEMENT perturbation (not
    strength modulation) will trigger faster Crow instability and reconnection.
    At x=pi/2 the tubes are displaced toward each other by 2*perturb_amp,
    reaching minimum separation = separation - 2*perturb_amp, driving strong
    localized vortex stretching during reconnection.
    """
    sigma = 0.3          # tube core radius (well resolved at 64^3)
    separation = 1.2     # initial center-to-center distance
    strength = 5.0       # vortex strength
    perturb_amp = 0.35   # displacement amplitude (min sep = 1.2 - 0.7 = 0.5)

    y1 = np.pi - separation / 2
    y2 = np.pi + separation / 2
    z0 = np.pi

    # Sinusoidal core displacement toward each other (Crow instability trigger)
    dy = perturb_amp * np.sin(2 * x)

    r1_sq = (y - (y1 + dy))**2 + (z - z0)**2
    r2_sq = (y - (y2 - dy))**2 + (z - z0)**2

    omega_x1 = strength * np.exp(-r1_sq / (2 * sigma**2))
    omega_x2 = -strength * np.exp(-r2_sq / (2 * sigma**2))
    omega_x = omega_x1 + omega_x2

    omega_x_hat = np.fft.rfftn(omega_x)
    k_sq = kx**2 + ky**2 + kz**2
    k_sq[0, 0, 0] = 1.0

    ux_hat = np.zeros_like(omega_x_hat)
    uy_hat = 1j * kz * omega_x_hat / k_sq
    uz_hat = -1j * ky * omega_x_hat / k_sq

    return ux_hat, uy_hat, uz_hat


# ===========================================================================
# Current experiment configuration
# ===========================================================================

# The agent should modify this section to try new things.
# Change the initial condition function, viscosity, or create entirely
# new initial condition constructors above.

CURRENT_CONFIG = {
    "name": "tg_multiscale_eps027_nu0001",
    "description": "BEST CONFIG: TG k=1 (A=2) + TG k=2 seed (eps=0.27) + nu=0.0001. Score=565.15. eps=0.27 is the highest eps that still gets the 1.5x bonus (cascade peaks exactly at 180s). Explored: A=2.2, eps range 0.05-0.275, vortex rings, reconnecting tubes - all worse.",
    "initial_condition": "tg_multiscale",
    "nu": 0.0001,
}


def get_initial_condition():
    """Return the current experiment's initial condition function."""
    ic_map = {
        "taylor_green": taylor_green,
        "anti_parallel_vortex_tubes": anti_parallel_vortex_tubes,
        "kida_vortex": kida_vortex,
        "perturbed_abc_flow": perturbed_abc_flow,
        "taylor_green_low_nu": taylor_green_low_nu,
        "tg_multiscale": tg_multiscale,
        "tg_multiscale_3d": tg_multiscale_3d,
        "tg_A19_eps030": tg_A19_eps030,
        "tg_natural_daughters": tg_natural_daughters,
        "shear_k2": shear_k2,
        "tg_antiphase_k2_low_nu": tg_antiphase_k2_low_nu,
        "tg_A22_eps0245": tg_A22_eps0245,
        "tg_eps_cliff_search": tg_eps_cliff_search,
        "colliding_vortex_rings": colliding_vortex_rings,
        "tg_tiny_eps": tg_tiny_eps,
        "tg_asymmetric_k2": tg_asymmetric_k2,
        "tg_direct_stretch": tg_direct_stretch,
        "tg_A22": tg_A22,
        "tg_k1_3d_k2": tg_k1_3d_k2,
        "tg_multiscale_k23": tg_multiscale_k23,
        "reconnecting_tubes": reconnecting_tubes,
    }
    return ic_map[CURRENT_CONFIG["initial_condition"]]


# ===========================================================================
# Main entry point
# ===========================================================================

if __name__ == "__main__":
    ic_fn = get_initial_condition()
    result = run_simulation(
        initial_condition_fn=ic_fn,
        nu=CURRENT_CONFIG["nu"],
        config=CURRENT_CONFIG,
    )
    print(f"\nExperiment result: score={result['score']:.4f}")

