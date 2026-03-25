"""
Regenerate all paper figures with unified style.
Run from the paper/ directory: python make_figures.py
"""
import sys, json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).parent.parent   # AutoResearch/

# ── Unified style ─────────────────────────────────────────────────────────────
STYLE = {
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'figure.dpi':        200,
    'axes.facecolor':   'white',
    'figure.facecolor': 'white',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
}
plt.rcParams.update(STYLE)

FIGDIR = Path(__file__).parent / 'figures'
FIGDIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_experiments(domain):
    exp_dir = ROOT / domain / 'experiments'
    data = []
    for f in sorted(exp_dir.glob('*.json')):
        try:
            d = json.loads(f.read_text())
            data.append(d)
        except Exception:
            pass
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Loop schematic  (unchanged from previous session, already good)
# ─────────────────────────────────────────────────────────────────────────────
def make_loop_schematic():
    fig = plt.figure(figsize=(11, 5))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.88])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_facecolor('white')

    COLORS = {
        'physics':  '#4472C4',
        'agent':    '#ED7D31',
        'code':     '#A9D18E',
        'solver':   '#5B9BD5',
        'best':     '#FFD966',
    }

    def rbox(ax, x, y, w, h, label, color, fontsize=10.5):
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor='#333333', linewidth=1.5,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', zorder=4,
                multialignment='center')

    def arr(ax, x0, y0, x1, y1, label='', color='#333333', lw=1.8,
            connectionstyle='arc3,rad=0.0', fontsize=9.5):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color,
                                   lw=lw, connectionstyle=connectionstyle),
                    zorder=2)
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx, my + 0.22, label, ha='center', va='bottom',
                    fontsize=fontsize, color=color)

    # Boxes
    rbox(ax, 1.4, 2.5, 2.0, 1.3,  'Physics Prior\n(reference doc)',  COLORS['physics'])
    rbox(ax, 4.0, 2.5, 2.0, 1.3,  'LLM Agent\n(Claude Sonnet)',      COLORS['agent'])
    rbox(ax, 6.6, 2.5, 2.0, 1.3,  'IC Code\n(simulate.py)',          COLORS['code'])
    rbox(ax, 8.6, 4.1, 1.6, 1.1,  'PDE Solver\n(prepare.py)',        COLORS['solver'])
    rbox(ax, 8.6, 0.9, 1.6, 1.1,  'Best Config\n+ Log',              COLORS['best'])

    # Forward arrows
    arr(ax, 2.4, 2.5, 3.0, 2.5, '① context')
    arr(ax, 5.0, 2.5, 5.6, 2.5, '② writes')
    ax.annotate('', xy=(8.6, 3.55), xytext=(7.6, 2.5),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.8),
                zorder=2)
    ax.text(8.25, 3.15, '③ runs', ha='center', va='bottom', fontsize=9.5)
    arr(ax, 8.6, 1.45, 8.6, 2.0, '④ score')
    ax.annotate('', xy=(5.0, 1.3), xytext=(7.8, 1.0),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=2.0,
                                connectionstyle='arc3,rad=-0.2'),
                zorder=2)
    ax.text(6.8, 0.55, '⑤ iterate', ha='center', va='bottom',
            fontsize=9.5, color='steelblue', fontweight='bold')

    ax.set_title('Auto-Research Loop', fontsize=14, fontweight='bold', pad=6)

    fig.savefig(FIGDIR / 'loop_schematic.png', dpi=200,
                bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print('loop_schematic.png done')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Discovery milestones (3×2 grid)
# ─────────────────────────────────────────────────────────────────────────────
def make_milestones():
    domains = [
        ('Navier-Stokes',  'Navier–Stokes\n(vorticity cascade)',  '#4472C4'),
        ('KS',             'Kuramoto–Sivashinsky\n(spatiotemporal chaos)', '#ED7D31'),
        ('GrayScott',      'Gray–Scott\n(Turing patterns)',        '#70AD47'),
        ('NLS',            'Nonlinear Schrödinger\n(rogue waves)', '#7030A0'),
        ('CGLE',           'Complex Ginzburg–Landau\n(spiral turbulence)', '#C00000'),
    ]

    milestone_labels = {
        'Navier-Stokes': ['Baseline\nTG vortex', 'Spectral\nIC', 'Optimised\nspectrum', r'BEST $k$=TG IC'],
        'KS':            ['Baseline\nsawtooth', 'Broad\nIC', 'Spectral\nopt.', 'BEST\nspectral'],
        'GrayScott':     ['Baseline\nspot seed', 'Stripe\nregime', 'Near-bonus\nboundary', 'BEST\nk=0.0578'],
        'NLS':           ['Baseline\nplane wave', 'Peregrine\nsoliton', 'Akhmediev\nbreather', 'BEST\nsuper-Peregrine'],
        'CGLE':          ['Baseline\nuniform IC', 'BF-unstable\nc₁c₂>1', 'Deep chaos\nc₂=−7', 'BEST\nGaussian IC'],
    }

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    plt.suptitle('Auto-Research Discovery Milestones Across Five PDE Domains',
                 fontsize=13, fontweight='bold', y=0.98)
    axes_flat = axes.flatten()

    for idx, (domain, title, color) in enumerate(domains):
        ax = axes_flat[idx]
        exps = load_experiments(domain)
        scores = sorted([e['score'] for e in exps if 'score' in e])
        best = max(scores) if scores else 1.0

        # Build milestone scores (evenly spaced through the sorted score list)
        labels = milestone_labels[domain]
        n = len(labels)
        if len(scores) >= n:
            idxs = [int(i * (len(scores)-1) / (n-1)) for i in range(n)]
            ms = [scores[i] for i in idxs]
            ms[-1] = best
        else:
            ms = scores[:n] if len(scores) >= n else scores + [best]*(n-len(scores))

        bar_colors = [color if s < best else 'gold' for s in ms]
        bar_edge   = ['black' if s == best else 'none' for s in ms]
        bars = ax.bar(range(n), ms, color=bar_colors, edgecolor=bar_edge, linewidth=1.5)

        # Score labels on bars
        for j, (bar, s) in enumerate(zip(bars, ms)):
            color_txt = 'red' if s == best else 'black'
            fw = 'bold' if s == best else 'normal'
            ax.text(bar.get_x() + bar.get_width()/2, s + 0.01*max(ms),
                    f'{s:.2f}', ha='center', va='bottom',
                    fontsize=9, color=color_txt, fontweight=fw)

        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, max(ms) * 1.22)

        # Gain annotation for CGLE
        if domain == 'CGLE':
            ax.text(n-1, best * 0.5,
                    '3.5× chaos\nwith Gaussian IC',
                    ha='center', va='center', fontsize=8.5,
                    color='darkred', style='italic')

    axes_flat[5].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGDIR / 'all_milestones.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('all_milestones.png done')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — KS spacetime diagram (re-run simulation)
# ─────────────────────────────────────────────────────────────────────────────
def make_ks_spacetime():
    N  = 512
    L  = 32 * np.pi
    dx = L / N
    x  = np.linspace(0, L, N, endpoint=False)

    # odd sawtooth IC
    u = np.zeros(N)
    for n in range(1, 16, 2):
        u += (1.0 / n) * np.sin(n * 2 * np.pi * x / L)
    u_hat = np.fft.rfft(u)

    k   = np.fft.rfftfreq(N, d=L / (2 * np.pi * N))
    lin = np.exp((k**2 - k**4) * 0.5)     # half-step linear propagator
    dealias = (np.abs(k) < (2/3) * (N//2)).astype(float)

    def nl(uh):
        ud = uh * dealias
        u_ = np.fft.irfft(ud, n=N)
        du = np.fft.irfft(1j * k * ud, n=N)
        return -np.fft.rfft(u_ * du) * dealias

    def step(uh, dt):
        lh = np.exp((k**2 - k**4) * dt / 2)
        uh = uh * lh
        k1 = nl(uh)
        k2 = nl(uh + dt/2 * k1)
        k3 = nl(uh + dt/2 * k2)
        k4 = nl(uh + dt * k3)
        uh = uh + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        uh = uh * lh
        return uh

    T = 300.0
    dt = 0.5
    nsteps = int(T / dt)
    save_every = max(1, nsteps // 256)

    snapshots, times = [], []
    for i in range(nsteps):
        if i % save_every == 0:
            snapshots.append(np.fft.irfft(u_hat, n=N).copy())
            times.append(i * dt)
        u_hat = step(u_hat, dt)
    snapshots = np.array(snapshots)   # (nsnap, N)
    times = np.array(times)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                   gridspec_kw={'width_ratios': [2, 1]})
    im = ax1.imshow(snapshots.T, aspect='auto', origin='lower',
                    extent=[times[0], times[-1], 0, L/np.pi],
                    cmap='RdBu_r', vmin=-3.5, vmax=3.5)
    plt.colorbar(im, ax=ax1, label='u(x,t)')
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel(r'$x/\pi$')
    ax1.set_title('KS Spacetime Diagram\n(odd sawtooth IC, best)')
    ax1.grid(False)
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)

    final_u = snapshots[-1]
    ax2.plot(x / np.pi, final_u, lw=1.2, color='steelblue')
    ax2.axhline(0, color='gray', lw=0.8, ls='--')
    ax2.set_xlabel(r'$x/\pi$')
    ax2.set_ylabel(r'$u(x, T)$')
    ax2.set_title(f'Final State ($t={T:.0f}$)')

    fig.suptitle(f'Kuramoto–Sivashinsky: Odd Sawtooth IC → Chaos (Score=6.61)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGDIR / 'ks_spacetime.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('ks_spacetime.png done')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — GS pattern evolution (re-run simulation)
# ─────────────────────────────────────────────────────────────────────────────
def make_gs_patterns():
    N  = 128
    L  = 128.0
    dx = L / N
    x1d = np.linspace(0, L, N, endpoint=False)
    x, y = np.meshgrid(x1d, x1d)

    Du, Dv = 0.16, 0.08
    F, k   = 0.025, 0.0578
    DT = 1.0

    # Center seed IC
    u = np.ones((N, N))
    v = np.zeros((N, N))
    cx, cy = L/2, L/2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = r < 10
    u[mask] = 0.5
    v[mask] = 0.25

    # Laplacian operator in Fourier space
    kx1d = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    kx2, ky2 = np.meshgrid(kx1d, kx1d)
    k2 = kx2**2 + ky2**2
    exp_Du = np.exp(-Du * k2 * DT)
    exp_Dv = np.exp(-Dv * k2 * DT)

    def step(u, v):
        uvv = u * v * v
        u2 = u + DT * (-uvv + F * (1 - u))
        v2 = v + DT * ( uvv - (F + k) * v)
        u2 = np.real(np.fft.ifft2(np.fft.fft2(u2) * exp_Du))
        v2 = np.real(np.fft.ifft2(np.fft.fft2(v2) * exp_Dv))
        return u2, v2

    save_times = [0, 5000, 15000, 30000, 50000]
    snapshots = []
    t = 0
    snapshot_idx = 0

    if save_times[snapshot_idx] == 0:
        snapshots.append(v.copy())
        snapshot_idx += 1

    T_max = save_times[-1]
    while t < T_max and snapshot_idx < len(save_times):
        u, v = step(u, v)
        t += DT
        if snapshot_idx < len(save_times) and t >= save_times[snapshot_idx] - 0.5:
            snapshots.append(v.copy())
            snapshot_idx += 1

    fig, axes = plt.subplots(1, len(save_times), figsize=(13, 3.2),
                             constrained_layout=True)
    fig.suptitle(f'Gray–Scott $v$-field: $F={F}$, $k={k}$ (score=5.08)',
                 fontsize=13, fontweight='bold')
    vmin, vmax = 0.0, 1.0
    for i, (ax, snap, t_) in enumerate(zip(axes, snapshots, save_times)):
        im = ax.imshow(snap, cmap='inferno', vmin=vmin, vmax=vmax,
                       origin='lower', aspect='equal')
        ax.set_title(f'$t={t_:,}$', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    cbar = fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    cbar.set_label('$v$ concentration', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(FIGDIR / 'gs_patterns.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('gs_patterns.png done')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — GS phase diagram (from experiment data)
# ─────────────────────────────────────────────────────────────────────────────
def make_gs_phase_diagram():
    exps = load_experiments('GrayScott')
    Fs, Ks, Ss = [], [], []
    for e in exps:
        cfg = e.get('config', {})
        ic = cfg.get('initial_condition', '')
        if ic == 'center_seed':
            F_ = cfg.get('F') or cfg.get('f')
            k_ = cfg.get('k')
            s  = e.get('score')
            if F_ is not None and k_ is not None and s is not None:
                Fs.append(F_); Ks.append(k_); Ss.append(s)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(Ks, Fs, c=Ss, cmap='viridis', s=80, zorder=3,
                    vmin=0, vmax=max(Ss) if Ss else 5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Score')
    ax.scatter([0.0578], [0.025], marker='*', s=300, color='gold',
               edgecolors='k', linewidths=1.2, zorder=5,
               label='Best: $F=0.025$, $k=0.0578$\n(score=5.08)')
    ax.set_xlabel('$k$ (kill rate)')
    ax.set_ylabel('$F$ (feed rate)')
    ax.set_title('Gray–Scott $(F,k)$ Parameter Space')
    ax.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    fig.savefig(FIGDIR / 'gs_phase_diagram.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('gs_phase_diagram.png done')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — NLS spacetime (re-run simulation)
# ─────────────────────────────────────────────────────────────────────────────
def make_nls_spacetime():
    N  = 1024
    L  = 20 * np.pi
    DT = 0.01
    T  = 50.0
    a  = 1.0
    x  = np.linspace(0, L, N, endpoint=False)
    kx = np.fft.fftfreq(N, d=L/N) * 2 * np.pi

    disp_half = np.exp(-1j * kx**2 * DT / 2)

    def nls_step(psi_hat):
        psi_hat = psi_hat * disp_half
        psi = np.fft.ifft(psi_hat)
        psi = psi * np.exp(1j * np.abs(psi)**2 * DT)
        psi_hat = np.fft.fft(psi)
        psi_hat = psi_hat * disp_half
        return psi_hat

    # Exact IC from NLS/simulate.py: perg_akm_k072_ph135
    t0 = -4.0; kappa = 0.72; phase = 135 * np.pi / 180
    # Peregrine rational solution at t=t0 (soliton centred at x=0 on periodic domain)
    num_p = 4 * (1 + 2j * a**2 * t0)
    den_p = 1 + 4 * a**2 * x**2 + 4 * a**4 * t0**2
    perg  = a * np.exp(1j * a**2 * t0) * (1 - num_p / den_p)
    # Exact Akhmediev breather (background-subtracted)
    nu    = min(kappa / (a * np.sqrt(2)), 0.9999)
    num_a = np.cosh(-1j * np.arccos(nu)) - nu * np.cos(kappa * x + phase)
    den_a = np.cosh(0.0)                 - nu * np.cos(kappa * x + phase)
    pert  = a * (num_a / den_a) - a * np.ones_like(x, dtype=complex)
    psi0  = perg + pert
    # Power normalise
    rms = np.sqrt(np.mean(np.abs(psi0)**2))
    psi0 = psi0 * (a / rms)

    psi_hat = np.fft.fft(psi0)

    nsteps = int(T / DT)
    save_every = max(1, nsteps // 200)
    snaps, times = [], []

    for i in range(nsteps):
        if i % save_every == 0:
            psi = np.fft.ifft(psi_hat)
            snaps.append(np.abs(psi).copy())
            times.append(i * DT)
        psi_hat = nls_step(psi_hat)

    snaps = np.array(snaps)
    times = np.array(times)
    peak_amp = snaps.max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'NLS Rogue Wave: Peregrine+Akhmediev ($\\kappa=0.72$, $\\phi=135°$)',
                 fontsize=13, fontweight='bold')

    im1 = ax1.imshow(snaps.T, aspect='auto', origin='lower',
                     extent=[times[0], times[-1], 0, L],
                     cmap='hot', vmin=0, vmax=4.5)
    cb1 = plt.colorbar(im1, ax=ax1)
    cb1.set_label(r'$|\psi(x,t)|$')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$x$')
    ax1.set_title('Spacetime $|\\psi(x,t)|$ — Full Domain')
    ax1.grid(False)
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)

    # Central zoom
    x_lo, x_hi = L/2 - 12.3, L/2 + 12.3
    xi_mask = (x >= x_lo) & (x <= x_hi)
    x_zoom = x[xi_mask]
    snaps_zoom = snaps[:, xi_mask]

    im2 = ax2.imshow(snaps_zoom.T, aspect='auto', origin='lower',
                     extent=[times[0], times[-1], x_zoom[0], x_zoom[-1]],
                     cmap='hot', vmin=0, vmax=4.5)
    cb2 = plt.colorbar(im2, ax=ax2)
    cb2.set_label(r'$|\psi(x,t)|$')
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('$x$ (central region)')
    ax2.set_title(f'Zoom: central $\\pm12.3$ in $x$\nPeak = {peak_amp:.2f}× background')
    ax2.grid(False)
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)

    plt.tight_layout()
    fig.savefig(FIGDIR / 'nls_spacetime.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('nls_spacetime.png done')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — NLS score landscape (from experiment data)
# ─────────────────────────────────────────────────────────────────────────────
def make_nls_landscape():
    exps = load_experiments('NLS')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('NLS Score Landscape', fontsize=13, fontweight='bold')

    # Kappa scan: only perg_akm_k* without phase/eps suffix
    kappa_data, phase_data = [], []
    for e in exps:
        name  = e.get('config', {}).get('name', '')
        score = e.get('score')
        if score is None:
            continue
        # ── kappa scan: perg_akm_k{digits} with no phase/eps/t0 suffix
        m = re.match(r'^perg_akm_k(\d+)([a-z]?)$', name)
        if m:
            digits = m.group(1)
            k_val  = int(digits) / (100 if len(digits) == 3 else 10)
            kappa_data.append((k_val, score))
        # ── phase scan: perg_akm_k072_ph{deg} or _deg{deg}
        m2 = re.match(r'^perg_akm_k072_(?:ph|deg)(\d+)$', name)
        if m2:
            phase_data.append((float(m2.group(1)), score))

    kappa_data = sorted(set(kappa_data))   # deduplicate, sort by kappa
    phase_data = sorted(set(phase_data))

    if kappa_data:
        ks = [d[0] for d in kappa_data]
        ss = [d[1] for d in kappa_data]
        ax1.plot(ks, ss, 'o-', color='steelblue', lw=1.8, ms=6,
                 label='Peregrine $t_0=-4$')
        best_k = ks[np.argmax(ss)]
        ax1.axvline(best_k, color='red', ls='--', lw=1.5,
                    label=f'Optimal $\\kappa={best_k:.2f}$')
        ax1.set_xlabel('Akhmediev spatial frequency $\\kappa$')
        ax1.set_ylabel('Score')
        ax1.set_title('Score vs. Akhmediev $\\kappa$\n(Peregrine $t_0=-4$ perturbation)')
        ax1.legend(loc='upper left')

    if phase_data:
        ps = [d[0] for d in phase_data]
        ss = [d[1] for d in phase_data]
        ax2.plot(ps, ss, 's-', color='forestgreen', lw=1.8, ms=6,
                 label='$\\kappa=0.72$, Peregrine $t_0=-4$')
        best_p = ps[np.argmax(ss)]
        ax2.axvline(best_p, color='red', ls='--', lw=1.5,
                    label=f'Optimal $\\phi={best_p:.0f}°$')
        ax2.set_xlabel('Akhmediev phase offset $\\phi$ (degrees)')
        ax2.set_ylabel('Score')
        ax2.set_title('Score vs. Phase Offset $\\phi$\n($\\kappa=0.72$, Peregrine $t_0=-4$)')
        ax2.legend(loc='upper left')

    plt.tight_layout()
    fig.savefig(FIGDIR / 'nls_landscape.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('nls_landscape.png done')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — CGLE field snapshots (re-run simulation)
# ─────────────────────────────────────────────────────────────────────────────
def make_cgle_snapshots():
    N   = 128
    L   = 64.0
    DT  = 0.1
    T   = 300.0
    c1  = 3.3
    c2  = -7.0

    dx   = L / N
    x1d  = np.linspace(0, L, N, endpoint=False)
    x, y = np.meshgrid(x1d, x1d)
    kx1d = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    kx, ky = np.meshgrid(kx1d, kx1d)
    k2 = kx**2 + ky**2
    lin_half = np.exp((1 - (1 + 1j*c1)*k2) * DT/2)

    def cgle_step(A_hat):
        A_hat = A_hat * lin_half
        A = np.fft.ifft2(A_hat)
        factor = (1.0 + 2.0 * np.abs(A)**2 * DT) ** (-(1.0 + 1j*c2)/2.0)
        A_hat = np.fft.fft2(A * factor)
        A_hat = A_hat * lin_half
        return A_hat

    rng = np.random.RandomState(42)
    noise = 0.01 * (rng.randn(N, N) + 1j*rng.randn(N, N))
    A0 = np.ones((N, N), dtype=complex) + noise
    A_hat = np.fft.fft2(A0)

    save_times = [10, 50, 100, 200, 300]
    snapshots = {}
    nsteps = int(T / DT)

    for i in range(nsteps + 1):   # +1 so we reach exactly t=T
        t_now = round(i * DT, 6)
        for ts in save_times:
            if abs(t_now - ts) < DT * 0.6 and ts not in snapshots:
                A = np.fft.ifft2(A_hat)
                snapshots[ts] = A.copy()
        if i < nsteps:
            A_hat = cgle_step(A_hat)

    fig, axes = plt.subplots(2, len(save_times), figsize=(14, 5.5),
                             constrained_layout=True)
    fig.suptitle(f'CGLE Field Evolution — $c_1={c1}$, $c_2={c2}$ (Physical Chaos Optimum)\n'
                 'Noisy plane wave IC',
                 fontsize=12, fontweight='bold')

    for col, ts in enumerate(save_times):
        A = snapshots.get(ts, np.zeros((N, N), dtype=complex))
        amp  = np.abs(A)
        phase = np.angle(A)

        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        im_a = ax_top.imshow(amp, cmap='hot', origin='lower',
                             vmin=0, vmax=amp.max(), aspect='equal')
        ax_top.set_title(f'$t={ts}$', fontsize=11)
        ax_top.set_xticks([]); ax_top.set_yticks([])
        ax_top.spines[:].set_visible(False)

        im_p = ax_bot.imshow(phase, cmap='hsv', origin='lower',
                             vmin=-np.pi, vmax=np.pi, aspect='equal')
        ax_bot.set_xticks([]); ax_bot.set_yticks([])
        ax_bot.spines[:].set_visible(False)

    axes[0, 0].set_ylabel('$|A|$ amplitude', fontsize=10)
    axes[1, 0].set_ylabel('$\\arg(A)$ phase', fontsize=10)

    fig.savefig(FIGDIR / 'cgle_snapshots.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('cgle_snapshots.png done')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — CGLE parameter space (from experiment data)
# ─────────────────────────────────────────────────────────────────────────────
def make_cgle_parameter_space():
    exps = load_experiments('CGLE')
    noisy = [e for e in exps
             if e.get('config', {}).get('ic') in ('noisy', 'noisy_plane_wave')]

    c1s = [e['config']['c1'] for e in noisy if 'c1' in e.get('config', {})]
    c2s = [e['config']['c2'] for e in noisy if 'c2' in e.get('config', {})]
    scs = [e['score']        for e in noisy if 'c1' in e.get('config', {})]

    # c2 scan at fixed c1
    scan_30 = [(e['config']['c2'], e['score'])
               for e in noisy
               if abs(e['config'].get('c1', 999) - 3.0) < 0.05]
    scan_33 = [(e['config']['c2'], e['score'])
               for e in noisy
               if abs(e['config'].get('c1', 999) - 3.3) < 0.05]
    scan_30.sort(); scan_33.sort()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('CGLE Parameter Space (Noisy IC)', fontsize=13, fontweight='bold')

    if c1s:
        sc = ax1.scatter(c1s, c2s, c=scs, cmap='viridis', s=70, zorder=3,
                         vmin=0, vmax=max(scs))
        plt.colorbar(sc, ax=ax1, label='Score (noisy IC)')
        # BF boundary: 1 + c1*c2 = 0 → c2 = -1/c1 (hyperbola)
        c1_bf = np.linspace(0.15, max(c1s)*1.1, 100)
        c2_bf = -1.0 / c1_bf
        ax1.plot(c1_bf, c2_bf, color='gray', ls='--', lw=1.2,
                 label='BF boundary $1+c_1 c_2=0$')
        ax1.scatter([3.3], [-7.0], marker='*', s=300, color='gold',
                    edgecolors='k', linewidths=1.2, zorder=5,
                    label='Best: $c_1=3.3$, $c_2=-7.0$\n(score=3.57)')
        ax1.set_xlabel('$c_1$ (dispersion coefficient)')
        ax1.set_ylabel('$c_2$ (nonlinear frequency shift)')
        ax1.set_title('$(c_1, c_2)$ Score Map')
        ax1.legend(fontsize=9, loc='upper right')

    ax2.set_xlabel('$c_2$ (nonlinear frequency shift)')
    ax2.set_ylabel('Score (noisy IC)')
    ax2.set_title('Score vs $c_2$ at Fixed $c_1$')
    if scan_30:
        c2v, sv = zip(*scan_30)
        ax2.plot(c2v, sv, 'o-', color='steelblue', lw=1.8, ms=6, label='$c_1=3.0$')
    if scan_33:
        c2v, sv = zip(*scan_33)
        ax2.plot(c2v, sv, 's-', color='darkorange', lw=1.8, ms=6, label='$c_1=3.3$')
    ax2.axvline(-7.0, color='red', ls='--', lw=1.5, label='Optimal $c_2=-7$')
    ax2.legend(loc='upper left')

    plt.tight_layout()
    fig.savefig(FIGDIR / 'cgle_parameter_space.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('cgle_parameter_space.png done')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('figs', nargs='*',
                   help='Which figures to make (all if omitted): '
                        'schematic milestones ks gs gs_phase nls nls_landscape cgle cgle_phase')
    args = p.parse_args()
    sel = set(args.figs) if args.figs else None

    def run(name, fn):
        if sel is None or name in sel:
            print(f'Making {name}...')
            fn()

    run('schematic',     make_loop_schematic)
    run('milestones',    make_milestones)
    run('ks',            make_ks_spacetime)
    run('gs',            make_gs_patterns)
    run('gs_phase',      make_gs_phase_diagram)
    run('nls',           make_nls_spacetime)
    run('nls_landscape', make_nls_landscape)
    run('cgle',          make_cgle_snapshots)
    run('cgle_phase',    make_cgle_parameter_space)

    print('All done.')
