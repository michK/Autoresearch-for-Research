"""
Plotting script for NS auto-research phase 1 results.
Generates several figures for the methodology paper.
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ─── Load all experiments ────────────────────────────────────────────────────
exp_files = sorted(glob.glob("experiments/*.json"))
experiments = []
for f in exp_files:
    with open(f) as fh:
        experiments.append(json.load(fh))

with open("leaderboard.json") as fh:
    leaderboard = json.load(fh)

# Sort experiments by timestamp
experiments.sort(key=lambda e: e["timestamp"])
print(f"Loaded {len(experiments)} experiments")

# ─── Helpers ────────────────────────────────────────────────────────────────
def get_ts(exp, key):
    return [m[key] for m in exp["metrics_history"]]

def get_times(exp):
    return [m["time"] for m in exp["metrics_history"]]

def get_wall(exp):
    return [m["wall_elapsed"] for m in exp["metrics_history"]]

# ─── Figure 1: Search trajectory ─────────────────────────────────────────────
# Shows score vs experiment number, with running best highlighted.
fig, ax = plt.subplots(figsize=(9, 4))

scores = [e["score"] for e in experiments]
names = [e["config"]["name"] for e in experiments]
xs = list(range(1, len(scores) + 1))

running_best = []
best_so_far = 0
for s in scores:
    best_so_far = max(best_so_far, s)
    running_best.append(best_so_far)

# Color by whether bonus was achieved — extracted from vorticity time series
# Bonus = vorts[-1] > vorts[-3]
def earned_bonus(exp):
    vorts = [m["max_vorticity"] for m in exp["metrics_history"]]
    if len(vorts) < 3:
        return False
    return vorts[-1] > vorts[-3]

FAILED_APPROACHES = {"colliding", "reconnect"}

colors = []
for e in experiments:
    name = e["config"]["name"]
    if any(k in name for k in FAILED_APPROACHES):
        colors.append("#e74c3c")   # red = failed approach (low KE)
    elif earned_bonus(e):
        colors.append("#2ecc71")   # green = bonus earned
    else:
        colors.append("#e67e22")   # orange = no bonus

ax.bar(xs, scores, color=colors, alpha=0.75, width=0.7, zorder=2)
ax.plot(xs, running_best, 'k--', linewidth=1.5, label='Running best', zorder=3)
ax.axhline(565.15, color='red', linewidth=1, linestyle=':', alpha=0.7, label='Final best = 565.15')

# Legend patches
green_p = mpatches.Patch(color='#2ecc71', alpha=0.75, label='Bonus earned')
orange_p = mpatches.Patch(color='#e67e22', alpha=0.75, label='No bonus / exploration')
red_p = mpatches.Patch(color='#e74c3c', alpha=0.75, label='Failed approach (low KE)')
ax.legend(handles=[green_p, orange_p, red_p], loc='lower right', framealpha=0.9)

ax.set_xlabel("Experiment number (chronological)")
ax.set_ylabel("Score")
ax.set_title("Figure 1: Search trajectory — score vs experiment number")
ax.set_xlim(0.3, len(xs) + 0.7)
ax.grid(axis='y', alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_search_trajectory.png")
plt.close(fig)
print("Saved fig1_search_trajectory.png")

# ─── Figure 2: Vorticity time series for key configs ─────────────────────────
KEY_CONFIGS = {
    "tg_multiscale_eps027_nu0001": ("Best: TG + k=2 seed (ε=0.27, bonus)", "#2ecc71", 2.5),
    "tg_multiscale_eps025_nu0001": ("TG + k=2 seed (ε=0.25, bonus)", "#27ae60", 1.5),
    "taylor_green_A2_nu0001":       ("Pure TG, no k=2 seed (bonus)", "#3498db", 1.5),
    "tg_eps0275_nu0001":            ("TG + k=2 seed (ε=0.275, no bonus)", "#e74c3c", 1.8),
    "tg_multiscale_eps03_nu0001":   ("TG + k=2 seed (ε=0.30, no bonus)", "#e67e22", 1.5),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plotted = set()
all_series = {}
for e in experiments:
    name = e["config"]["name"]
    cfg = KEY_CONFIGS.get(name)
    if cfg and name not in plotted:
        label, color, lw = cfg
        ts = get_ts(e, "max_vorticity")
        t = get_times(e)
        all_series[name] = (t, ts, label, color, lw)
        plotted.add(name)

for name, (t, ts, label, color, lw) in all_series.items():
    axes[0].plot(t, ts, color=color, linewidth=lw, label=label)
    axes[1].plot(t, ts, color=color, linewidth=lw, label=label)

axes[0].set_xlabel("Simulation time")
axes[0].set_ylabel("Max vorticity ‖ω‖∞")
axes[0].set_title("Full time range")
axes[0].legend(loc='upper left', framealpha=0.9, fontsize=8)
axes[0].grid(alpha=0.3)

# Zoom into the final 20% where bonus cliff matters
if all_series:
    max_t = max(t[-1] for t, _, _, _, _ in all_series.values())
    axes[1].set_xlim(max_t * 0.75, max_t)
    axes[1].set_xlabel("Simulation time")
    axes[1].set_ylabel("Max vorticity ‖ω‖∞")
    axes[1].set_title("Zoom: final 25% — bonus cliff visible")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc='upper left', framealpha=0.9, fontsize=8)

fig.suptitle("Figure 2: Vorticity growth — ε=0.27 is highest that still earns the ×1.5 bonus", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_vorticity_timeseries.png")
plt.close(fig)
print("Saved fig2_vorticity_timeseries.png")

# ─── Figure 3: The eps cliff ─────────────────────────────────────────────────
# Score vs eps for TG multiscale at nu=0.0001
EPS_MAP = {
    "tg_tiny_eps_nu0001": 0.05,
    "tg_multiscale_nu0001": 0.15,        # eps=0.15
    "tg_multiscale_eps020_nu0001": 0.20,
    "tg_multiscale_eps025_nu0001": 0.25,
    "tg_multiscale_eps027_nu0001": 0.27,
    "tg_eps0275_nu0001": 0.275,
    "tg_multiscale_eps03_nu0001": 0.30,
}

eps_scores = {}
for e in experiments:
    name = e["config"]["name"]
    if name in EPS_MAP:
        eps_scores[EPS_MAP[name]] = e["score"]

# Sort
eps_vals = sorted(eps_scores.keys())
score_vals = [eps_scores[e] for e in eps_vals]

fig, ax = plt.subplots(figsize=(7, 4))
bonus_eps = [e for e in eps_vals if eps_scores[e] > 450]
no_bonus_eps = [e for e in eps_vals if eps_scores[e] <= 450]
ax.scatter([e for e in eps_vals if e <= 0.27], [eps_scores[e] for e in eps_vals if e <= 0.27],
           color='#2ecc71', s=80, zorder=5, label='Bonus earned (×1.5)')
ax.scatter([e for e in eps_vals if e > 0.27], [eps_scores[e] for e in eps_vals if e > 0.27],
           color='#e74c3c', s=80, zorder=5, label='No bonus')
ax.plot(eps_vals, score_vals, 'k-', alpha=0.4, linewidth=1)

# Annotate the cliff
ax.axvline(0.27, color='gray', linestyle='--', alpha=0.6)
ax.annotate("Bonus cliff\nε = 0.27", xy=(0.27, 480), xytext=(0.235, 430),
            arrowprops=dict(arrowstyle='->', color='gray'), color='gray', fontsize=9)

ax.set_xlabel("k=2 seed amplitude ε")
ax.set_ylabel("Score")
ax.set_title("Figure 3: Score vs ε — sharp bonus cliff at ε = 0.27\n(TG k=1, A=2, ν=0.0001)")
ax.legend(framealpha=0.9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_eps_cliff.png")
plt.close(fig)
print("Saved fig3_eps_cliff.png")

# ─── Figure 4: Enstrophy and energy for the best config ──────────────────────
best_exp = next(e for e in experiments if e["config"]["name"] == "tg_multiscale_eps027_nu0001")

t = get_times(best_exp)
vort = get_ts(best_exp, "max_vorticity")
enst = get_ts(best_exp, "enstrophy")
ener = get_ts(best_exp, "energy")

fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

axes[0].plot(t, vort, color='#2ecc71', linewidth=2)
axes[0].set_ylabel("Max vorticity ‖ω‖∞")
axes[0].set_title("Figure 4: Best config diagnostics (TG + k=2, ε=0.27, ν=0.0001, score=565.15)")
axes[0].grid(alpha=0.3)

axes[1].semilogy(t, enst, color='#3498db', linewidth=1.5)
axes[1].set_ylabel("Enstrophy (log)")
axes[1].grid(alpha=0.3)

axes[2].plot(t, ener, color='#e67e22', linewidth=1.5)
axes[2].set_ylabel("Energy")
axes[2].set_xlabel("Simulation time")
axes[2].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(f"{OUT}/fig4_best_config_diagnostics.png")
plt.close(fig)
print("Saved fig4_best_config_diagnostics.png")

# ─── Figure 5: Search efficiency ─────────────────────────────────────────────
# Running best vs experiment #, with annotations for key decisions
fig, ax = plt.subplots(figsize=(9, 4.5))

ax.plot(xs, running_best, 'k-o', linewidth=2, markersize=4, zorder=3)
ax.fill_between(xs, 0, running_best, alpha=0.1, color='blue')

# Annotate key hypothesis moments
annotations = []
for i, e in enumerate(experiments):
    name = e["config"]["name"]
    s = e["score"]
    if "taylor_green" in name and "nu0001" in name and "multiscale" not in name:
        annotations.append((i+1, running_best[i], "Pure TG\nbaseline", "below"))
    elif name == "tg_multiscale_nu0001" and s > 490:
        annotations.append((i+1, running_best[i], "k=2 seed\ndiscovery", "above"))
    elif name == "tg_multiscale_eps027_nu0001" and s > 560:
        annotations.append((i+1, running_best[i], "Optimal\nε=0.27", "above"))
    elif "colliding" in name:
        annotations.append((i+1, s, "Rings:\nlow KE", "below"))
    elif "reconnect" in name and "nu0001" in name:
        annotations.append((i+1, s, "Tubes:\nlow KE", "below"))

for (xi, yi, label, pos) in annotations:
    if pos == "above":
        ax.annotate(label, xy=(xi, yi), xytext=(xi+0.5, yi+15),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                    fontsize=8, color='#2c3e50')
    else:
        ax.annotate(label, xy=(xi, yi), xytext=(xi+0.5, max(50, yi-60)),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                    fontsize=8, color='#e74c3c')

ax.set_xlabel("Experiment number")
ax.set_ylabel("Running best score")
ax.set_title("Figure 5: Convergence of auto-research loop — hypothesis-driven search")
ax.set_xlim(0.3, len(xs) + 0.7)
ax.set_ylim(0)
ax.grid(alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_convergence.png")
plt.close(fig)
print("Saved fig5_convergence.png")

# ─── Figure 6: Approach comparison ───────────────────────────────────────────
# Group experiments by approach class, show max score per class
classes = {
    "Pure TG\n(no seed)": [],
    "TG + k=2 seed\n(bonus)": [],
    "TG + k=2 seed\n(no bonus)": [],
    "TG variants\n(3D / asym)": [],
    "Vortex rings /\ntubes": [],
    "High Re\n(lower ν)": [],
}

for e in experiments:
    name = e["config"]["name"]
    s = e["score"]
    if "colliding" in name or "reconnect" in name:
        classes["Vortex rings /\ntubes"].append(s)
    elif "multiscale" in name and "3d" not in name and "k23" not in name and "A22" not in name and "asymm" not in name:
        nu = e["config"].get("nu", 0.0001)
        if nu < 0.00009:
            classes["High Re\n(lower ν)"].append(s)
        else:
            eps_val = EPS_MAP.get(name, 0)
            if s > 450:
                classes["TG + k=2 seed\n(bonus)"].append(s)
            else:
                classes["TG + k=2 seed\n(no bonus)"].append(s)
    elif "taylor_green" in name and "multiscale" not in name:
        nu = e["config"].get("nu", 0.0001)
        if nu < 0.00009:
            classes["High Re\n(lower ν)"].append(s)
        else:
            classes["Pure TG\n(no seed)"].append(s)
    elif "low_nu" in name or "nu00005" in name or "nu000095" in name or "nu00008" in name:
        classes["High Re\n(lower ν)"].append(s)
    else:
        classes["TG variants\n(3D / asym)"].append(s)

fig, ax = plt.subplots(figsize=(9, 4.5))
class_names = list(classes.keys())
max_scores = [max(v) if v else 0 for v in classes.values()]
mean_scores = [np.mean(v) if v else 0 for v in classes.values()]
counts = [len(v) for v in classes.values()]

x = np.arange(len(class_names))
bars = ax.bar(x, max_scores, width=0.5, color='#3498db', alpha=0.7, label='Best score in class')
ax.scatter(x, mean_scores, color='#e74c3c', s=60, zorder=5, label='Mean score in class')

for xi, (mx, mn, c) in enumerate(zip(max_scores, mean_scores, counts)):
    ax.text(xi, mx + 5, f"n={c}", ha='center', fontsize=8, color='gray')

ax.axhline(565.15, color='red', linestyle=':', linewidth=1, alpha=0.7, label='Global best = 565')
ax.set_xticks(x)
ax.set_xticklabels(class_names, fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Figure 6: Score by approach class — best and mean within each class")
ax.legend(framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 630)
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_approach_comparison.png")
plt.close(fig)
print("Saved fig6_approach_comparison.png")

print(f"\nAll figures saved to ./{OUT}/")
