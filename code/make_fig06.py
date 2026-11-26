#!/usr/bin/env python3
"""
make_fig06.py -- Regenerate Fig. 6 (PINN ablations) at publication quality.

Changes relative to the original:
  * panel (c) retitled "Inter-phase coupling"
  * legend lifted clear of the bars into dedicated headroom
  * value labels on every bar, since the coupled/uncoupled difference is small
  * an explicit annotation of the relative change in phase-imbalance RMSE
  * 600 dpi raster output plus vector PDF and SVG

Author: Md Tabrez
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 7.2,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.1,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C1, C2, C3, C4 = "#1f4e79", "#c0392b", "#2e8b57", "#e08a1e"
C6 = "#5d6d7e"
COL2 = 7.16

R = json.load(open("results/results.json"))
PL = R["pinn_lambda"]
PD = R["pinn_data_eff"]
MP = R["pinn_multiphase"]

fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.05))

# ------------------------------------------------------------------ (a) ----
ax = axs[0]
lam = [max(d["lam_p"], 3e-3) for d in PL]
rmse = [d["rmse"] for d in PL]
ax.semilogx(lam, rmse, "o-", color=C1, ms=3.2, mfc="white", mew=1.0)
best = min(PL, key=lambda d: d["rmse"])
bx = max(best["lam_p"], 3e-3)
ax.axvline(bx, color=C2, ls="--", lw=0.8, zorder=1)
ax.set_ylim(0, max(rmse) * 1.22)
ax.annotate(r"$\lambda_p^{\star}=%g$" % best["lam_p"],
            xy=(bx, best["rmse"]), xytext=(bx * 1.9, max(rmse) * 0.62),
            color=C2, fontsize=6.4,
            arrowprops=dict(arrowstyle="-", color=C2, lw=0.6,
                            shrinkA=0, shrinkB=2))
ax.set_xlabel(r"physics weight $\lambda_p$")
ax.set_ylabel(r"$\Theta_H$ RMSE (K)")
ax.set_title("(a)  Physics weight", fontsize=7.6, loc="left", pad=4)

# ------------------------------------------------------------------ (b) ----
ax = axs[1]
nl = [max(d["n_labels"], 0.7) for d in PD]
ax.semilogx(nl, [d["pinn_rmse"] for d in PD], "o-", color=C2, ms=3.2,
            mfc="white", mew=1.0, label="PINN", zorder=3)
ax.semilogx(nl, [d["data_rmse"] for d in PD], "s--", color=C6, ms=3.2,
            mfc="white", mew=1.0, label="data-driven", zorder=2)
ax.set_ylim(0, max(d["data_rmse"] for d in PD) * 1.25)
ax.set_xlabel(r"hot-spot labels $N_{hs}$")
ax.set_ylabel(r"$\Theta_H$ RMSE (K)")
ax.legend(frameon=False, loc="upper right", handlelength=1.6,
          borderaxespad=0.2)
ax.set_title("(b)  Label efficiency", fontsize=7.6, loc="left", pad=4)

# ------------------------------------------------------------------ (c) ----
ax = axs[2]
overall = [m["rmse"] for m in MP]
imbal = [m["imbalance_rmse"] for m in MP]
xx = np.arange(2)
w = 0.34

b1 = ax.bar(xx - w / 2 - 0.015, overall, w, color=C1, label="overall",
            edgecolor="white", linewidth=0.4, zorder=3)
b2 = ax.bar(xx + w / 2 + 0.015, imbal, w, color=C4, label="phase imbalance",
            edgecolor="white", linewidth=0.4, zorder=3)

# dedicated headroom so the legend never sits over a bar
top = max(overall) * 1.52
ax.set_ylim(0, top)

for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + top * 0.018,
                f"{h:.3f}", ha="center", va="bottom", fontsize=5.9,
                color="#2c3e50", zorder=4)

# The coupled/uncoupled difference is too small to read off the bars, so it is
# stated directly. The label sits above the coupled bar rather than spanning the
# group, which would route a connector straight across the "overall" bars.
d_rel = 100.0 * (1 - imbal[1] / imbal[0])
ax.text(xx[1] + w / 2 + 0.015, imbal[1] + top * 0.115,
        f"−{d_rel:.1f}% vs.\nuncoupled", ha="center", va="bottom",
        fontsize=6.0, color=C4, linespacing=1.25, zorder=4)

ax.set_xlim(-0.56, 1.62)          # room for the annotation beside the last bar
ax.set_xticks(xx)
ax.set_xticklabels(["uncoupled\n$\\lambda_{couple}=0$",
                    "coupled\n$\\lambda_{couple}=1$"], fontsize=6.4)
ax.set_ylabel("RMSE (K)")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.995),
          ncol=2, handlelength=1.1, handletextpad=0.45, columnspacing=1.0,
          borderaxespad=0.0, fontsize=6.3)
ax.set_title("(c)  Inter-phase coupling", fontsize=7.6, loc="left", pad=6)

fig.tight_layout(w_pad=1.4)

os.makedirs("figs", exist_ok=True)
for ext, dpi in (("png", 600), ("pdf", 600), ("svg", 600)):
    fig.savefig(f"figs/fig06_pinn_ablation.{ext}", dpi=dpi)
plt.close(fig)
print("wrote figs/fig06_pinn_ablation.{png,pdf,svg} at 600 dpi")
