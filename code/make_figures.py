#!/usr/bin/env python3
"""
make_figures.py -- Generate every figure in the manuscript from results.json.

    python3 make_figures.py [--results results/results.json] [--outdir figs]

All figures are sized for IEEE two-column format (3.5 in single column,
7.16 in double column) and written as 400 dpi PNG plus vector PDF.

Author: Md Tabrez
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------- style ----
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
    "figure.dpi": 130,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C1, C2, C3, C4 = "#1f4e79", "#c0392b", "#2e8b57", "#e08a1e"
C5, C6 = "#7d3c98", "#5d6d7e"
COL1, COL2 = 3.5, 7.16
OUT = "figs"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"    {name}")


def _box(ax, x, y, w, h, text, fc, ec=None, fs=6.3, tc="white", bold=True):
    ec = ec or fc
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, fontweight="bold" if bold else "normal",
            linespacing=1.35)


def _arrow(ax, p0, p1, color="#34495e", ls="-", lw=0.9, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=7,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=1, shrinkB=1, zorder=4))


# =====================================================================
# Fig. 1  Four-layer system architecture
# =====================================================================
def fig_architecture(R):
    fig, ax = plt.subplots(figsize=(COL2, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off"); ax.grid(False)

    layers = [
        (4.85, "#eaf1f8", "LAYER 1  \u2014  SIMULATED SENSING", C1),
        (3.35, "#fdeeec", "LAYER 2  \u2014  EDGE PROCESSING", C2),
        (1.85, "#e9f5ee", "LAYER 3  \u2014  DIGITAL TWIN CORE", C3),
        (0.35, "#fdf3e4", "LAYER 4  \u2014  APPLICATION", C4),
    ]
    for y, fcb, title, ec in layers:
        ax.add_patch(Rectangle((0.15, y), 9.7, 1.28, fc=fcb, ec=ec, lw=0.8,
                               zorder=0, alpha=0.85))
        ax.text(0.30, y + 1.12, title, fontsize=6.4, color=ec, fontweight="bold")

    b = [
        # layer 1
        (0.45, 4.95, 1.65, 0.62, "IEC 60076-7\nthermal model", C1),
        (2.30, 4.95, 1.65, 0.62, "Multi-phase\nload model", C1),
        (4.15, 4.95, 1.65, 0.62, "Fault injection\n(5 modes)", C1),
        (6.00, 4.95, 1.65, 0.62, "Sensor noise +\ncalibration error", C1),
        (7.85, 4.95, 1.65, 0.62, "6-channel\nwaveform synth.", C1),
        # layer 2
        (0.45, 3.45, 1.90, 0.62, "Preprocessing\n(Alg. 1)  108-D", C2),
        (2.55, 3.45, 1.75, 0.62, "VAE integrity\nvalidator", C2),
        (4.50, 3.45, 1.75, 0.62, "PINN thermal\nstate estimator", C2),
        (6.45, 3.45, 1.75, 0.62, "Distilled student\nGRU + RF", C2),
        (8.40, 3.45, 1.10, 0.62, "HI / RUL\n(Weibull)", C2),
        # layer 3
        (0.45, 1.95, 2.05, 0.62, "IEC 61850-7-4\nlogical nodes", C3),
        (2.70, 1.95, 2.05, 0.62, "CIM / IEC 61970\nasset ontology", C3),
        (4.95, 1.95, 2.10, 0.62, "OPC UA Part 17\nPubSub", C3),
        (7.25, 1.95, 2.25, 0.62, "IEC 62443-3-3\nsecurity zones", C3),
        # layer 4
        (0.45, 0.45, 2.55, 0.62, "Asset health\nvisualisation", C4),
        (3.20, 0.45, 2.55, 0.62, "Maintenance\nprioritisation", C4),
        (5.95, 0.45, 3.55, 0.62, "CMMS work-order generation", C4),
    ]
    for x, y, w, h, t, c in b:
        _box(ax, x, y, w, h, t, c, fs=5.9)

    for x in (2.4, 5.0, 7.6):
        _arrow(ax, (x, 4.93), (x, 4.09))
        _arrow(ax, (x, 3.43), (x, 2.59))
        _arrow(ax, (x, 1.93), (x, 1.09))
    ax.text(9.72, 4.5, "quantised\nfeatures", fontsize=5.4, ha="right", color="#555")
    ax.text(9.72, 3.0, "HI / RUL /\nalarm state", fontsize=5.4, ha="right", color="#555")
    save(fig, "fig01_architecture")


# =====================================================================
# Fig. 2  Sensor placement graph
# =====================================================================
def fig_placement(R):
    import networkx as nx
    P = R["placement"]
    G = nx.Graph(); G.add_edges_from([tuple(e) for e in P["edges"]])
    S = set(P["sensors"])
    pos = nx.spring_layout(G, seed=7, iterations=180)

    fig, ax = plt.subplots(figsize=(COL1, 2.7))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#b8c4cc", width=0.55)
    others = [n for n in G.nodes if n not in S]
    nx.draw_networkx_nodes(G, pos, nodelist=others, ax=ax, node_size=26,
                           node_color="#dfe6ea", edgecolors="#7f8c8d", linewidths=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=sorted(S), ax=ax, node_size=58,
                           node_color=C2, edgecolors="#7b241c", linewidths=0.7)
    ax.set_axis_off(); ax.grid(False)
    ax.set_title(f"{P['n_sensors']} sensing nodes dominate {P['n_assets']} assets "
                 f"({P['coverage_pct']:.0f}% coverage)", fontsize=6.8)
    save(fig, "fig02_sensor_placement")


# =====================================================================
# Fig. 3  PINN architecture
# =====================================================================
def fig_pinn_arch(R):
    fig, ax = plt.subplots(figsize=(COL2, 2.55))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off"); ax.grid(False)

    _box(ax, 0.15, 1.75, 1.05, 0.85, "$t$\n(normalised)", C6, fs=6.2)
    _box(ax, 1.42, 1.75, 1.20, 0.85, "Fourier\nfeatures\n$[t,\\sin,\\cos]$", C5, fs=5.9)

    for i, x in enumerate(np.linspace(2.95, 5.55, 5)):
        _box(ax, x, 1.60, 0.46, 1.15, "", C1, fs=5.5)
        ax.text(x + 0.23, 2.17, "64", fontsize=5.6, color="white",
                ha="center", va="center", rotation=90, fontweight="bold")
    ax.text(4.30, 3.00, "5 hidden layers, tanh", fontsize=6.2, ha="center", color=C1)

    _box(ax, 6.35, 2.25, 1.35, 0.62, "$\\Delta\\Theta_{TO}(t)$", C3, fs=6.4)
    _box(ax, 6.35, 1.45, 1.35, 0.62, "$\\Delta\\Theta_{H}(t)$", C3, fs=6.4)
    _box(ax, 8.05, 1.85, 1.75, 0.62,
         "$\\Theta_H=\\Theta_A+\\Delta\\Theta_{TO}+\\Delta\\Theta_{H}$", C3, fs=5.7)

    _arrow(ax, (1.20, 2.17), (1.42, 2.17))
    _arrow(ax, (2.62, 2.17), (2.95, 2.17))
    _arrow(ax, (6.05, 2.17), (6.35, 2.55))
    _arrow(ax, (6.05, 2.17), (6.35, 1.77))
    _arrow(ax, (7.70, 2.17), (8.05, 2.17))

    # loss terms
    losses = [
        (0.15, 0.15, 2.35, 0.72, "$\\mathcal{L}_{data}$\nsparse $\\Theta_{TO}$ obs.", C2),
        (2.72, 0.15, 3.30, 0.72,
         "$\\mathcal{L}_{phys}$: autodiff residual of\n"
         "IEC 60076-7 ODEs at collocation pts", C2),
        (6.24, 0.15, 1.65, 0.72, "$\\mathcal{L}_{bc}$\ninitial condition", C2),
        (8.10, 0.15, 1.72, 0.72,
         "$\\mathcal{L}=\\lambda_d\\mathcal{L}_{d}+\\lambda_p\\mathcal{L}_{p}$\n"
         "$+\\lambda_{bc}\\mathcal{L}_{bc}$", "#5b2c6f"),
    ]
    for x, y, w, h, t, c in losses:
        _box(ax, x, y, w, h, t, c, fs=5.6)
    _arrow(ax, (4.30, 1.55), (4.35, 0.90), color=C2, ls="--")
    ax.text(4.55, 1.22, r"$\partial/\partial t$ via autodiff", fontsize=5.5, color=C2)
    _arrow(ax, (9.00, 0.90), (9.00, 1.83), color="#5b2c6f", ls="--")
    save(fig, "fig03_pinn_architecture")


# =====================================================================
# Fig. 4  Knowledge distillation pipeline
# =====================================================================
def fig_distill(R):
    ec = R["classification"].get("ensemble_weights", {})
    fig, ax = plt.subplots(figsize=(COL2, 2.35))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off"); ax.grid(False)

    ax.add_patch(Rectangle((0.15, 1.95), 4.35, 1.45, fc="#eaf1f8", ec=C1,
                           lw=0.8, alpha=0.8))
    ax.text(0.30, 3.22, "OFFLINE  (workstation)", fontsize=6.2, color=C1,
            fontweight="bold")
    _box(ax, 0.40, 2.55, 1.75, 0.55, "Bi-GRU 1$\\times$96", C1, fs=6.0)
    _box(ax, 0.40, 2.02, 1.75, 0.47, "RF (500, depth 20)", C1, fs=6.0)
    _box(ax, 2.45, 2.20, 1.85, 0.80, "TEACHER\nensemble", C1, fs=6.4)

    ax.add_patch(Rectangle((5.45, 1.95), 4.40, 1.45, fc="#fdeeec", ec=C2,
                           lw=0.8, alpha=0.8))
    ax.text(5.60, 3.22, "EDGE  (deployed)", fontsize=6.2, color=C2, fontweight="bold")
    _box(ax, 5.70, 2.55, 1.75, 0.55, "Uni-GRU 1$\\times$32", C2, fs=6.0)
    _box(ax, 5.70, 2.02, 1.75, 0.47, "RF (50, depth 10)", C2, fs=6.0)
    _box(ax, 7.75, 2.20, 1.95, 0.80, "STUDENT\nensemble  (INT8)", C2, fs=6.4)

    _arrow(ax, (4.35, 2.60), (5.68, 2.60), color=C5, lw=1.3)
    ax.text(5.00, 2.78, "distillation", fontsize=6.0, color=C5, ha="center",
            fontweight="bold")

    _box(ax, 1.45, 0.85, 7.10, 0.72,
         "$\\mathcal{L}_{distill}=\\alpha\\,\\mathcal{L}_{CE}(y,z_s)"
         "\\;+\\;\\beta T^2\\mathcal{L}_{KL}(\\sigma(z_t/T)\\,\\|\\,\\sigma(z_s/T))"
         "\\;+\\;\\gamma\\,\\mathcal{L}_{MSE}(h_t,h_s)$", C5, fs=7.0)
    ax.text(5.00, 0.42, r"$T=4$,   $\gamma=0.1$;   $(\alpha,\beta)$ selected on validation from {(0.3, 0.6), (0.5, 0.4), (0.7, 0.2), (0.9, 0.05)}",
            fontsize=6.2, ha="center", color="#444")
    save(fig, "fig04_distillation")


# =====================================================================
# Fig. 5  PINN hot-spot reconstruction
# =====================================================================
def fig_pinn_recon(R):
    c = R["pinn_main"]["curves"]
    t = np.array(c["t_min"]) / 1440.0
    fig = plt.figure(figsize=(COL2, 3.3))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.9], hspace=0.75, wspace=0.26)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(t, c["K"], color=C6, lw=0.8, label="load factor $K$")
    ax0.set_ylabel("$K$ (p.u.)")
    ax0b = ax0.twinx()
    ax0b.plot(t, c["Theta_A"], color=C4, lw=0.8, label="$\\Theta_A$")
    ax0b.set_ylabel("$\\Theta_A$ ($^\\circ$C)", color=C4)
    ax0b.grid(False)
    ax0.set_xlabel("time (days)")
    ax0.set_title("(a)  Exogenous forcing", fontsize=7.2, loc="left")

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, c["Th_true"], color="#2c3e50", lw=1.2, label="ground truth")
    ax1.plot(t, c["Th_pinn"], color=C2, lw=0.9, ls="--", label="PINN")
    ax1.plot(t, c["Th_data"], color=C6, lw=0.8, ls=":", label="data-driven")
    ax1.set_xlabel("time (days)"); ax1.set_ylabel("$\\Theta_H$ ($^\\circ$C)")
    ax1.legend(frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.3, columnspacing=0.9)
    ax1.set_ylim(None, max(c["Th_true"]) * 1.28)
    ax1.set_title("(b)  Hot-spot reconstruction", fontsize=7.2, loc="left")

    ax2 = fig.add_subplot(gs[1, 1])
    for h, col, lab in ((c["hist_pinn"], C2, "PINN"), (c["hist_data"], C6, "data-driven")):
        h = np.array(h)
        ax2.semilogy(h[:, 0], np.maximum(h[:, 1], 1e-6), color=col, lw=0.9, label=lab)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("total loss")
    ax2.legend(frameon=False)
    ax2.set_title("(c)  Convergence", fontsize=7.2, loc="left")
    save(fig, "fig05_pinn_reconstruction")


# =====================================================================
# Fig. 6  PINN ablations (lambda_p, label efficiency, multi-phase)
# =====================================================================
def fig_pinn_ablation(R):
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 1.95))
    L = R["pinn_lambda"]
    lam = [max(d["lam_p"], 3e-3) for d in L]
    ax = axs[0]
    ax.semilogx(lam, [d["rmse"] for d in L], "o-", color=C1, ms=3)
    best = min(L, key=lambda d: d["rmse"])
    ax.axvline(max(best["lam_p"], 3e-3), color=C2, ls="--", lw=0.8)
    ax.text(max(best["lam_p"], 3e-3) * 1.25, ax.get_ylim()[1] * 0.75,
            f"$\\lambda_p^\\star$={best['lam_p']:g}", color=C2, fontsize=6.2)
    ax.set_xlabel("physics weight $\\lambda_p$"); ax.set_ylabel("$\\Theta_H$ RMSE (K)")
    ax.set_title("(a)  Physics weight", fontsize=7.2, loc="left")

    Dq = R["pinn_data_eff"]
    ax = axs[1]
    nl = [max(d["n_labels"], 0.7) for d in Dq]
    ax.semilogx(nl, [d["pinn_rmse"] for d in Dq], "o-", color=C2, ms=3, label="PINN")
    ax.semilogx(nl, [d["data_rmse"] for d in Dq], "s--", color=C6, ms=3,
                label="data-driven")
    ax.set_xlabel("hot-spot labels $N_{hs}$"); ax.set_ylabel("$\\Theta_H$ RMSE (K)")
    ax.legend(frameon=False)
    ax.set_title("(b)  Label efficiency", fontsize=7.2, loc="left")

    M = R["pinn_multiphase"]
    ax = axs[2]
    lab = ["uncoupled", "coupled"]
    xx = np.arange(2); w = 0.36
    ax.bar(xx - w / 2, [m["rmse"] for m in M], w, color=C1, label="overall")
    ax.bar(xx + w / 2, [m["imbalance_rmse"] for m in M], w, color=C4,
           label="phase imbalance")
    ax.set_xticks(xx); ax.set_xticklabels(lab)
    ax.set_ylabel("RMSE (K)"); ax.legend(frameon=False)
    ax.set_title("(c)  Multi-phase coupling", fontsize=7.2, loc="left")
    fig.tight_layout()
    save(fig, "fig06_pinn_ablation")


# =====================================================================
# Fig. 7  Classification performance comparison
# =====================================================================
def fig_perf(R):
    M = R["classification"]["metrics"]
    names = list(M.keys())
    short = [n.replace("Student, ", "Student\n").replace("Teacher ", "Teacher\n")
             .replace(" (proposed)", "\n(proposed)").replace("RandomForest", "Random\nForest")
             for n in names]
    mets = ["accuracy", "precision", "recall", "f1"]
    labs = ["Accuracy", "Precision", "Recall", "F1-score"]
    cols = [C1, C3, C4, C2]

    fig, ax = plt.subplots(figsize=(COL2, 2.35))
    x = np.arange(len(names)); w = 0.20
    for i, (m, lb, c) in enumerate(zip(mets, labs, cols)):
        v = [M[n][m] for n in names]
        b = ax.bar(x + (i - 1.5) * w, v, w, label=lb, color=c, edgecolor="white", lw=0.3)
        if m == "accuracy":
            for bi, vi in zip(b, v):
                ax.text(bi.get_x() + bi.get_width() / 2, vi + 0.008, f"{vi*100:.1f}",
                        ha="center", fontsize=5.4, color=C1)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=5.9)
    ax.set_ylabel("score"); ax.set_ylim(0.5, 1.03)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig07_model_performance")


# =====================================================================
# Fig. 8  Confusion matrix
# =====================================================================
def fig_confusion(R):
    cm = np.array(R["classification"]["confusion"], dtype=float)
    cls = R["meta"]["fault_classes"]
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(COL1, 3.05))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    short = [c.replace(" / ", "/\n").replace(" ", "\n") for c in cls]
    ax.set_xticks(range(len(cls))); ax.set_yticks(range(len(cls)))
    ax.set_xticklabels(short, fontsize=5.4); ax.set_yticklabels(short, fontsize=5.4)
    for i in range(len(cls)):
        for j in range(len(cls)):
            ax.text(j, i, f"{int(cm[i,j])}\n{cmn[i,j]*100:.0f}%", ha="center",
                    va="center", fontsize=5.2,
                    color="white" if cmn[i, j] > 0.55 else "#2c3e50")
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=5.8)
    save(fig, "fig08_confusion_matrix")


# =====================================================================
# Fig. 9  ROC curves
# =====================================================================
def fig_roc(R):
    roc = R["classification"]["roc"]
    fig, ax = plt.subplots(figsize=(COL1, 2.55))
    cols = [C1, C2, C3, C4, C5, C6]
    for (k, v), c in zip(roc.items(), cols):
        ax.plot(v["fpr"], v["tpr"], color=c, lw=1.0,
                label=f"{k.split('/')[0].strip()[:18]} ({v['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#95a5a6", lw=0.6, ls="--")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.legend(frameon=False, loc="lower right", fontsize=5.6)
    save(fig, "fig09_roc_curves")


# =====================================================================
# Fig. 10  Training convergence
# =====================================================================
def fig_convergence(R):
    H = R["classification"]["hists"]
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.0))
    cols = {"GRU": C6, "CNN-LSTM": C4, "Teacher": C1, "Student-KD": C2}
    for k, h in H.items():
        h = np.array(h)
        axs[0].plot(h[:, 0], h[:, 1], color=cols.get(k, C5), lw=0.9, label=k)
        axs[1].plot(h[:, 0], h[:, 2] * 100, color=cols.get(k, C5), lw=0.9, label=k)
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("training loss")
    axs[0].set_yscale("log"); axs[0].legend(frameon=False, fontsize=5.8)
    axs[0].set_title("(a)  Loss", fontsize=7.2, loc="left")
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("validation accuracy (%)")
    axs[1].set_title("(b)  Validation accuracy", fontsize=7.2, loc="left")
    fig.tight_layout()
    save(fig, "fig10_convergence")


# =====================================================================
# Fig. 11  Edge cost: latency / model size / FLOPs
# =====================================================================
def fig_edge_cost(R):
    rows = R["edge_cost"]["rows"]
    proj = R["edge_cost"]["projections"]
    names = ["Teacher", "Student"]
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.05))

    ax = axs[0]
    ax.bar(names, [r["mflops"] for r in rows], color=[C1, C2], width=0.5)
    for i, r in enumerate(rows):
        ax.text(i, r["mflops"] * 1.03, f"{r['mflops']:.2f}", ha="center", fontsize=6.0)
    ax.set_ylabel("MFLOP per inference")
    ax.set_title("(a)  Compute", fontsize=7.2, loc="left")

    ax = axs[1]
    w = 0.35; x = np.arange(2)
    ax.bar(x - w / 2, [r["size_fp32_kB"] for r in rows], w, color=C6, label="NN FP32")
    ax.bar(x + w / 2, [r["total_int8_kB"] for r in rows], w, color=C3,
           label="NN INT8 + RF")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("model size (kB)"); ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=5.8)
    ax.set_title("(b)  Memory", fontsize=7.2, loc="left")

    ax = axs[2]
    tg = sorted({p["target"] for p in proj})
    x = np.arange(len(tg)); w = 0.35
    for i, mdl in enumerate([r["model"] for r in rows]):
        v = [next(p["projected_latency_ms"] for p in proj
                  if p["target"] == t and p["model"] == mdl) for t in tg]
        ax.bar(x + (i - 0.5) * w, v, w, color=[C1, C2][i], label=names[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["Cortex-A72\n(projected)", "Cortex-M4F\n(projected)"],
                       fontsize=5.7)
    ax.set_yscale("log"); ax.set_ylabel("latency (ms)")
    ax.axhline(5000, color=C2, ls="--", lw=0.7)
    ax.text(1.45, 5600, "5 s budget", fontsize=5.5, color=C2, ha="right")
    ax.legend(frameon=False, fontsize=5.8)
    ax.set_title("(c)  Projected latency", fontsize=7.2, loc="left")
    fig.tight_layout()
    save(fig, "fig11_edge_cost")


# =====================================================================
# Fig. 12  RUL estimation
# =====================================================================
def fig_rul(R):
    U = R["rul"]
    p = np.array(U["pred"]); a = np.array(U["actual"])
    fig = plt.figure(figsize=(COL2, 2.15))
    gs = gridspec.GridSpec(1, 3, wspace=0.52)

    ax = fig.add_subplot(gs[0])
    fr = np.array(U["frac_life"])
    sc = ax.scatter(a, p, c=fr, cmap="viridis", s=6, alpha=0.75, lw=0)
    lim = [0, max(a.max(), p.max()) * 1.03]
    ax.plot(lim, lim, color="#2c3e50", lw=0.8)
    ax.plot(lim, [l * 1.2 for l in lim], color=C2, lw=0.6, ls="--")
    ax.plot(lim, [l * 0.8 for l in lim], color=C2, lw=0.6, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("actual RUL (days)"); ax.set_ylabel("predicted RUL (days)")
    ax.text(0.05, 0.93, f"$R^2$={U['r2']:.3f}\nMAE={U['mae']:.0f} d",
            transform=ax.transAxes, fontsize=6.0, va="top")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("fraction of life elapsed", fontsize=5.3, labelpad=1)
    cb.ax.tick_params(labelsize=5.5)
    ax.set_title("(a)", fontsize=7.2, loc="left")

    ax = fig.add_subplot(gs[1])
    err = p - a
    ax.hist(err, bins=42, color=C1, alpha=0.85, edgecolor="white", lw=0.2)
    ax.axvline(0, color="#2c3e50", lw=0.8)
    ax.axvline(np.median(err), color=C2, lw=0.9, ls="--")
    ax.set_xlabel("RUL error (days)"); ax.set_ylabel("count")
    ax.text(0.03, 0.95, f"RMSE={U['rmse']:.0f} d\nmedian={np.median(err):+.0f} d",
            transform=ax.transAxes, fontsize=6.0, va="top")
    ax.set_title("(b)", fontsize=7.2, loc="left")

    ax = fig.add_subplot(gs[2])
    for tr, c in zip(U["trajectories"], [C1, C2, C3, C4, C5]):
        t = np.array(tr["t_days"]); h = np.array(tr["hi"])
        ax.plot(t, h, color=c, lw=0.85)
        ax.plot(tr["eol_day"], h[-1], "x", color=c, ms=4)
    he = U.get("hi_eol", 0.55)
    for thr, lab in ((0.5 * he, "degraded"), (0.8 * he, "warning"),
                     (he, "end of life")):
        ax.axhline(thr, color="#95a5a6", lw=0.5, ls=":")
        ax.text(ax.get_xlim()[1] * 0.02, thr + 0.015, lab, fontsize=5.2, color="#7f8c8d")
    ax.set_xlabel("time (days)"); ax.set_ylabel("health index")
    ax.set_ylim(0, max(0.72, he * 1.25))
    ax.set_title("(c)", fontsize=7.2, loc="left")
    save(fig, "fig12_rul")


# =====================================================================
# Fig. 13  Bandwidth
# =====================================================================
def fig_bandwidth(R):
    B = R["bandwidth"]
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.0))
    b = B["base"]
    ax = axs[0]
    v = [b["raw_daily_MB"], b["edge_daily_MB"]]
    bars = ax.bar(["raw streaming", "edge preprocessing"], v, color=[C6, C3], width=0.5)
    for bi, vi in zip(bars, v):
        ax.text(bi.get_x() + bi.get_width() / 2, vi * 1.15, f"{vi:.1f}", ha="center",
                fontsize=6.2)
    ax.set_yscale("log"); ax.set_ylabel("upstream volume (MB/day/node)")
    ax.set_title(f"(a)  {b['reduction_pct']:.1f}% reduction", fontsize=7.2, loc="left")

    ax = axs[1]
    s = B["sweep"]
    ax.plot([d["anomaly_fraction"] * 100 for d in s],
            [d["reduction_pct"] for d in s], "o-", color=C1, ms=3)
    ax.axvline(b["anomaly_fraction"] * 100, color=C2, ls="--", lw=0.8)
    ax.text(b["anomaly_fraction"] * 100 + 1.4, 60, "operating\npoint",
            fontsize=5.8, color=C2)
    ax.set_xlabel("anomaly-flagged windows (%)")
    ax.set_ylabel("bandwidth reduction (%)")
    ax.set_ylim(0, 100)
    ax.set_title("(b)  Sensitivity", fontsize=7.2, loc="left")
    fig.tight_layout()
    save(fig, "fig13_bandwidth")


# =====================================================================
# Fig. 14  Ablation studies
# =====================================================================
def fig_ablation(R):
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 1.95))

    ax = axs[0]
    P = R["physics_ablation"]
    ks = list(P.keys())
    v = [P[k]["accuracy"] * 100 for k in ks]
    ax.bar(["sensor\nfeatures", "+ physics\nfeatures"], v, color=[C6, C2], width=0.5)
    for i, vi in enumerate(v):
        ax.text(i, vi + 0.35, f"{vi:.1f}", ha="center", fontsize=6.2)
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(min(v) - 6, max(v) + 4)
    ax.set_title("(a)  Physics features", fontsize=7.2, loc="left")

    ax = axs[1]
    M = R["classification"]["metrics"]
    keys = ["Teacher (Bi-GRU + RF)", "Student, no distillation",
            "Student, distilled (proposed)", "Student, distilled + INT8"]
    lab = ["Teacher", "Student\nscratch", "Student\n+KD", "Student\n+KD+INT8"]
    v = [M[k]["accuracy"] * 100 for k in keys if k in M]
    ax.bar(lab[:len(v)], v, color=[C1, C6, C2, C3], width=0.6)
    for i, vi in enumerate(v):
        ax.text(i, vi + 0.3, f"{vi:.1f}", ha="center", fontsize=6.0)
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(min(v) - 5, max(v) + 3)
    ax.tick_params(axis="x", labelsize=5.6)
    ax.set_title("(b)  Distillation / quantisation", fontsize=7.2, loc="left")

    ax = axs[2]
    Tf = R["train_fraction"]
    ax.plot([d["fraction"] * 100 for d in Tf], [d["accuracy"] * 100 for d in Tf],
            "o-", color=C2, ms=3)
    ax.set_xlabel("labelled training data (%)"); ax.set_ylabel("accuracy (%)")
    ax.set_xscale("log")
    ax.set_title("(c)  Label budget", fontsize=7.2, loc="left")
    fig.tight_layout()
    save(fig, "fig14_ablation")


# =====================================================================
# Fig. 15  Sensitivity analysis
# =====================================================================
def fig_sensitivity(R):
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 1.95))

    ax = axs[0]
    N = R["noise_sensitivity"]
    ax.plot([d["snr_db"] for d in N], [d["accuracy"] * 100 for d in N], "o-",
            color=C1, ms=3, label="accuracy")
    ax.plot([d["snr_db"] for d in N], [d["f1"] * 100 for d in N], "s--",
            color=C4, ms=3, label="F1")
    ax.invert_xaxis()
    ax.set_xlabel("sensor SNR (dB)"); ax.set_ylabel("score (%)")
    ax.legend(frameon=False)
    ax.set_title("(a)  Sensor noise", fontsize=7.2, loc="left")

    ax = axs[1]
    S = R["security"]["threshold_sweep"]
    ax.plot([d["k_sigma"] for d in S], [d["mean_detection"] * 100 for d in S],
            "o-", color=C3, ms=3, label="mean detection")
    ax.plot([d["k_sigma"] for d in S], [d["fpr"] * 100 for d in S], "s--",
            color=C2, ms=3, label="false positive")
    ax.axvline(3.0, color="#7f8c8d", ls=":", lw=0.8)
    ax.set_xlabel("threshold $k$ ($\\mu+k\\sigma$)"); ax.set_ylabel("rate (%)")
    ax.legend(frameon=False, fontsize=5.8)
    ax.set_title("(b)  Detector threshold", fontsize=7.2, loc="left")

    ax = axs[2]
    L = R["pinn_lambda"]
    ax.semilogx([max(d["lam_p"], 3e-3) for d in L], [d["r2"] for d in L], "o-",
                color=C5, ms=3)
    ax.set_xlabel("physics weight $\\lambda_p$"); ax.set_ylabel("$R^2$ of $\\Theta_H$")
    ax.set_ylim(min(0, min(d["r2"] for d in L)) - 0.05, 1.02)
    ax.set_title("(c)  Physics weight", fontsize=7.2, loc="left")
    fig.tight_layout()
    save(fig, "fig15_sensitivity")


# =====================================================================
# Fig. 16  Security / attack detection
# =====================================================================
def fig_security(R):
    S = R["security"]
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.15))

    ax = axs[0]
    names = [r["attack"] for r in S["rows"]]
    short = [n.replace(" (", "\n(").replace("Temperature offset spoofing", "Temperature\nspoofing")
             .replace("Feature-vector replay", "Feature\nreplay")
             .replace("Vibration spoofing", "Vibration\nspoofing")
             .replace("Combined multi-sensor", "Combined\nmulti-sensor") for n in names]
    dr = [r["detection_rate"] * 100 for r in S["rows"]]
    b = ax.bar(range(len(dr)), dr, color=C3, width=0.58)
    for i, v in enumerate(dr):
        ax.text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=5.8)
    ax.axhline(S["fpr_clean"] * 100, color=C2, ls="--", lw=0.8)
    ax.text(len(dr) - 0.4, S["fpr_clean"] * 100 + 2.5,
            f"FPR {S['fpr_clean']*100:.2f}%", fontsize=5.6, color=C2, ha="right")
    ax.set_xticks(range(len(dr))); ax.set_xticklabels(short, fontsize=5.2)
    ax.set_ylabel("detection rate (%)"); ax.set_ylim(0, 108)
    ax.set_title("(a)  Attack detection", fontsize=7.2, loc="left")

    ax = axs[1]
    cols = [C1, C2, C3, C4, C5]
    for (k, v), c in zip(S["roc"].items(), cols):
        ax.plot(v["fpr"], v["tpr"], color=c, lw=1.0,
                label=f"{k.split('(')[0].strip()[:20]} ({v['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#95a5a6", lw=0.6, ls="--")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.legend(frameon=False, loc="lower right", fontsize=5.4)
    ax.set_title("(b)  Detector ROC", fontsize=7.2, loc="left")
    fig.tight_layout()
    save(fig, "fig16_security")


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.json")
    ap.add_argument("--outdir", default="figs")
    a = ap.parse_args()
    global OUT
    OUT = a.outdir
    os.makedirs(OUT, exist_ok=True)
    R = json.load(open(a.results))
    print("  generating figures:")
    for fn in (fig_architecture, fig_placement, fig_pinn_arch, fig_distill,
               fig_pinn_recon, fig_pinn_ablation, fig_perf, fig_confusion,
               fig_roc, fig_convergence, fig_edge_cost, fig_rul, fig_bandwidth,
               fig_ablation, fig_sensitivity, fig_security):
        try:
            fn(R)
        except Exception as e:
            print(f"    !! {fn.__name__}: {type(e).__name__}: {e}")
    print("  done")


if __name__ == "__main__":
    main()
