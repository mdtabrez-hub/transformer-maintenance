#!/usr/bin/env python3
"""make_figures_r1.py -- Regenerate Figs. 6 and 14 from the revision results.
Author: Md Tabrez"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figures as MF
from make_figures import plt, C1, C2, C3, C4, C6, COL2, save

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def fig06(N):
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 1.95))
    ax = axs[0]
    lam = sorted((float(k), v) for k, v in N['lam'].items())
    x = [max(k, 3e-3) for k, _ in lam]
    ax.errorbar(x, [v[0] for _, v in lam], yerr=[v[1] for _, v in lam], fmt='o-', color=C1, ms=3, capsize=2, lw=1)
    ax.set_xscale('log')
    best = min(lam, key=lambda kv: kv[1][0])[0]
    ax.axvline(max(best, 3e-3), color=C2, ls='--', lw=0.8)
    ax.set_xlabel('physics weight $\\lambda_p$'); ax.set_ylabel('$\\Theta_H$ RMSE (K)')
    ax.set_title('(a)  Physics weight', fontsize=7.2, loc='left')
    ax = axs[1]
    lb = sorted((int(k), v) for k, v in N['lbl'].items())
    x = [max(k, 0.7) for k, _ in lb]
    for key, c, m, lab in (('pinn', C2, 'o-', 'PINN'), ('nn', C6, 's--', 'data-driven'),
                           ('gb', C3, '^-', 'gray-box')):
        ax.errorbar(x, [v[key][0] for _, v in lb], yerr=[v[key][1] for _, v in lb], fmt=m, color=c, ms=3,
                    capsize=2, lw=1, label=lab)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('hot-spot labels $N_{hs}$ (0 plotted at 0.7)'); ax.set_ylabel('$\\Theta_H$ RMSE (K)')
    ax.set_ylim(0.02, 40); ax.legend(frameon=False, fontsize=5.6, loc='lower left', ncol=3, handlelength=1.2, columnspacing=0.6)
    ax.set_title('(b)  Hot-spot labels', fontsize=7.2, loc='left')
    ax = axs[2]
    MP = N['mp_raw']
    xx = np.arange(2); w = 0.36
    for k, (key, c, lab) in enumerate((('rmse', C1, 'overall'), ('imbalance_rmse', C4, 'phase imbalance'))):
        m = [np.mean([d[key] for d in MP if d['lam_couple'] == lc]) for lc in (0.0, 1.0)]
        s = [np.std([d[key] for d in MP if d['lam_couple'] == lc], ddof=1) for lc in (0.0, 1.0)]
        ax.bar(xx + (k - 0.5) * w, m, w, yerr=s, color=c, label=lab, capsize=2)
    ax.set_xticks(xx); ax.set_xticklabels(['uncoupled', 'coupled'])
    ax.set_ylabel('RMSE (K)'); ax.set_ylim(0, 2.6); ax.legend(frameon=False, loc='upper center', ncol=2, fontsize=5.8)
    ax.set_title('(c)  Three-phase coupling', fontsize=7.2, loc='left')
    fig.tight_layout()
    save(fig, 'fig06_pinn_ablation')


def fig14(N):
    CV = json.load(open(os.path.join(RES, 'cv_summary.json')))['summary']
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 1.95))
    ax = axs[0]
    keys = [('RandomForest, no physics', 'RF\nsensor'), ('RandomForest', 'RF\n+physics'),
            ('Student, no distillation, no physics', 'Student\nsensor'),
            ('Student, no distillation', 'Student\n+physics')]
    v = [100 * CV[k]['mean'] for k, _ in keys]; e = [100 * CV[k]['sd'] for k, _ in keys]
    ax.bar([l for _, l in keys], v, yerr=e, color=[C6, C2, C6, C2], width=0.6, capsize=2)
    for i, vi in enumerate(v):
        ax.text(i, vi + e[i] + 0.3, f'{vi:.1f}', ha='center', fontsize=5.8)
    ax.set_ylabel('accuracy (%)'); ax.set_ylim(78, 95); ax.tick_params(axis='x', labelsize=5.6)
    ax.set_title('(a)  Physics features', fontsize=7.2, loc='left')
    ax = axs[1]
    keys = [('Teacher', 'Teacher'), ('Student, no distillation', 'Student\nscratch'),
            ('Student, distilled', 'Student\n+KD'), ('Student, distilled + INT8', 'Student\n+KD+INT8')]
    v = [100 * CV[k]['mean'] for k, _ in keys]; e = [100 * CV[k]['sd'] for k, _ in keys]
    ax.bar([l for _, l in keys], v, yerr=e, color=[C1, C6, C2, C3], width=0.6, capsize=2)
    for i, vi in enumerate(v):
        ax.text(i, vi + e[i] + 0.3, f'{vi:.1f}', ha='center', fontsize=5.8)
    ax.set_ylabel('accuracy (%)'); ax.set_ylim(78, 95); ax.tick_params(axis='x', labelsize=5.6)
    ax.set_title('(b)  Distillation / quantization', fontsize=7.2, loc='left')
    ax = axs[2]
    sz = sorted((float(k), v) for k, v in N['size'].items())
    n = [v['n'] for _, v in sz]
    ax.errorbar(n, [v['st'][0] for _, v in sz], yerr=[v['st'][1] for _, v in sz], fmt='o-', color=C2, ms=3,
                capsize=2, lw=1, label='student (scratch)')
    ax.errorbar(n, [v['rf'][0] for _, v in sz], yerr=[v['rf'][1] for _, v in sz], fmt='s--', color=C1, ms=3,
                capsize=2, lw=1, label='RF-200')
    ax.set_xscale('log'); ax.set_xlabel('training episodes'); ax.set_ylabel('accuracy (%)')
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(FixedLocator([25, 50, 100, 200, 400]))
    ax.xaxis.set_major_formatter(FixedFormatter(['25', '50', '100', '200', '400']))
    ax.legend(frameon=False, fontsize=6)
    ax.set_title('(c)  Label budget', fontsize=7.2, loc='left')
    fig.tight_layout()
    save(fig, 'fig14_ablation')


if __name__ == '__main__':
    MF.OUT = sys.argv[1] if len(sys.argv) > 1 else '../figures'
    N = json.load(open(os.path.join(RES, 'r1_numbers.json')))
    fig06(N); fig14(N)
