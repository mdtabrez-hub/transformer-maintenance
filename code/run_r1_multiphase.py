#!/usr/bin/env python3
"""Re-run the inter-phase coupling ablation for the three-phase unit (Nph = 3).
The original run used SimConfig.n_phases = 6, left over from an earlier draft.
Author: Md Tabrez"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim.core import SimConfig
from sim import experiments as E
res = []
for nph in (3,):
    cfg = SimConfig(); cfg.n_phases = nph
    for seed in (0, 1, 2):
        traj = E.make_multiphase_trajectory(cfg, seed=200 + seed)
        for lc in (0.0, 1.0):
            r = E.train_pinn_multiphase(traj, cfg.thermal, n_phases=nph, lam_p=E.LAM_P_STAR,
                                        lam_couple=lc, n_obs_hs=8, epochs=1200, seed=seed)
            res.append(dict(n_phases=nph, seed=seed, lam_couple=lc, rmse=r["rmse"],
                            imbalance_rmse=r["imbalance_rmse"]))
            print(res[-1], flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", "r1_multiphase.json"), "w"), indent=1)
