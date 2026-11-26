#!/usr/bin/env python3
"""
run_all.py -- Execute the complete simulation study and serialise every result.

    python3 run_all.py [--quick]

Produces results/results.json, from which all figures and every number quoted
in the manuscript are generated. The run is fully deterministic given
sim.core.GLOBAL_SEED.

Author: Md Tabrez
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.core import SimConfig, GLOBAL_SEED, FAULT_CLASSES, CHANNEL_NAMES
from sim import experiments as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="reduced budget for a fast smoke run")
    ap.add_argument("--out", default="results/results.json")
    ap.add_argument("--stage", default="all",
                    choices=["all", "A1", "A2", "A3", "A4", "B1", "B2",
                             "C1", "C2", "merge"],
                    help="A=PINN, B=classification/security/edge, C=noise/RUL")
    args = ap.parse_args()

    q = args.quick
    cfg = SimConfig()
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)

    stage = args.stage
    n_ep = 40 if q else 130
    epochs_clf = 25 if q else 80
    pinn_ep = 600 if q else 1800
    t_start = time.time()

    R = {
        "meta": {
            "seed": GLOBAL_SEED,
            "config": cfg.to_dict(),
            "fault_classes": FAULT_CLASSES,
            "channels": CHANNEL_NAMES,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "quick": q,
        }
    }

    print("=" * 74)
    print("EDGE-AI PHYSICS-INFORMED DIGITAL TWIN -- SIMULATION STUDY")
    print(f"seed = {GLOBAL_SEED}   stage = {stage}   mode = {'quick' if q else 'full'}")
    print("=" * 74)

    os.makedirs("results", exist_ok=True)

    if stage in ("A1", "all"):
        R["pinn_main"] = E.exp_pinn_main(cfg, seeds=(0,) if q else (0, 1, 2),
                                         epochs=pinn_ep)
    if stage in ("A2", "all"):
        R["pinn_lambda"] = E.exp_pinn_lambda(cfg, epochs=pinn_ep)
    if stage in ("A3", "all"):
        R["pinn_data_eff"] = E.exp_pinn_data_efficiency(cfg, epochs=pinn_ep)
    if stage in ("A4", "all"):
        R["pinn_multiphase"] = E.exp_pinn_multiphase(
            cfg, epochs=max(600, pinn_ep - 600))

    if stage in ("B1", "all"):
        print("  [--] building classification dataset", flush=True)
        t0 = time.time()
        D = E.build_clf_data(cfg, n_ep=n_ep, seq_len=16,
                             cache="results/clf_cache.npz")
        print(f"       {D['Xseq'].shape} in {time.time()-t0:.0f} s", flush=True)
        R["dataset"] = dict(shape=list(D["Xseq"].shape), d=int(D["d"]),
                            n_train=int(len(D["itr"])), n_val=int(len(D["iva"])),
                            n_test=int(len(D["ite"])))
        clf, models = E.exp_classification(cfg, D, epochs=epochs_clf)
        R["classification"] = clf

    if stage in ("B1", "all"):
        R["train_fraction"] = E.exp_train_fraction(
            cfg, D, epochs=max(20, epochs_clf - 25))
        R["security"] = E.exp_security(cfg, D, models)
        R["edge_cost"] = E.exp_edge_cost(cfg, D, models)
        R["bandwidth"] = E.exp_bandwidth(
            cfg, R["security"]["anomaly_flag_rate_normal"])
        R["placement"] = E.exp_sensor_placement(cfg)

    if stage in ("B2", "all"):
        R.pop("classification", None); R.pop("dataset", None)
        R["physics_ablation"] = E.exp_physics_feature_ablation(
            cfg, epochs=max(20, epochs_clf - 25), n_ep=25 if q else 70)

    if stage in ("C1", "all"):
        R["noise_sensitivity"] = E.exp_noise_sensitivity(
            cfg, epochs=max(20, epochs_clf - 35), n_ep=25 if q else 45)
    if stage in ("C2", "all"):
        R["rul"] = E.exp_rul(cfg)

    if stage == "merge":
        merged = {}
        for st in ("A1", "A2", "A3", "A4", "B1", "B2", "C1", "C2"):
            fp = f"results/stage{st}.json"
            if os.path.exists(fp):
                d = json.load(open(fp))
                meta = d.pop("meta", {})
                merged.update(d)
                merged.setdefault("meta", meta)
                merged["meta"][f"runtime_stage{st}_s"] = meta.get("runtime_s")
        with open(args.out, "w") as f:
            json.dump(merged, f, indent=1, default=float)
        print(f"merged {len(merged)} top-level keys -> {args.out}")
        return

    R["meta"]["runtime_s"] = time.time() - t_start
    out = args.out if stage == "all" else f"results/stage{stage}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(R, f, indent=1, default=float)

    print("=" * 74)
    print(f"complete in {R['meta']['runtime_s']/60:.1f} min -> {out}")
    print("=" * 74)


if __name__ == "__main__":
    main()
