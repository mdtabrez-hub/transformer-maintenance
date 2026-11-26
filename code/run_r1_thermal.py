#!/usr/bin/env python3
"""
run_r1_thermal.py -- Revision-1 hot-spot estimation study (PINN and baselines).

    python3 run_r1_thermal.py --workers 2 --out ../results/r1_thermal.jsonl

Every job is independent and appended to a JSON-lines file, so an interrupted
run resumes where it stopped. Seeds: trajectory seed 100+s, training seed s,
parameter-mismatch seed s+991, exactly as in the original study.

Author: Md Tabrez
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim import r1_thermal as T
from sim.models import perturb_params

SEEDS = (0, 1, 2, 3, 4)
SCEN = ("S0", "S1", "S2", "S3", "S4", "S5")
LAMBDAS = (0.0, 0.03, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
LABELS = (0, 2, 6, 20, 60, 200)


def scenario_inputs(scen, s, n_hs, ic_mode):
    tr = T.make_scenario(scen, 100 + s)
    gain, off = (1.02, 1.0) if scen == "S5" else (1.0, 0.0)
    obs = T.make_observations(tr, s, n_obs_hs=n_hs, gain=gain, offset=off)
    tp = perturb_params(tr["nominal"], 0.10, s + 991)
    ic = T.initial_condition(tr, tp, ic_mode)
    return tr, obs, tp, ic


def job_list():
    J = []
    # 1. benchmark across ground-truth generators (assumed initial condition)
    for scen in SCEN:
        for s in SEEDS:
            J.append(dict(kind="bench", scen=scen, seed=s, n_hs=6, est="pinn", lam_p=10.0))
            J.append(dict(kind="bench", scen=scen, seed=s, n_hs=0, est="pinn", lam_p=10.0))
            J.append(dict(kind="bench", scen=scen, seed=s, n_hs=6, est="nn", lam_p=0.0))
            J.append(dict(kind="bench", scen=scen, seed=s, n_hs=6, est="pinn_inv", lam_p=10.0))
    # 2. original protocol (exact initial condition), five seeds
    for s in SEEDS:
        for est, lp in (("pinn", 10.0), ("nn", 0.0)):
            J.append(dict(kind="table7", scen="S0", seed=s, n_hs=6, est=est, lam_p=lp))
    # 3. physics-weight sweep, five seeds
    for s in SEEDS:
        for lp in LAMBDAS:
            J.append(dict(kind="lambda", scen="S0", seed=s, n_hs=6, est="pinn", lam_p=lp))
    # 4. label-efficiency sweep, five seeds
    for s in SEEDS:
        for nl in LABELS:
            J.append(dict(kind="labels", scen="S0", seed=s, n_hs=nl, est="pinn", lam_p=10.0))
            J.append(dict(kind="labels", scen="S0", seed=s, n_hs=nl, est="nn", lam_p=0.0))
    # 5. one-at-a-time parameter mismatch, seed 0
    for nm in T.PNAMES:
        for sign in (+1, -1):
            J.append(dict(kind="oat", scen="S0", seed=0, n_hs=6, est="pinn", lam_p=10.0,
                          param=nm, sign=sign))
    return J


def key(j):
    return json.dumps({k: j[k] for k in sorted(j)}, sort_keys=True)


def run(j):
    t0 = time.time()
    ic_mode = "assumed" if j["kind"] == "bench" else "exact"
    tr, obs, tp, ic = scenario_inputs(j["scen"], j["seed"], j["n_hs"], ic_mode)
    if j["kind"] == "oat":
        import copy
        tp = copy.deepcopy(tr["nominal"])
        setattr(tp, j["param"], getattr(tp, j["param"]) * (1 + 0.10 * j["sign"]))
        ic = T.initial_condition(tr, tp, "exact")
    extra = {}
    if j["est"] == "pinn_inv":
        a, b, sc = T.est_pinn(tr, obs, tp, ic, lam_p=j["lam_p"], seed=j["seed"], inverse=True)
        extra["scales"] = sc
    else:
        a, b = T.est_pinn(tr, obs, tp, ic, lam_p=j["lam_p"], seed=j["seed"])
    m = T.metrics(tr, a, b)
    out = dict(j, **m, **extra, runtime_s=time.time() - t0)
    if j["kind"] == "bench" and j["seed"] == 0 and j["n_hs"] == 6:
        out["trace"] = dict(Th_hat=(tr["Theta_A"] + a + b)[::3].round(3).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="../results/r1_thermal.jsonl")
    ap.add_argument("--kinds", default="bench,table7,lambda,labels,oat")
    args = ap.parse_args()
    kinds = set(args.kinds.split(","))
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            d = json.loads(line)
            done.add(d["key"])
    todo = [j for j in job_list() if j["kind"] in kinds and key(j) not in done]
    print(f"{len(todo)} jobs to run ({len(done)} already done)", flush=True)
    with Pool(args.workers) as pool, open(args.out, "a") as f:
        for r in pool.imap_unordered(run, todo):
            r["key"] = key({k: r[k] for k in r if k in
                            ("kind", "scen", "seed", "n_hs", "est", "lam_p", "param", "sign")})
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"{r['kind']:6s} {r['scen']} s{r['seed']} hs{r['n_hs']:<3d} {r['est']:8s} "
                  f"lp={r['lam_p']:<6g} RMSE={r['rmse']:.3f} ({r['runtime_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
