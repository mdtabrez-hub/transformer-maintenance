#!/usr/bin/env python3
"""
run_r1_edge.py -- End-to-end host timing and memory of the deployed pipelines.

Measures, on the host CPU with one thread, for the random forest alone and
for the distilled INT8 student ensemble:
  * Algorithm 1 feature extraction for one new window (6 x 1024 samples)
  * physics features, standardization and episode pooling
  * model inference on a 16-window episode
  * peak Python heap allocation during one inference (tracemalloc)
  * stored model size and tree-traversal comparison counts
These are host measurements; no embedded target was available.

Author: Md Tabrez
"""

import json
import os
import sys
import time
import tracemalloc

import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.core import SimConfig, synth_window, extract_features, physics_features
from sim import experiments as E
from sim.models import Ensemble

HERE = os.path.dirname(os.path.abspath(__file__))


def timeit(fn, n=200, warm=20):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    t = np.array(t) * 1e3
    return dict(median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)))


def peak_kb(fn):
    tracemalloc.start(); fn(); _, pk = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return pk / 1e3


def tree_stats(rf, X):
    """Mean decision-path length (comparisons) per sample, summed over trees."""
    ind = rf.decision_path(X)[0]
    nodes_per_sample = np.asarray(ind.sum(axis=1)).ravel()
    comps = nodes_per_sample - len(rf.estimators_)       # leaves are not comparisons
    return float(comps.mean()), int(sum(t.tree_.node_count for t in rf.estimators_))


def main():
    cfg = SimConfig()
    z = np.load(os.path.join(HERE, "..", "artifacts", "datasets", "classification_seq.npz"))
    X = np.concatenate([z["Xs"], z["P"]], axis=-1); y = z["y"]
    D = E._finalise_splits(cfg, X, y)
    clf, mdl = E.exp_classification(cfg, D, seed=0, epochs=80)
    rf200 = E._fit_rf(D, D["itr"], 200, 16, 0)
    student = Ensemble(mdl["rf_student"], mdl["student_q"],
                       w_rf=clf["ensemble_weights"]["student_kd"], n_classes=6)
    rng = np.random.default_rng(1)
    ch = synth_window(3, cfg, rng, severity=0.2)
    sc = D["scaler"]
    buf = z["Xs"][0].copy()

    def extract():
        return extract_features(ch, cfg)

    def prep():
        f = extract_features(ch, cfg)
        b = np.vstack([buf[1:], f[None, :]])
        P = physics_features(b, cfg.thermal, cfg.feat_per_channel)
        Z = np.clip(sc.transform(np.concatenate([b, P], axis=-1)), -8, 8).astype(np.float32)[None]
        Zf = np.concatenate([Z[:, -1, :], Z.mean(axis=1), Z.std(axis=1)], axis=1)
        return Z, Zf

    Z, Zf = prep()
    res = dict(feature_extraction=timeit(extract, n=300),
               preprocessing_total=timeit(prep, n=300))
    res["rf200_inference"] = timeit(lambda: rf200.predict(Zf), n=300)
    res["student_gru_inference"] = timeit(lambda: mdl["student_q"](torch.tensor(Z)), n=300)
    res["student_rf50_inference"] = timeit(lambda: mdl["rf_student"].predict_proba(Zf), n=300)
    res["student_ensemble_inference"] = timeit(lambda: student.predict(Z, Zf), n=300)
    res["rf200_end_to_end"] = timeit(lambda: rf200.predict(prep()[1]), n=300)
    res["student_end_to_end"] = timeit(lambda: student.predict(*prep()), n=300)
    res["peak_heap_kB"] = dict(
        rf200=peak_kb(lambda: rf200.predict(Zf)),
        student=peak_kb(lambda: student.predict(Z, Zf)),
        preprocessing=peak_kb(prep))
    Xte = D["Xflat"][D["ite"]]
    c200, n200 = tree_stats(rf200, Xte)
    c50, n50 = tree_stats(mdl["rf_student"], Xte)
    npar = sum(p.numel() for p in mdl["student_q"].parameters())
    res["model"] = dict(rf200_nodes=n200, rf200_size_kB=n200 * 12 / 1e3, rf200_comparisons=c200,
                        rf50_nodes=n50, rf50_size_kB=n50 * 12 / 1e3, rf50_comparisons=c50,
                        student_gru_params=int(npar), student_gru_int8_kB=npar / 1e3,
                        student_gru_mflop=(E.gru_flops(111, 32, 16) + E.linear_flops(32, 32)
                                           + E.linear_flops(32, 6)) / 1e6)
    res["accuracy_split0"] = dict(rf200=float((rf200.predict(Xte) == y[D["ite"]]).mean()),
                                  student=clf["metrics"]["Student, distilled + INT8"]["accuracy"])
    out = os.path.join(HERE, "..", "results", "r1_edge.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
