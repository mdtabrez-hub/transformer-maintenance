#!/usr/bin/env python3
"""
run_r1_clf.py -- Revision-1 statistical evaluation of the fault classifiers.

  stage cv     : 2 x 5 repeated stratified cross-validation of every variant,
                 with and without the physics-derived features
  stage size   : training-set size sensitivity (5 folds x 5 fractions)
  stage sev    : cross-severity train/test split
  stage shift  : cross-condition evaluation of the deployed models

    python3 run_r1_clf.py --stage cv

Author: Md Tabrez
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.core import SimConfig, physics_features
from sim import experiments as E
from sim.models import (GRUClassifier, CNNLSTM, train_torch_classifier, distill,
                        Ensemble, fake_quantize_int8)
from sim.r1_clf import build_shift_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts", "datasets", "classification_seq.npz")
RES = os.path.join(HERE, "..", "results")


def load_main(cfg):
    z = np.load(ART)
    cache = os.path.join(RES, "r1_severity.npy")
    if os.path.exists(cache):
        sev = np.load(cache)
    else:
        X, y, sev = build_shift_dataset(cfg, [(c,) for c in range(6)], 130, cfg.seed + 7)
        assert np.abs(X - z["Xs"]).max() == 0.0, "regenerated dataset differs"
        np.save(cache, sev)
    return z["Xs"], z["y"], z["P"], sev


def make_D(X, y, itr, iva, ite):
    sc = StandardScaler(); d = X.shape[-1]
    sc.fit(X[itr].reshape(-1, d))
    Z = np.clip(sc.transform(X.reshape(-1, d)).reshape(X.shape), -8, 8).astype(np.float32)
    pool = lambda A: np.concatenate([A[:, -1, :], A.mean(axis=1), A.std(axis=1)], axis=1)
    return dict(Xseq=Z, Xflat=pool(Z), y=y, itr=itr, iva=iva, ite=ite, d=d, scaler=sc)


def gru_pred(m, X):
    with torch.no_grad():
        return m(torch.tensor(X)).argmax(1).numpy()


def classify_all(cfg, D, seed, epochs=80):
    """Same models and settings as sim.experiments.exp_classification."""
    itr, iva, ite = D["itr"], D["iva"], D["ite"]
    Xs, Xf, y, d, C = D["Xseq"], D["Xflat"], D["y"], D["d"], cfg.n_classes
    P = {}
    rf = E._fit_rf(D, itr, 200, 16, seed); P["RandomForest"] = rf.predict(Xf[ite])
    g = GRUClassifier(d, C, hidden=32, layers=1, bidir=False)
    g, _, _ = train_torch_classifier(g, Xs[itr], y[itr], Xs[iva], y[iva], epochs=epochs, seed=seed)
    P["GRU"] = gru_pred(g, Xs[ite])
    cl = CNNLSTM(d, C)
    cl, _, _ = train_torch_classifier(cl, Xs[itr], y[itr], Xs[iva], y[iva], epochs=epochs, seed=seed)
    P["CNN-LSTM"] = gru_pred(cl, Xs[ite])
    tg = GRUClassifier(d, C, hidden=96, layers=1, bidir=True, feat_dim=32)
    tg, _, _ = train_torch_classifier(tg, Xs[itr], y[itr], Xs[iva], y[iva], epochs=epochs, seed=seed)
    trf = E._fit_rf(D, itr, 500, 20, seed)
    wt = E._tune_weight(trf, tg, D, iva)
    P["Teacher"] = Ensemble(trf, tg, w_rf=wt, n_classes=C).predict(Xs[ite], Xf[ite])
    ss = GRUClassifier(d, C, hidden=32, layers=1, bidir=False, feat_dim=32)
    ss, _, _ = train_torch_classifier(ss, Xs[itr], y[itr], Xs[iva], y[iva], epochs=epochs, seed=seed)
    srf = E._fit_rf(D, itr, 50, 10, seed)
    wns = E._tune_weight(srf, ss, D, iva)
    P["Student, no distillation"] = Ensemble(srf, ss, w_rf=wns, n_classes=C).predict(Xs[ite], Xf[ite])
    best = (-1.0, None, None)
    for al, be in ((0.3, 0.6), (0.5, 0.4), (0.7, 0.2), (0.9, 0.05)):
        cand = GRUClassifier(d, C, hidden=32, layers=1, bidir=False, feat_dim=32)
        cand, _, va = distill(cand, tg, Xs[itr], y[itr], None, Xs[iva], y[iva],
                              alpha=al, beta=be, gamma=0.1, epochs=epochs, seed=seed)
        if va > best[0]:
            best = (va, cand, (al, be))
    _, skd, hp = best
    wkd = E._tune_weight(srf, skd, D, iva)
    pkd = Ensemble(srf, skd, w_rf=wkd, n_classes=C).predict_proba(Xs[ite], Xf[ite])
    P["Student, distilled"] = pkd.argmax(1)
    sq = fake_quantize_int8(skd)
    pq = Ensemble(srf, sq, w_rf=wkd, n_classes=C).predict_proba(Xs[ite], Xf[ite])
    P["Student, distilled + INT8"] = pq.argmax(1)
    info = dict(kd_alpha=hp[0], kd_beta=hp[1], w_teacher=wt, w_student_kd=wkd,
                w_student_scratch=wns,
                int8_max_prob_diff=float(np.abs(pq - pkd).max()),
                int8_agreement=float((pq.argmax(1) == pkd.argmax(1)).mean()))
    return P, info


def student_scratch(cfg, D, idx_tr, seed, epochs=55):
    """GRU-32 + RF-50 ensemble without distillation (original ablation config)."""
    m = GRUClassifier(D["d"], cfg.n_classes, hidden=32, feat_dim=32)
    m, _, _ = train_torch_classifier(m, D["Xseq"][idx_tr], D["y"][idx_tr],
                                     D["Xseq"][D["iva"]], D["y"][D["iva"]], epochs=epochs, seed=seed)
    rf = E._fit_rf(D, idx_tr, 50, 10, seed)
    ens = Ensemble(rf, m, n_classes=cfg.n_classes, w_rf=E._tune_weight(rf, m, D, D["iva"]))
    return ens


def folds(y, cfg, reps=2, k=5):
    for rep in range(reps):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed + rep)
        for f, (trv, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            itr, iva = train_test_split(trv, test_size=0.32, random_state=cfg.seed + 10 * rep + f,
                                        stratify=y[trv])
            yield rep, f, itr, iva, te


def stage_cv(cfg, out):
    Xs, y, P, sev = load_main(cfg)
    Xphys = np.concatenate([Xs, P], axis=-1)
    rows = []
    for rep, f, itr, iva, te in folds(y, cfg):
        t0 = time.time(); seed = 5 * rep + f
        D = make_D(Xphys, y, itr, iva, te)
        preds, info = classify_all(cfg, D, seed)
        Dn = make_D(Xs, y, itr, iva, te)
        rfn = E._fit_rf(Dn, itr, 200, 16, seed)
        preds["RandomForest, no physics"] = rfn.predict(Dn["Xflat"][te])
        preds["Student, no distillation, no physics"] = \
            student_scratch(cfg, Dn, itr, seed, epochs=80).predict(Dn["Xseq"][te], Dn["Xflat"][te])
        rows.append(dict(rep=rep, fold=f, seed=seed, n_train=len(itr), n_val=len(iva),
                         n_test=len(te), test_idx=te.tolist(),
                         preds={k: v.tolist() for k, v in preds.items()},
                         acc={k: float((v == y[te]).mean()) for k, v in preds.items()},
                         info=info, runtime_s=time.time() - t0))
        print(f"rep{rep} fold{f}: " + "  ".join(f"{k[:18]}={v*100:.1f}" for k, v in rows[-1]["acc"].items())
              + f"  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(dict(rows=rows, y=y.tolist(), sev=sev.tolist()), open(out, "w"))


def stage_size(cfg, out):
    Xs, y, P, sev = load_main(cfg)
    X = np.concatenate([Xs, P], axis=-1)
    rows = []
    for rep, f, itr, iva, te in folds(y, cfg, reps=1):
        D = make_D(X, y, itr, iva, te)
        rng = np.random.default_rng(cfg.seed + f)
        for fr in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
            k = max(cfg.n_classes * 3, int(len(itr) * fr))
            # stratified subsample so every class is present
            sub, _ = (train_test_split(itr, train_size=k, random_state=cfg.seed + f, stratify=y[itr])
                      if fr < 1.0 else (itr, None))
            ens = student_scratch(cfg, D, sub, seed=f)
            rf = E._fit_rf(D, sub, 200, 16, f)
            rows.append(dict(fold=f, fraction=fr, n_train=int(len(sub)),
                             acc_student=float((ens.predict(D["Xseq"][te], D["Xflat"][te]) == y[te]).mean()),
                             acc_rf=float((rf.predict(D["Xflat"][te]) == y[te]).mean())))
            print(rows[-1], flush=True)
    json.dump(rows, open(out, "w"))


def stage_sev(cfg, out):
    """Train on the lower-severity half of each fault class, test on the upper half."""
    Xs, y, P, sev = load_main(cfg)
    X = np.concatenate([Xs, P], axis=-1)
    rows = []
    for direction in ("low->high", "high->low"):
        tr_idx, te_idx = [], []
        rng = np.random.default_rng(cfg.seed)
        for c in range(6):
            idx = np.where(y == c)[0]
            if c == 0:
                perm = rng.permutation(idx); h = len(idx) // 2
                tr_idx += perm[:h].tolist(); te_idx += perm[h:].tolist(); continue
            order = idx[np.argsort(sev[idx])]; h = len(order) // 2
            lo, hi = order[:h], order[h:]
            a, b = (lo, hi) if direction == "low->high" else (hi, lo)
            tr_idx += a.tolist(); te_idx += b.tolist()
        tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
        for seed in range(3):
            itr, iva = train_test_split(tr_idx, test_size=0.32, random_state=cfg.seed + seed,
                                        stratify=y[tr_idx])
            D = make_D(X, y, itr, iva, te_idx)
            ens = student_scratch(cfg, D, itr, seed, epochs=80)
            rf = E._fit_rf(D, itr, 200, 16, seed)
            rows.append(dict(direction=direction, seed=seed,
                             sev_train=[float(sev[itr][y[itr] > 0].min()), float(sev[itr][y[itr] > 0].max())],
                             sev_test=[float(sev[te_idx][y[te_idx] > 0].min()), float(sev[te_idx][y[te_idx] > 0].max())],
                             acc_student=float((ens.predict(D["Xseq"][te_idx], D["Xflat"][te_idx]) == y[te_idx]).mean()),
                             acc_rf=float((rf.predict(D["Xflat"][te_idx]) == y[te_idx]).mean())))
            print(rows[-1], flush=True)
    json.dump(rows, open(out, "w"))


SHIFTS = {
    "X0 nominal (fresh seed)": dict(),
    "X1 hot climate, heavy load": dict(Ta_mu=38.0, K_mu=0.85),
    "X2 60 Hz network": dict(f_line0=60.0),
    "X3 low SNR (4 dB)": dict(snr_db=4.0),
    "X4 wider calibration spread": dict(calib_gain_sd=0.20, calib_off_sd=0.12),
    "X5 weaker faults (severity 0.01-0.06)": dict(sev_range=(0.01, 0.06)),
}
MIXED = {"Insulation + PD": (2, 4), "Thermal overload + cooling fault": (1, 3),
         "Tap changer + PD": (5, 4), "Cooling fault + insulation": (3, 2)}


def stage_shift(cfg, out, n_per_class=40):
    """Train on the paper's split, test on newly generated shifted episodes."""
    Xs, y, P, sev = load_main(cfg)
    X = np.concatenate([Xs, P], axis=-1)
    D = E._finalise_splits(cfg, X, y)
    res = dict(shifts={}, mixed={})
    models = []
    for seed in range(3):
        preds, _ = None, None
        Dm = dict(D)
        # train the deployed student, teacher and RF once per seed
        clf, mdl = E.exp_classification(cfg, Dm, seed=seed, epochs=80)
        rf200 = E._fit_rf(Dm, Dm["itr"], 200, 16, seed)
        wkd = clf["ensemble_weights"]["student_kd"]; wt = clf["ensemble_weights"]["teacher"]
        models.append(dict(student=Ensemble(mdl["rf_student"], mdl["student_q"], w_rf=wkd, n_classes=6),
                           teacher=Ensemble(mdl["rf_teacher"], mdl["teacher_gru"], w_rf=wt, n_classes=6),
                           rf=rf200, acc_in=clf["metrics"]["Student, distilled + INT8"]["accuracy"]))
        print(f"seed {seed}: in-distribution student acc {models[-1]['acc_in']:.4f}", flush=True)
    sc = D["scaler"]

    def prep(Xn):
        Pn = physics_features(Xn, cfg.thermal, cfg.feat_per_channel)
        Xa = np.concatenate([Xn, Pn], axis=-1); d = Xa.shape[-1]
        Z = np.clip(sc.transform(Xa.reshape(-1, d)).reshape(Xa.shape), -8, 8).astype(np.float32)
        Zf = np.concatenate([Z[:, -1, :], Z.mean(axis=1), Z.std(axis=1)], axis=1)
        return Z, Zf

    for i, (name, kw) in enumerate(SHIFTS.items()):
        t0 = time.time()
        Xn, yn, _ = build_shift_dataset(cfg, [(c,) for c in range(6)], n_per_class,
                                        seed=cfg.seed + 500 + i, **kw)
        Z, Zf = prep(Xn)
        r = {k: [] for k in ("student", "teacher", "rf")}
        percls = []
        for m in models:
            ps = m["student"].predict(Z, Zf)
            r["student"].append(float((ps == yn).mean()))
            r["teacher"].append(float((m["teacher"].predict(Z, Zf) == yn).mean()))
            r["rf"].append(float((m["rf"].predict(Zf) == yn).mean()))
            percls.append([float((ps[yn == c] == c).mean()) for c in range(6)])
        res["shifts"][name] = dict(acc=r, per_class_student=np.mean(percls, axis=0).tolist())
        print(name, {k: round(np.mean(v) * 100, 1) for k, v in r.items()},
              f"({time.time()-t0:.0f}s)", flush=True)
    for i, (name, fcs) in enumerate(MIXED.items()):
        Xn, yn, _ = build_shift_dataset(cfg, [fcs], n_per_class, seed=cfg.seed + 700 + i)
        Z, Zf = prep(Xn)
        rr = []
        for m in models:
            ps = m["student"].predict(Z, Zf)
            rr.append(dict(either=float(np.isin(ps, fcs).mean()),
                           first=float((ps == fcs[0]).mean()), second=float((ps == fcs[1]).mean()),
                           normal=float((ps == 0).mean()),
                           other=float((~np.isin(ps, fcs + (0,))).mean())))
        res["mixed"][name] = {k: float(np.mean([d[k] for d in rr])) for k in rr[0]}
        print(name, res["mixed"][name], flush=True)
    json.dump(res, open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cv", "size", "sev", "shift"])
    a = ap.parse_args()
    cfg = SimConfig()
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, f"r1_clf_{a.stage}.json")
    dict(cv=stage_cv, size=stage_size, sev=stage_sev, shift=stage_shift)[a.stage](cfg, out)


if __name__ == "__main__":
    main()
