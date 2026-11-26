#!/usr/bin/env python3
"""
export_artifacts.py -- Produce the generated datasets and trained model weights
referenced by the paper's data availability statement.

    python3 export_artifacts.py [--outdir ../artifacts]

Everything written here is reproducible from the master seed in sim/core.py;
the files are published so that a reader can verify the reported numbers
without re-running training.

Writes:
    artifacts/datasets/classification_seq.npz   episode features, labels, physics features
    artifacts/datasets/pinn_trajectory.npz      thermal trajectory used by the PINN
    artifacts/weights/pinn.pt                   PINN state dict (lambda_p = 10)
    artifacts/weights/teacher_gru.pt            Bi-GRU teacher
    artifacts/weights/student_gru_kd.pt         distilled Uni-GRU student
    artifacts/weights/student_gru_int8.pt       student after INT8 fake-quantisation
    artifacts/weights/vae.pt                    integrity-validation autoencoder
    artifacts/weights/rf_teacher.joblib         teacher random forest
    artifacts/weights/rf_student.joblib         student random forest
    artifacts/weights/scaler.joblib             feature standardiser
    artifacts/MANIFEST.json                     sizes, SHA-256 digests, provenance

Author: Md Tabrez
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.core import SimConfig, GLOBAL_SEED
from sim import experiments as E
from sim.models import (GRUClassifier, train_torch_classifier, distill,
                        fake_quantize_int8, train_vae, train_pinn)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="../artifacts")
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    out = os.path.abspath(args.outdir)
    dd, wd = os.path.join(out, "datasets"), os.path.join(out, "weights")
    os.makedirs(dd, exist_ok=True)
    os.makedirs(wd, exist_ok=True)

    cfg = SimConfig()
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    t0 = time.time()

    # ---------------- PINN trajectory and weights ----------------
    print("[1/4] PINN", flush=True)
    traj = E.make_trajectory(cfg, seed=100)
    np.savez_compressed(os.path.join(dd, "pinn_trajectory.npz"),
                        t_min=traj["t_min"], K=traj["K"],
                        Theta_A=traj["Theta_A"], dTO=traj["dTO"],
                        dH=traj["dH"], Theta_H=traj["Theta_H"],
                        Theta_TO=traj["Theta_TO"])
    r = train_pinn(traj, cfg.thermal, n_obs_hs=6, lam_p=E.LAM_P_STAR,
                   epochs=1800, n_coll=1500, seed=0)
    torch.save({"state_dict": r["model"].state_dict(),
                "lam_p": E.LAM_P_STAR, "rmse_K": r["rmse"], "r2": r["r2"],
                "arch": {"hidden": 64, "layers": 5, "n_freq": 12}},
               os.path.join(wd, "pinn.pt"))
    print(f"      RMSE {r['rmse']:.3f} K  R2 {r['r2']:.4f}", flush=True)

    # ---------------- classification dataset ----------------
    print("[2/4] classification dataset", flush=True)
    cache = os.path.join(dd, "classification_seq.npz")
    D = E.build_clf_data(cfg, n_ep=130, seq_len=16, cache=cache)
    joblib.dump(D["scaler"], os.path.join(wd, "scaler.joblib"))
    print(f"      {D['Xseq'].shape}", flush=True)

    # ---------------- teacher / student / forests ----------------
    print("[3/4] classifiers", flush=True)
    itr, iva, ite = D["itr"], D["iva"], D["ite"]
    Xs, y, d, C = D["Xseq"], D["y"], D["d"], cfg.n_classes

    tgru = GRUClassifier(d, C, hidden=96, layers=1, bidir=True, feat_dim=32)
    tgru, _, _ = train_torch_classifier(tgru, Xs[itr], y[itr], Xs[iva], y[iva],
                                        epochs=args.epochs, seed=0)
    trf = E._fit_rf(D, itr, 500, 20, 0)
    srf = E._fit_rf(D, itr, 50, 10, 0)

    s_kd = GRUClassifier(d, C, hidden=32, layers=1, bidir=False, feat_dim=32)
    s_kd, _, _ = distill(s_kd, tgru, Xs[itr], y[itr], None, Xs[iva], y[iva],
                         alpha=0.9, beta=0.05, gamma=0.1,
                         epochs=args.epochs, seed=0)
    s_q = fake_quantize_int8(s_kd)

    arch_t = {"hidden": 96, "layers": 1, "bidir": True, "feat_dim": 32,
              "d_in": int(d), "n_classes": int(C)}
    arch_s = {"hidden": 32, "layers": 1, "bidir": False, "feat_dim": 32,
              "d_in": int(d), "n_classes": int(C)}
    torch.save({"state_dict": tgru.state_dict(), "arch": arch_t},
               os.path.join(wd, "teacher_gru.pt"))
    torch.save({"state_dict": s_kd.state_dict(), "arch": arch_s},
               os.path.join(wd, "student_gru_kd.pt"))
    torch.save({"state_dict": s_q.state_dict(), "arch": arch_s,
                "quantisation": "symmetric per-tensor INT8, de-quantised in place"},
               os.path.join(wd, "student_gru_int8.pt"))
    joblib.dump(trf, os.path.join(wd, "rf_teacher.joblib"), compress=3)
    joblib.dump(srf, os.path.join(wd, "rf_student.joblib"), compress=3)

    # ---------------- VAE ----------------
    print("[4/4] integrity validator", flush=True)
    norm_tr = itr[y[itr] == 0]
    vae = train_vae(Xs[norm_tr][:, -1, :], epochs=250, seed=0)
    torch.save({"state_dict": vae.state_dict(),
                "arch": {"d_in": int(d), "latent": 8}},
               os.path.join(wd, "vae.pt"))

    # ---------------- manifest ----------------
    files = {}
    for root in (dd, wd):
        for fn in sorted(os.listdir(root)):
            p = os.path.join(root, fn)
            if os.path.isfile(p):
                rel = os.path.relpath(p, out)
                files[rel] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    manifest = {
        "seed": GLOBAL_SEED,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "runtime_s": round(time.time() - t0, 1),
        "note": "All files are deterministic given the seed; re-running this "
                "script reproduces identical digests on the same library versions.",
        "files": files,
    }
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nwrote {len(files)} artifacts to {out} in "
          f"{manifest['runtime_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
