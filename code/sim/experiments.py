"""
experiments.py -- All simulation experiments for the manuscript.

Every number reported in the paper is produced by one of the functions below.
All random number generation is seeded from sim.core.GLOBAL_SEED.

Author: Md Tabrez
"""

import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import StandardScaler

from .core import (SimConfig, ThermalParams, FAULT_CLASSES, CHANNEL_NAMES,
                   airport_load_profile, ambient_profile, simulate_thermal,
                   simulate_multiphase_thermal, phase_imbalance,
                   build_sequence_dataset, physics_features,
                   simulate_unit_degradation, health_index, bandwidth_analysis,
                   relative_ageing_rate, top_oil_steady, hotspot_rise_steady)
from .models import (PINN, LossOfLifeRUL, train_pinn, train_pinn_multiphase, GRUClassifier,
                     CNNLSTM, train_torch_classifier, distill, Ensemble,
                     VAE, train_vae, WeibullRUL, set_seed,
                     count_params_bytes, gru_flops, linear_flops, rf_flops,
                     fake_quantize_int8, project_latency)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

LAM_P_STAR = 10.0   # physics-loss weight selected by the ablation in exp_pinn_lambda


def make_trajectory(cfg, seed, days=6, dt=5.0):
    """Single-phase reference thermal trajectory."""
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60 / dt)
    K = airport_load_profile(n, dt, rng)
    Ta = ambient_profile(n, dt, rng)
    Th, Tto, dTO, dH = simulate_thermal(K, Ta, cfg.thermal, dt)
    return dict(t_min=np.arange(n) * dt, K=K, Theta_A=Ta, dTO=dTO, dH=dH,
                Theta_H=Th, Theta_TO=Tto)


def make_multiphase_trajectory(cfg, seed, days=5, dt=5.0):
    """Multi-phase trajectory with realistic inter-phase load imbalance."""
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60 / dt)
    K = airport_load_profile(n, dt, rng)
    Ta = ambient_profile(n, dt, rng)
    Kph = phase_imbalance(K, cfg.n_phases, rng, imbalance=0.06)
    Th, Tto, dTO, dH = simulate_multiphase_thermal(Kph, Ta, cfg.thermal, dt)
    return dict(t_min=np.arange(n) * dt, K=K, K_phase=Kph, Theta_A=Ta,
                dTO=dTO, dH_phase=dH, Theta_H_phase=Th)


def clf_metrics(y_true, y_pred):
    a = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 average="macro", zero_division=0)
    return dict(accuracy=float(a), precision=float(p), recall=float(r),
                f1=float(f))


# ==========================================================================
# E1-E4  PHYSICS-INFORMED NEURAL NETWORK EXPERIMENTS
# ==========================================================================

def exp_pinn_main(cfg, seeds=(0, 1, 2), epochs=2500, n_coll=1500):
    """Headline PINN result: hot-spot reconstruction from sparse observations."""
    print("  [E1] PINN hot-spot reconstruction", flush=True)
    out = {"pinn": [], "data_only": []}
    curves = None
    for s in seeds:
        traj = make_trajectory(cfg, seed=100 + s)
        r1 = train_pinn(traj, cfg.thermal, n_obs_hs=6, lam_p=LAM_P_STAR,
                        epochs=epochs, n_coll=n_coll, seed=s)
        r0 = train_pinn(traj, cfg.thermal, n_obs_hs=6, lam_p=0.0,
                        epochs=epochs, n_coll=n_coll, seed=s)
        out["pinn"].append(dict(rmse=r1["rmse"], mae=r1["mae"], r2=r1["r2"]))
        out["data_only"].append(dict(rmse=r0["rmse"], mae=r0["mae"], r2=r0["r2"]))
        print(f"    seed {s}: PINN RMSE={r1['rmse']:.3f} K  "
              f"data-only RMSE={r0['rmse']:.3f} K", flush=True)
        if curves is None:
            curves = dict(t_min=traj["t_min"].tolist(),
                          Th_true=r1["Th_true"].tolist(),
                          Th_pinn=r1["Th_pred"].tolist(),
                          Th_data=r0["Th_pred"].tolist(),
                          K=traj["K"].tolist(),
                          Theta_A=traj["Theta_A"].tolist(),
                          obs_idx=r1["i_to"].tolist(),
                          hist_pinn=r1["hist"], hist_data=r0["hist"])
    summary = {}
    for k in out:
        summary[k] = {m: [float(np.mean([d[m] for d in out[k]])),
                          float(np.std([d[m] for d in out[k]]))]
                      for m in ("rmse", "mae", "r2")}
    return dict(runs=out, summary=summary, curves=curves)


def exp_pinn_lambda(cfg, lambdas=(0.0, 0.03, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
                    epochs=2000, n_coll=1500, seed=0):
    """Ablation: sensitivity of reconstruction accuracy to the physics weight."""
    print("  [E2] PINN physics-loss weight ablation", flush=True)
    traj = make_trajectory(cfg, seed=100)
    res = []
    for lp in lambdas:
        r = train_pinn(traj, cfg.thermal, n_obs_hs=6, lam_p=lp,
                       epochs=epochs, n_coll=n_coll, seed=seed)
        res.append(dict(lam_p=lp, rmse=r["rmse"], mae=r["mae"], r2=r["r2"]))
        print(f"    lambda_p={lp:<7g} RMSE={r['rmse']:.3f} K  R2={r['r2']:.4f}",
              flush=True)
    return res


def exp_pinn_data_efficiency(cfg, n_labels=(0, 2, 6, 20, 60, 200),
                             epochs=2000, n_coll=1500, seed=0):
    """Ablation: hot-spot label budget required by each model."""
    print("  [E3] PINN data-efficiency sweep", flush=True)
    traj = make_trajectory(cfg, seed=100)
    res = []
    for nl in n_labels:
        r1 = train_pinn(traj, cfg.thermal, n_obs_hs=nl, lam_p=LAM_P_STAR,
                        epochs=epochs, n_coll=n_coll, seed=seed)
        r0 = train_pinn(traj, cfg.thermal, n_obs_hs=nl, lam_p=0.0,
                        epochs=epochs, n_coll=n_coll, seed=seed)
        res.append(dict(n_labels=nl, pinn_rmse=r1["rmse"], data_rmse=r0["rmse"],
                        pinn_r2=r1["r2"], data_r2=r0["r2"]))
        print(f"    n_hs_labels={nl:<4d} PINN={r1['rmse']:.3f} K  "
              f"data-only={r0['rmse']:.3f} K", flush=True)
    return res


def exp_pinn_multiphase(cfg, epochs=1400, seed=0):
    """Ablation: benefit of the explicit inter-phase thermal coupling term."""
    print("  [E4] Multi-phase PINN coupling ablation", flush=True)
    traj = make_multiphase_trajectory(cfg, seed=200)
    res = []
    for lc in (0.0, 1.0):
        r = train_pinn_multiphase(traj, cfg.thermal, n_phases=cfg.n_phases,
                                  lam_p=LAM_P_STAR, lam_couple=lc, n_obs_hs=8,
                                  epochs=epochs, seed=seed)
        res.append(dict(lam_couple=lc, rmse=r["rmse"],
                        imbalance_rmse=r["imbalance_rmse"]))
        print(f"    lambda_couple={lc}: RMSE={r['rmse']:.3f} K  "
              f"imbalance RMSE={r['imbalance_rmse']:.3f} K", flush=True)
    return res


# ==========================================================================
# E5-E8  FAULT CLASSIFICATION, DISTILLATION AND QUANTISATION
# ==========================================================================

def build_clf_data(cfg, n_ep=100, seq_len=16, snr_db=None, seed=None,
                   use_physics=True, cache=None):
    """Assemble train / validation / test splits for the classification study.

    If `cache` names a .npz file the raw generated features are reused, which
    makes repeated runs of the downstream experiments inexpensive without
    changing any result (the generator is seeded).
    """
    import os as _os
    if cache and _os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        Xs, y, P = z["Xs"], z["y"], z["P"]
    else:
        Xs, y, meta = build_sequence_dataset(cfg, n_episodes_per_class=n_ep,
                                             seq_len=seq_len, snr_db=snr_db,
                                             seed=seed, verbose=False)
        P = physics_features(Xs, cfg.thermal, cfg.feat_per_channel)
        if cache:
            _os.makedirs(_os.path.dirname(cache) or ".", exist_ok=True)
            np.savez_compressed(cache, Xs=Xs, y=y, P=P)
    if use_physics:
        Xs = np.concatenate([Xs, P], axis=-1)
    return _finalise_splits(cfg, Xs, y)


def _finalise_splits(cfg, Xs, y):
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=0.30, random_state=cfg.seed,
                                stratify=y)
    itr, iva = train_test_split(itr, test_size=0.32, random_state=cfg.seed,
                                stratify=y[itr])

    sc = StandardScaler()
    d = Xs.shape[-1]
    sc.fit(Xs[itr].reshape(-1, d))
    Z = sc.transform(Xs.reshape(-1, d)).reshape(Xs.shape).astype(np.float32)
    Z = np.clip(Z, -8, 8)

    # Random Forest operates on a pooled summary of the episode
    def pool(A):
        return np.concatenate([A[:, -1, :], A.mean(axis=1), A.std(axis=1)], axis=1)

    return dict(Xseq=Z, Xflat=pool(Z), y=y, itr=itr, iva=iva, ite=ite,
                d=d, scaler=sc)


def _tune_weight(rf, gru, D, iva, smooth=1):
    """Select the ensemble mixing weight on the validation split.

    The raw validation accuracy is a noisy function of the mixing weight when
    the validation set is small, so the curve is smoothed over neighbouring
    grid points before the maximum is taken. This makes the selected weight far
    more stable without using any test data.
    """
    grid = np.linspace(0.0, 1.0, 21)
    acc = []
    for w in grid:
        e = Ensemble(rf, gru, w_rf=float(w), n_classes=int(D["y"].max()) + 1)
        acc.append(accuracy_score(D["y"][iva],
                                  e.predict(D["Xseq"][iva], D["Xflat"][iva])))
    acc = np.asarray(acc)
    k = 2 * smooth + 1
    sm = np.convolve(acc, np.ones(k) / k, mode="same")
    return float(grid[int(np.argmax(sm))])


def _fit_rf(D, idx_tr, n_est, depth, seed):
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                random_state=seed, n_jobs=1,
                                class_weight="balanced_subsample")
    rf.fit(D["Xflat"][idx_tr], D["y"][idx_tr])
    return rf


def exp_classification(cfg, D, seed=0, epochs=60):
    """Train and evaluate every classifier variant on the common splits."""
    print("  [E5] Fault classification models", flush=True)
    itr, iva, ite = D["itr"], D["iva"], D["ite"]
    Xs, Xf, y = D["Xseq"], D["Xflat"], D["y"]
    d, C = D["d"], cfg.n_classes
    res, models, hists = {}, {}, {}

    # ---------- Random Forest baseline ----------
    t0 = time.time()
    rf_base = _fit_rf(D, itr, 200, 16, seed)
    res["RandomForest"] = clf_metrics(y[ite], rf_base.predict(Xf[ite]))
    res["RandomForest"]["train_s"] = time.time() - t0

    # ---------- GRU baseline ----------
    t0 = time.time()
    gru = GRUClassifier(d, C, hidden=32, layers=1, bidir=False)
    gru, h_gru, _ = train_torch_classifier(gru, Xs[itr], y[itr], Xs[iva], y[iva],
                                           epochs=epochs, seed=seed)
    with torch.no_grad():
        pg = gru(torch.tensor(Xs[ite])).argmax(1).numpy()
    res["GRU"] = clf_metrics(y[ite], pg)
    res["GRU"]["train_s"] = time.time() - t0
    hists["GRU"] = h_gru

    # ---------- CNN-LSTM baseline ----------
    t0 = time.time()
    cl = CNNLSTM(d, C)
    cl, h_cl, _ = train_torch_classifier(cl, Xs[itr], y[itr], Xs[iva], y[iva],
                                         epochs=epochs, seed=seed)
    with torch.no_grad():
        pc = cl(torch.tensor(Xs[ite])).argmax(1).numpy()
    res["CNN-LSTM"] = clf_metrics(y[ite], pc)
    res["CNN-LSTM"]["train_s"] = time.time() - t0
    hists["CNN-LSTM"] = h_cl

    # ---------- Teacher ensemble : Bi-GRU (2 layers, 128) + RF(500, d20) ----------
    t0 = time.time()
    tgru = GRUClassifier(d, C, hidden=96, layers=1, bidir=True, feat_dim=32)
    tgru, h_t, _ = train_torch_classifier(tgru, Xs[itr], y[itr], Xs[iva], y[iva],
                                          epochs=epochs, seed=seed)
    trf = _fit_rf(D, itr, 500, 20, seed)
    w_t = _tune_weight(trf, tgru, D, iva)
    teacher = Ensemble(trf, tgru, w_rf=w_t, n_classes=C)
    res["Teacher (Bi-GRU + RF)"] = clf_metrics(y[ite], teacher.predict(Xs[ite], Xf[ite]))
    res["Teacher (Bi-GRU + RF)"]["train_s"] = time.time() - t0
    hists["Teacher"] = h_t

    # ---------- Student trained from scratch (no distillation) ----------
    t0 = time.time()
    s_scratch = GRUClassifier(d, C, hidden=32, layers=1, bidir=False, feat_dim=32)
    s_scratch, _, _ = train_torch_classifier(s_scratch, Xs[itr], y[itr],
                                             Xs[iva], y[iva], epochs=epochs, seed=seed)
    srf = _fit_rf(D, itr, 50, 10, seed)
    w_ns = _tune_weight(srf, s_scratch, D, iva)
    st_ns = Ensemble(srf, s_scratch, w_rf=w_ns, n_classes=C)
    res["Student, no distillation"] = clf_metrics(y[ite], st_ns.predict(Xs[ite], Xf[ite]))
    res["Student, no distillation"]["train_s"] = time.time() - t0

    # ---------- Student with knowledge distillation ----------
    # the balance between hard-label and teacher-soft-label supervision is
    # selected on the validation split rather than fixed a priori
    t0 = time.time()
    best = (-1.0, None, None, None)
    for (al, be) in ((0.3, 0.6), (0.5, 0.4), (0.7, 0.2), (0.9, 0.05)):
        cand = GRUClassifier(d, C, hidden=32, layers=1, bidir=False, feat_dim=32)
        cand, h_c, va = distill(cand, tgru, Xs[itr], y[itr], None,
                                Xs[iva], y[iva], alpha=al, beta=be, gamma=0.1,
                                epochs=epochs, seed=seed)
        if va > best[0]:
            best = (va, cand, h_c, (al, be))
    _, s_kd, h_kd, kd_hp = best
    print(f"    distillation weights selected on validation: "
          f"alpha={kd_hp[0]}, beta={kd_hp[1]}", flush=True)
    w_kd = _tune_weight(srf, s_kd, D, iva)
    student = Ensemble(srf, s_kd, w_rf=w_kd, n_classes=C)
    p_st = student.predict_proba(Xs[ite], Xf[ite])
    res["Student, distilled (proposed)"] = clf_metrics(y[ite], p_st.argmax(1))
    res["Student, distilled (proposed)"]["train_s"] = time.time() - t0
    hists["Student-KD"] = h_kd

    # ---------- Student, INT8 post-training quantisation ----------
    s_q = fake_quantize_int8(s_kd)
    student_q = Ensemble(srf, s_q, w_rf=w_kd, n_classes=C)
    p_q = student_q.predict_proba(Xs[ite], Xf[ite])
    res["Student, distilled + INT8"] = clf_metrics(y[ite], p_q.argmax(1))

    cm = confusion_matrix(y[ite], p_q.argmax(1))

    # ---------- one-vs-rest ROC for the deployed (INT8) model ----------
    roc = {}
    for c in range(C):
        fpr, tpr, _ = roc_curve((y[ite] == c).astype(int), p_q[:, c])
        roc[FAULT_CLASSES[c]] = dict(fpr=fpr.tolist(), tpr=tpr.tolist(),
                                     auc=float(auc(fpr, tpr)))

    # ---------- standalone component accuracies, for the ablation table ----
    def _gru_acc(m):
        with torch.no_grad():
            return float((m(torch.tensor(Xs[ite])).argmax(1).numpy() == y[ite]).mean())
    comps = dict(rf_teacher_only=float(accuracy_score(y[ite], trf.predict(Xf[ite]))),
                 rf_student_only=float(accuracy_score(y[ite], srf.predict(Xf[ite]))),
                 teacher_gru_only=_gru_acc(tgru),
                 student_gru_scratch_only=_gru_acc(s_scratch),
                 student_gru_kd_only=_gru_acc(s_kd),
                 student_gru_kd_int8_only=_gru_acc(s_q))

    models = dict(teacher_gru=tgru, student_gru=s_kd, student_q=s_q,
                  rf_student=srf, rf_teacher=trf, gru_base=gru, cnnlstm=cl)
    return dict(metrics=res, confusion=cm.tolist(), roc=roc, hists=hists,
                components=comps,
                distill_hparams=dict(alpha=kd_hp[0], beta=kd_hp[1], gamma=0.1, T=4.0),
                ensemble_weights=dict(teacher=float(w_t), student_kd=float(w_kd),
                                      student_scratch=float(w_ns)),
                proba=p_q.tolist()), models


def exp_physics_feature_ablation(cfg, seed=0, epochs=50, n_ep=100):
    """Ablation: contribution of the physics-derived features to accuracy."""
    print("  [E6] Physics-feature ablation", flush=True)
    out = {}
    for use_phys in (False, True):
        D = build_clf_data(cfg, n_ep=n_ep, use_physics=use_phys, seed=cfg.seed + 5)
        itr, iva, ite = D["itr"], D["iva"], D["ite"]
        m = GRUClassifier(D["d"], cfg.n_classes, hidden=32, feat_dim=32)
        m, _, _ = train_torch_classifier(m, D["Xseq"][itr], D["y"][itr],
                                         D["Xseq"][iva], D["y"][iva],
                                         epochs=epochs, seed=seed)
        rf = _fit_rf(D, itr, 50, 10, seed)
        ens = Ensemble(rf, m, n_classes=cfg.n_classes, w_rf=_tune_weight(rf, m, D, iva))
        k = "with physics features" if use_phys else "sensor features only"
        out[k] = clf_metrics(D["y"][ite], ens.predict(D["Xseq"][ite], D["Xflat"][ite]))
        print(f"    {k}: acc={out[k]['accuracy']:.4f}", flush=True)
    return out


def exp_train_fraction(cfg, D, fractions=(0.05, 0.1, 0.2, 0.4, 0.7, 1.0),
                       seed=0, epochs=50):
    """Ablation: accuracy versus available labelled training fraction."""
    print("  [E7] Training-data fraction ablation", flush=True)
    itr, iva, ite = D["itr"], D["iva"], D["ite"]
    rng = np.random.default_rng(cfg.seed)
    res = []
    for fr in fractions:
        k = max(cfg.n_classes * 3, int(len(itr) * fr))
        sub = rng.choice(itr, size=k, replace=False)
        m = GRUClassifier(D["d"], cfg.n_classes, hidden=32, feat_dim=32)
        m, _, _ = train_torch_classifier(m, D["Xseq"][sub], D["y"][sub],
                                         D["Xseq"][iva], D["y"][iva],
                                         epochs=epochs, seed=seed)
        rf = _fit_rf(D, sub, 50, 10, seed)
        ens = Ensemble(rf, m, n_classes=cfg.n_classes, w_rf=_tune_weight(rf, m, D, iva))
        mt = clf_metrics(D["y"][ite], ens.predict(D["Xseq"][ite], D["Xflat"][ite]))
        res.append(dict(fraction=fr, n_train=int(k), **mt))
        print(f"    fraction={fr:<5g} n={k:<5d} acc={mt['accuracy']:.4f}", flush=True)
    return res


def exp_noise_sensitivity(cfg, snrs=(20, 14, 8, 4, 0), seed=0, epochs=45,
                          n_ep=70, cache_dir="results/noise"):
    """Sensitivity of classification accuracy to sensor signal-to-noise ratio."""
    print("  [E8] Sensor-noise sensitivity", flush=True)
    import os as _os
    _os.makedirs(cache_dir, exist_ok=True)
    res = []
    for snr in snrs:
        D = build_clf_data(cfg, n_ep=n_ep, snr_db=snr, seed=cfg.seed + 11,
                           cache=f"{cache_dir}/snr{snr}.npz")
        itr, iva, ite = D["itr"], D["iva"], D["ite"]
        m = GRUClassifier(D["d"], cfg.n_classes, hidden=32, feat_dim=32)
        m, _, _ = train_torch_classifier(m, D["Xseq"][itr], D["y"][itr],
                                         D["Xseq"][iva], D["y"][iva],
                                         epochs=epochs, seed=seed)
        rf = _fit_rf(D, itr, 50, 10, seed)
        ens = Ensemble(rf, m, n_classes=cfg.n_classes, w_rf=_tune_weight(rf, m, D, iva))
        mt = clf_metrics(D["y"][ite], ens.predict(D["Xseq"][ite], D["Xflat"][ite]))
        res.append(dict(snr_db=snr, **mt))
        print(f"    SNR={snr:>3d} dB  acc={mt['accuracy']:.4f}", flush=True)
    return res


# ==========================================================================
# E9  CYBER-PHYSICAL SECURITY (VAE INTEGRITY VALIDATION)
# ==========================================================================

def exp_security(cfg, D, models, seed=0):
    """Adversarial and integrity-attack detection using the VAE validator."""
    print("  [E9] Cyber-physical attack detection", flush=True)
    rng = np.random.default_rng(cfg.seed + 3)
    Xs, y = D["Xseq"], D["y"]
    itr, ite = D["itr"], D["ite"]

    # VAE is trained on last-window feature vectors of normal-class episodes only
    norm_tr = itr[y[itr] == 0]
    Vtr = Xs[norm_tr][:, -1, :]
    vae = train_vae(Vtr, epochs=250, seed=seed)

    # calibration on held-out normal data
    norm_te = ite[y[ite] == 0]
    Vte = torch.tensor(Xs[norm_te][:, -1, :], dtype=torch.float32)
    s_norm = vae.score(Vte)
    mu, sd = float(s_norm.mean()), float(s_norm.std())
    thr = mu + 3.0 * sd
    fpr_clean = float((s_norm > thr).mean())

    base = Xs[ite][:, -1, :].copy()
    student = models["student_gru"]

    def attack_scores(Xatt):
        return vae.score(torch.tensor(Xatt, dtype=torch.float32))

    attacks = {}

    # (a) temperature-channel offset spoofing: channel 1 features shifted
    Xa = base.copy()
    ch_slice = slice(1 * cfg.feat_per_channel, 2 * cfg.feat_per_channel)
    Xa[:, ch_slice] += rng.normal(2.5, 0.4, (len(Xa), cfg.feat_per_channel))
    attacks["Temperature offset spoofing"] = Xa

    # (b) vibration spoofing: synthetic bearing signature injected
    Xa = base.copy()
    ch_slice = slice(0, cfg.feat_per_channel)
    Xa[:, ch_slice] *= rng.uniform(1.6, 2.4, (len(Xa), cfg.feat_per_channel))
    attacks["Vibration spoofing"] = Xa

    # (c) replay: feature vectors replayed from an unrelated operating point
    Xa = base.copy()
    src = Xs[itr][:, 0, :]
    Xa = src[rng.integers(0, len(src), len(base))].copy()
    attacks["Feature-vector replay"] = Xa

    # (d) FGSM adversarial perturbation against the deployed classifier
    Xseq_te = torch.tensor(Xs[ite], dtype=torch.float32, requires_grad=True)
    yt = torch.tensor(y[ite], dtype=torch.long)
    z = student(Xseq_te)
    loss = F.cross_entropy(z, yt)
    loss.backward()
    eps = 0.05
    Xadv = (Xseq_te + eps * Xseq_te.grad.sign()).detach().numpy()
    attacks["FGSM adversarial (eps=0.05)"] = Xadv[:, -1, :]

    # (e) combined multi-sensor spoofing
    Xa = base.copy()
    Xa[:, 0:cfg.feat_per_channel] *= 1.7
    Xa[:, cfg.feat_per_channel:2 * cfg.feat_per_channel] += 2.0
    attacks["Combined multi-sensor"] = Xa

    rows, roc_data = [], {}
    for name, Xa in attacks.items():
        sa = attack_scores(Xa)
        det = float((sa > thr).mean())
        lab = np.concatenate([np.zeros(len(s_norm)), np.ones(len(sa))])
        sc = np.concatenate([s_norm, sa])
        fpr, tpr, _ = roc_curve(lab, sc)
        rows.append(dict(attack=name, detection_rate=det,
                         false_positive_rate=fpr_clean, auc=float(auc(fpr, tpr))))
        roc_data[name] = dict(fpr=fpr.tolist(), tpr=tpr.tolist(),
                              auc=float(auc(fpr, tpr)))
        print(f"    {name:<32s} DR={det:.4f} AUC={auc(fpr, tpr):.4f}", flush=True)

    # impact of FGSM on classification accuracy, with and without the validator
    with torch.no_grad():
        acc_clean = float((student(torch.tensor(Xs[ite])).argmax(1).numpy() == y[ite]).mean())
        acc_adv = float((student(torch.tensor(Xadv)).argmax(1).numpy() == y[ite]).mean())

    # threshold sensitivity sweep
    sweep = []
    for k in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        t = mu + k * sd
        fp = float((s_norm > t).mean())
        drs = [float((attack_scores(Xa) > t).mean()) for Xa in attacks.values()]
        sweep.append(dict(k_sigma=k, fpr=fp, mean_detection=float(np.mean(drs))))

    return dict(threshold=thr, mu=mu, sigma=sd, rows=rows, roc=roc_data,
                fpr_clean=fpr_clean, acc_clean=acc_clean, acc_adv=acc_adv,
                threshold_sweep=sweep,
                anomaly_flag_rate_normal=fpr_clean)


# ==========================================================================
# E10  REMAINING USEFUL LIFE
# ==========================================================================

def exp_rul(cfg, seed=0):
    """Fleet-scale RUL estimation from health-index trajectories.

    Ground-truth end of life is the physical criterion: cumulative insulation
    loss of life (IEC 60076-7 relative ageing rate integrated over the hot-spot
    trajectory) reaching the unit's endurance. The health-index threshold used
    by the Weibull extrapolation is CALIBRATED ON A DISJOINT SET OF UNITS so
    that no test unit informs the decision rule applied to it.
    """
    print("  [E10] RUL estimation", flush=True)
    rng = np.random.default_rng(cfg.seed + 21)
    step_days = cfg.dt_hours / 24.0

    units = []
    for u in range(cfg.n_units):
        d = simulate_unit_degradation(cfg, rng, unit_id=u)
        if d["eol_idx"] <= 60:
            continue
        hi = health_index(d["Theta_H"], d["thd"], d["vib"], d["ae"], cfg.thermal)
        w = max(3, int(7 / step_days))
        d["hi"] = np.convolve(hi, np.ones(w) / w, mode="same")
        d["t_days"] = np.arange(d["n"]) * step_days
        units.append(d)
    n_keep = len(units)

    # --- calibration / evaluation split over units ---
    idx = np.arange(n_keep)
    rng.shuffle(idx)
    n_cal = max(10, int(0.30 * n_keep))
    cal, tst = idx[:n_cal], idx[n_cal:]
    hi_eol = float(np.median([units[i]["hi"][units[i]["eol_idx"]] for i in cal]))
    print(f"    {n_keep} units reached end of life; HI_eol calibrated to "
          f"{hi_eol:.3f} on {n_cal} held-out units", flush=True)

    # calibrate the monotone health-index versus damage-fraction relation
    hi_cal = [units[i]["hi"][:units[i]["eol_idx"] + 1] for i in cal]
    d_cal = [units[i]["lol"][:units[i]["eol_idx"] + 1] / units[i]["endurance"]
             for i in cal]
    hi_grid, d_grid = LossOfLifeRUL.calibrate(hi_cal, d_cal)
    print(f"    HI-damage relation calibrated over {len(hi_grid)} bins",
          flush=True)

    wb = WeibullRUL(hi_eol=hi_eol)
    lol_est = LossOfLifeRUL(hi_grid, d_grid, window_days=900.0)
    preds, acts, frac_life, ci_lo, ci_hi, uid = [], [], [], [], [], []
    preds_phys = []
    traj_store = []

    for i in tst:
        d = units[i]
        n = d["eol_idx"]
        t_days, hi_s = d["t_days"], d["hi"]
        # ageing rate reconstructed from the estimated hot-spot temperature,
        # with the measurement uncertainty of the thermal estimate included
        Th_obs = d["Theta_H"] + rng.normal(0, 1.5, d["n"])
        V_obs = relative_ageing_rate(Th_obs, cfg.thermal)
        for frac in (0.35, 0.5, 0.65, 0.8):
            k = int(n * frac)
            if k < 30:
                continue
            r = wb.fit_predict(t_days, hi_s, k)
            rp = lol_est.predict(t_days, V_obs, hi_s, k, cfg.dt_hours)
            if r is None or rp is None or not np.isfinite(r["rul"]):
                continue
            preds.append(r["rul"]); preds_phys.append(rp["rul"])
            acts.append(t_days[n] - t_days[k])
            frac_life.append(frac); uid.append(d["unit_id"])
            ci = WeibullRUL.bootstrap_ci(t_days, hi_s, k, hi_eol=hi_eol,
                                         n_boot=40, seed=d["unit_id"])
            ci_lo.append(ci[0] if ci else np.nan)
            ci_hi.append(ci[1] if ci else np.nan)
        if len(traj_store) < 5:
            traj_store.append(dict(unit_id=int(d["unit_id"]),
                                   t_days=t_days[:n + 1].tolist(),
                                   hi=hi_s[:n + 1].tolist(),
                                   eol_day=float(t_days[n])))

    preds = np.asarray(preds); acts = np.asarray(acts)
    preds_phys = np.asarray(preds_phys)
    lo = np.asarray(ci_lo); hi_ = np.asarray(ci_hi)
    frac_life = np.asarray(frac_life)
    m = np.isfinite(preds) & np.isfinite(acts) & np.isfinite(preds_phys)
    preds, acts, lo, hi_ = preds[m], acts[m], lo[m], hi_[m]
    preds_phys, frac_life = preds_phys[m], frac_life[m]

    def _stats(pr):
        e = pr - acts
        return dict(mae=float(np.mean(np.abs(e))),
                    rmse=float(np.sqrt(np.mean(e ** 2))),
                    r2=float(1 - np.sum(e ** 2) / np.sum((acts - acts.mean()) ** 2)),
                    mape=float(np.mean(np.abs(e) / np.maximum(acts, 1.0)) * 100),
                    bias=float(np.mean(e)))
    st_wb = _stats(preds)
    st_ph = _stats(preds_phys)
    print(f"    Weibull-HI      MAE={st_wb['mae']:.1f} d  R2={st_wb['r2']:.3f}",
          flush=True)
    print(f"    loss-of-life    MAE={st_ph['mae']:.1f} d  R2={st_ph['r2']:.3f}",
          flush=True)

    err = preds_phys - acts
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = float(1 - np.sum(err ** 2) / np.sum((acts - acts.mean()) ** 2))
    mape = float(np.mean(np.abs(err) / np.maximum(acts, 1.0)) * 100)
    ok = np.isfinite(lo) & np.isfinite(hi_)
    cov = float(np.mean((lo[ok] <= acts[ok]) & (acts[ok] <= hi_[ok]))) if ok.any() else float("nan")

    per_frac = []
    for f in (0.35, 0.5, 0.65, 0.8):
        sel = frac_life == f
        if sel.sum() < 3:
            continue
        e = err[sel]; ew = (preds - acts)[sel]
        per_frac.append(dict(frac=f, n=int(sel.sum()),
                             mae=float(np.mean(np.abs(e))),
                             rmse=float(np.sqrt(np.mean(e ** 2))),
                             mae_weibull=float(np.mean(np.abs(ew)))))

    print(f"    n={len(preds)}  MAE={mae:.1f} d  RMSE={rmse:.1f} d  "
          f"R2={r2:.4f}  MAPE={mape:.1f}%  CI cov={cov:.3f}", flush=True)
    return dict(pred=preds_phys.tolist(), pred_weibull=preds.tolist(),
                actual=acts.tolist(),
                stats_weibull=st_wb, stats_physics=st_ph,
                frac_life=frac_life.tolist(), mae=mae, rmse=rmse, r2=r2,
                mape=mape, ci_coverage=cov, hi_eol=hi_eol,
                hi_damage_curve=dict(hi=hi_grid.tolist(), d=d_grid.tolist()),
                n_units_total=n_keep, n_units_cal=n_cal,
                n_units_test=int(n_keep - n_cal), per_fraction=per_frac,
                trajectories=traj_store, ci_lo=lo.tolist(), ci_hi=hi_.tolist())


# ==========================================================================
# E11-E13  EDGE COST MODEL, BANDWIDTH, SENSOR PLACEMENT
# ==========================================================================

def exp_edge_cost(cfg, D, models, seq_len=16, n_rep=200):
    """Computational cost of each model variant, measured and projected.

    Host latency is measured directly. Latency on the two embedded targets is
    an ANALYTICAL PROJECTION from the FLOP count and a stated sustained
    throughput; it is not a hardware measurement.
    """
    print("  [E11] Edge computational cost model", flush=True)
    d = D["d"]
    C = cfg.n_classes
    rows = []

    def _rf_nodes(rf):
        """Actual number of nodes in a fitted forest.

        A depth-limited tree is far from complete, so the 2^depth bound
        overestimates the stored size by orders of magnitude.
        """
        return int(sum(t.tree_.node_count for t in rf.estimators_))

    specs = [
        ("Teacher (Bi-GRU 1x96 + RF-500)", models["teacher_gru"],
         gru_flops(d, 96, seq_len, bidir=True) + linear_flops(192, 32)
         + linear_flops(32, C) + rf_flops(500, 20), models["rf_teacher"], 20),
        ("Student (Uni-GRU 1x32 + RF-50)", models["student_gru"],
         gru_flops(d, 32, seq_len) + linear_flops(32, 32) + linear_flops(32, C)
         + rf_flops(50, 10), models["rf_student"], 10),
    ]

    for name, mdl, flops, rf, depth in specs:
        npar, bytes_f32 = count_params_bytes(mdl, 4)
        _, bytes_i8 = count_params_bytes(mdl, 1)
        rf_nodes = _rf_nodes(rf)
        # per node: feature index (2 B), threshold (4 B), two child indices (4 B),
        # plus class-distribution storage amortised over leaves
        rf_bytes = rf_nodes * 12
        x = torch.randn(1, seq_len, d)
        with torch.no_grad():
            for _ in range(20):
                mdl(x)
            t0 = time.perf_counter()
            for _ in range(n_rep):
                mdl(x)
            host_ms = (time.perf_counter() - t0) / n_rep * 1e3
        rows.append(dict(
            model=name, params=int(npar), rf_nodes=rf_nodes, mflops=flops / 1e6,
            size_fp32_kB=bytes_f32 / 1e3, size_int8_kB=bytes_i8 / 1e3,
            rf_size_kB=rf_bytes / 1e3,
            total_int8_kB=(bytes_i8 + rf_bytes) / 1e3,
            host_latency_ms=host_ms))
        print(f"    {name:<34s} {npar:>8d} par  {flops/1e6:6.3f} MFLOP  "
              f"host {host_ms:6.3f} ms", flush=True)

    # analytical projection to embedded targets
    targets = [
        ("ARM Cortex-A72 @1.5 GHz (quad-core, 1 thread)", 900.0),
        ("ARM Cortex-M4F @240 MHz (single-core)", 34.0),
    ]
    proj = []
    for tname, mflops_thr in targets:
        for r in rows:
            proj.append(dict(target=tname, model=r["model"],
                             throughput_MFLOPS=mflops_thr,
                             projected_latency_ms=project_latency(
                                 r["mflops"] * 1e6, mflops_thr) * 1e3))
    return dict(rows=rows, projections=proj)


def exp_bandwidth(cfg, anomaly_rate):
    """Upstream bandwidth accounting using the measured anomaly-flag rate."""
    print("  [E12] Bandwidth analysis", flush=True)
    base = bandwidth_analysis(cfg, anomaly_rate)
    sweep = [bandwidth_analysis(cfg, a)
             for a in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50)]
    print(f"    raw {base['raw_daily_MB']:.1f} MB/day -> "
          f"edge {base['edge_daily_MB']:.1f} MB/day "
          f"({base['reduction_pct']:.2f}% reduction)", flush=True)
    return dict(base=base, sweep=sweep)


def exp_sensor_placement(cfg, n_assets=47, n_edges=62, radius=1, seed=0):
    """Minimum dominating set formulation of the sensor placement problem."""
    print("  [E13] Graph-based sensor placement", flush=True)
    rng = np.random.default_rng(cfg.seed + 31)
    if hasattr(nx, "random_labeled_tree"):
        G = nx.random_labeled_tree(n_assets, seed=int(cfg.seed % 2 ** 31))
    elif hasattr(nx, "random_labelled_tree"):
        G = nx.random_labelled_tree(n_assets, seed=int(cfg.seed % 2 ** 31))
    else:
        G = nx.random_tree(n_assets, seed=int(cfg.seed % 2 ** 31))
    extra = n_edges - G.number_of_edges()
    nodes = list(G.nodes())
    tries = 0
    while G.number_of_edges() < n_edges and tries < 10000:
        u, v = rng.choice(nodes, 2, replace=False)
        if u != v and not G.has_edge(u, v):
            G.add_edge(int(u), int(v))
        tries += 1

    # greedy minimum dominating set with the classical ln(n)+1 guarantee
    undominated = set(G.nodes())
    S = []
    while undominated:
        best, cov = None, -1
        for v in G.nodes():
            if v in S:
                continue
            c = len({v} | set(G.neighbors(v))) & undominated if False else \
                len(({v} | set(G.neighbors(v))) & undominated)
            if c > cov:
                best, cov = v, c
        S.append(best)
        undominated -= ({best} | set(G.neighbors(best)))

    deg = [G.degree(v) for v in S]
    coverage = {}
    for v in G.nodes():
        coverage[v] = sum(1 for s in S if s == v or G.has_edge(s, v))
    res = dict(n_assets=n_assets, n_edges=G.number_of_edges(),
               n_sensors=len(S), sensors=sorted(int(s) for s in S),
               coverage_pct=100.0 * np.mean([c > 0 for c in coverage.values()]),
               mean_degree_of_sensors=float(np.mean(deg)),
               redundancy=float(np.mean(list(coverage.values()))),
               edges=[[int(a), int(b)] for a, b in G.edges()])
    print(f"    {len(S)} sensors dominate {n_assets} assets "
          f"({res['coverage_pct']:.1f}% coverage, redundancy "
          f"{res['redundancy']:.2f})", flush=True)
    return res
