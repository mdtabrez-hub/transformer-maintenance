#!/usr/bin/env python3
"""
run_r1_rul.py -- Revision-1 remaining-useful-life study.

Ground-truth variants (the estimator always assumes the IEC 60076-7 law):
  G0  lognormal endurance, sigma = 0.26, IEC 6 K doubling law (original)
  G1  lognormal sigma = 0.15            G2  lognormal sigma = 0.40
  G3  Weibull endurance (shape 3.0, same median)
  G4  moisture-accelerated ageing: true rate x (1 + 2 t / horizon)
  G5  Arrhenius law of thermally upgraded paper (B = 15 000 K)

Calibration labels for the health-index map g are, by default, the damage
fraction that the estimator itself can compute for a run-to-failure
calibration unit: IEC accumulated ageing from the noisy hot-spot estimate,
divided by its value at the observed end of life. With --true-labels the
original oracle labels (true loss of life / true endurance) are used.

Estimators: Weibull health-index extrapolation; physics-informed loss of life
(proposed); linear health-index extrapolation; IEC ageing budget; random-forest
regression on health-index and ageing features. Split-conformal intervals are
reported for the proposed estimator.

Author: Md Tabrez
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.core import (SimConfig, airport_load_profile, ambient_profile, top_oil_steady,
                      hotspot_rise_steady, relative_ageing_rate, health_index)
from sim.models import WeibullRUL, LossOfLifeRUL

FRACS = (0.35, 0.5, 0.65, 0.8)


def simulate_unit(cfg, rng, variant, unit_id=0):
    """sim.core.simulate_unit_degradation with ground-truth variants."""
    p = cfg.thermal
    dt_min = cfg.dt_hours * 60.0
    n = int(cfg.horizon_days * 24 / cfg.dt_hours)
    base = float(np.clip(rng.normal(0.66, 0.11), 0.35, 0.95))
    K = airport_load_profile(n, dt_min, rng, base=base)
    Ta = ambient_profile(n, dt_min, rng, mean_C=float(rng.normal(29, 3)))
    fouling_rate = float(np.clip(rng.normal(0.06, 0.03), 0.0, 0.20))
    cooling = 1.0 - fouling_rate * np.linspace(0, 1, n) ** 1.5
    to_ss = top_oil_steady(K, p) / cooling
    hs_ss = hotspot_rise_steady(K, p)
    ao = 1.0 - np.exp(-dt_min / p.tau_o); aw = 1.0 - np.exp(-dt_min / p.tau_w)
    dTO = np.empty(n); dH = np.empty(n)
    dTO[0], dH[0] = to_ss[0], hs_ss[0]
    for i in range(1, n):
        dTO[i] = dTO[i - 1] + ao * (to_ss[i] - dTO[i - 1])
        dH[i] = dH[i - 1] + aw * (hs_ss[i] - dH[i - 1])
    Th = Ta + dTO + dH
    V = relative_ageing_rate(Th, p)
    if variant == "G4":
        V = V * (1.0 + 2.0 * np.linspace(0, 1, n))
    elif variant == "G5":
        V = np.exp(15000.0 / 383.0 - 15000.0 / (Th + 273.0))
    lol = np.cumsum(V) * cfg.dt_hours
    med = lol[-1] * 0.55
    if variant == "G1":
        endurance = float(rng.lognormal(np.log(med), 0.15))
    elif variant == "G2":
        endurance = float(rng.lognormal(np.log(med), 0.40))
    elif variant == "G3":
        k = 3.0
        endurance = float(med / np.log(2) ** (1 / k) * rng.weibull(k))
    else:
        endurance = float(rng.lognormal(np.log(med), 0.26))
    d = np.clip(lol / endurance, 0, 3)
    thd = 0.022 + 0.055 * d ** 1.8 + rng.normal(0, 0.004, n)
    vib = 1.0 + 0.85 * d ** 2.1 + rng.normal(0, 0.06, n)
    ae = 0.10 + 0.75 * d ** 2.4 + rng.normal(0, 0.02, n)
    eol_idx = int(np.argmax(lol >= endurance)) if np.any(lol >= endurance) else -1
    return dict(unit_id=unit_id, Theta_H=Th, lol=lol, endurance=endurance, eol_idx=eol_idx,
                thd=thd, vib=vib, ae=ae, n=n)


def rf_features(t_days, hi, L_iec, V_iec, k, step_days, win_days=900.0):
    w = max(8, int(win_days / step_days)); lo = max(0, k + 1 - w)
    tt = t_days[lo:k + 1] - t_days[k]
    slope = np.polyfit(tt, hi[lo:k + 1], 1)[0] if len(tt) > 3 else 0.0
    v90 = V_iec[max(0, k - int(90 / step_days)):k + 1].mean()
    return [t_days[k], hi[k], slope * 365.0, hi[lo:k + 1].mean(), L_iec[k], v90]


def run_variant(cfg, variant, true_labels=False, seed_offset=21):
    rng = np.random.default_rng(cfg.seed + seed_offset)
    rng_lab = np.random.default_rng(cfg.seed + 99)
    step_days = cfg.dt_hours / 24.0
    units = []
    for u in range(cfg.n_units):
        d = simulate_unit(cfg, rng, variant, unit_id=u)
        if d["eol_idx"] <= 60:
            continue
        hi = health_index(d["Theta_H"], d["thd"], d["vib"], d["ae"], cfg.thermal)
        w = max(3, int(7 / step_days))
        d["hi"] = np.convolve(hi, np.ones(w) / w, mode="same")
        d["t_days"] = np.arange(d["n"]) * step_days
        units.append(d)
    n_keep = len(units)
    idx = np.arange(n_keep); rng.shuffle(idx)
    n_cal = max(10, int(0.30 * n_keep))
    cal, tst = idx[:n_cal], idx[n_cal:]
    hi_eol = float(np.median([units[i]["hi"][units[i]["eol_idx"]] for i in cal]))

    # ---- calibration labels ----
    for i in cal:
        u = units[i]; e = u["eol_idx"]
        if true_labels:
            u["d_lab"] = u["lol"][:e + 1] / u["endurance"]
        else:
            Th_est = u["Theta_H"] + rng_lab.normal(0, 1.5, u["n"])
            L = np.cumsum(relative_ageing_rate(Th_est, cfg.thermal)) * cfg.dt_hours
            u["d_lab"] = L[:e + 1] / L[e]
            u["L_iec_eol"] = float(L[e])
    hi_grid, d_grid = LossOfLifeRUL.calibrate([units[i]["hi"][:units[i]["eol_idx"] + 1] for i in cal],
                                              [units[i]["d_lab"] for i in cal])
    wb = WeibullRUL(hi_eol=hi_eol)
    lol_est = LossOfLifeRUL(hi_grid, d_grid, window_days=900.0)

    # ---- data-driven regressor and IEC budget prior, trained on calibration units ----
    Xr, yr, L_end = [], [], []
    for i in cal:
        u = units[i]; e = u["eol_idx"]
        Th_obs = u["Theta_H"] + rng_lab.normal(0, 1.5, u["n"])
        V = relative_ageing_rate(Th_obs, cfg.thermal)
        L = np.cumsum(V) * cfg.dt_hours
        L_end.append(L[e])
        for fr in np.arange(0.2, 0.91, 0.05):
            k = int(e * fr)
            if k < 30:
                continue
            Xr.append(rf_features(u["t_days"], u["hi"], L, V, k, step_days))
            yr.append(u["t_days"][e] - u["t_days"][k])
    rfr = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=1)
    rfr.fit(np.array(Xr), np.array(yr))
    L_end_med = float(np.median(L_end))

    # ---- conformal residuals of the proposed estimator on calibration units ----
    conf = {f: [] for f in FRACS}
    for i in cal:
        u = units[i]; e = u["eol_idx"]
        Th_obs = u["Theta_H"] + rng_lab.normal(0, 1.5, u["n"])
        V = relative_ageing_rate(Th_obs, cfg.thermal)
        for fr in FRACS:
            k = int(e * fr)
            rp = lol_est.predict(u["t_days"], V, u["hi"], k, cfg.dt_hours)
            if rp is None:
                continue
            act = u["t_days"][e] - u["t_days"][k]
            conf[fr].append(abs(rp["rul"] - act) / max(rp["rul"], 1.0))
    qrel = {f: float(np.quantile(v, 0.8)) if len(v) > 4 else np.nan for f, v in conf.items()}

    names = ["Weibull HI", "Physics-informed loss of life", "Linear HI extrapolation",
             "IEC ageing budget", "Random-forest regression"]
    P = {k: [] for k in names}; acts, fracs, cover = [], [], []
    for i in tst:
        d = units[i]; n = d["eol_idx"]; t_days, hi_s = d["t_days"], d["hi"]
        Th_obs = d["Theta_H"] + rng.normal(0, 1.5, d["n"])
        V_obs = relative_ageing_rate(Th_obs, cfg.thermal)
        L_obs = np.cumsum(V_obs) * cfg.dt_hours
        for frac in FRACS:
            k = int(n * frac)
            if k < 30:
                continue
            r = wb.fit_predict(t_days, hi_s, k)
            rp = lol_est.predict(t_days, V_obs, hi_s, k, cfg.dt_hours)
            if r is None or rp is None or not np.isfinite(r["rul"]):
                continue
            # linear HI extrapolation to the calibrated threshold
            wwin = max(8, int(900.0 / step_days)); lo = max(0, k + 1 - wwin)
            tt = t_days[lo:k + 1] - t_days[k]
            sl, ic = np.polyfit(tt, hi_s[lo:k + 1], 1)
            lin = float(np.clip((hi_eol - ic) / max(sl, 1e-7), 0, 4000))
            vbar = V_obs[max(0, k - int(365 / step_days)):k + 1].mean() * 24.0
            budget = float(np.clip((L_end_med - L_obs[k]) / max(vbar, 1e-9), 0, 4000))
            rfp = float(rfr.predict(np.array([rf_features(t_days, hi_s, L_obs, V_obs, k, step_days)]))[0])
            act = t_days[n] - t_days[k]
            for nm, v in zip(names, (r["rul"], rp["rul"], lin, budget, rfp)):
                P[nm].append(v)
            acts.append(act); fracs.append(frac)
            q = qrel[frac]
            cover.append(bool(abs(rp["rul"] - act) <= q * max(rp["rul"], 1.0)))
    acts = np.array(acts); fracs = np.array(fracs)

    def st(pr, sel=None):
        pr = np.asarray(pr); a = acts
        if sel is not None:
            pr, a = pr[sel], a[sel]
        e = pr - a
        return dict(mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                    r2=float(1 - np.sum(e ** 2) / np.sum((a - a.mean()) ** 2)),
                    bias=float(np.mean(e)), n=int(len(a)))
    out = dict(variant=variant, true_labels=true_labels, n_units=n_keep, n_cal=n_cal,
               n_test=int(n_keep - n_cal), n_points=int(len(acts)), hi_eol=hi_eol,
               L_end_med=L_end_med,
               overall={k: st(v) for k, v in P.items()},
               per_stage={str(f): {k: st(v, fracs == f) for k, v in P.items()} for f in FRACS},
               conformal=dict(q_rel=qrel, coverage=float(np.mean(cover)),
                              coverage_stage={str(f): float(np.mean(np.array(cover)[fracs == f]))
                                              for f in FRACS}),
               hi_damage_curve=dict(hi=hi_grid.tolist(), d=d_grid.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/r1_rul.json")
    a = ap.parse_args()
    cfg = SimConfig()
    res = []
    for variant, tl in (("G0", True), ("G0", False), ("G1", False), ("G2", False),
                        ("G3", False), ("G4", False), ("G5", False)):
        t0 = time.time()
        r = run_variant(cfg, variant, true_labels=tl)
        res.append(r)
        print(variant, "oracle" if tl else "IEC labels",
              {k: round(v["mae"], 1) for k, v in r["overall"].items()},
              f"cov={r['conformal']['coverage']:.2f} n={r['n_points']} ({time.time()-t0:.0f}s)",
              flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
