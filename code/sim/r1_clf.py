"""
r1_clf.py -- Revision-1 classification experiments.

  * shifted-condition generator: operating point, climate, line frequency,
    SNR, calibration spread, severity range and mixed (concurrent) faults
  * repeated stratified cross-validation of every classifier variant
  * paired statistics: corrected resampled t-test (Nadeau and Bengio, 2003)
    and exact McNemar test on out-of-fold predictions

The nominal generator is sim.core.synth_window; synth_window_shift reproduces
it exactly when no override is given (checked in run_r1_clf.py).

Author: Md Tabrez
"""

import numpy as np
from scipy import signal as sps
from scipy import stats

from .core import (SimConfig, top_oil_steady, hotspot_rise_steady, _awgn_ac,
                   TEMP_NOISE_K, TEMP_OFFSET_K, extract_features)


def synth_window_shift(fault_classes, cfg, rng, severity, snr_db=None, calib=None,
                       K_mu=0.68, K_sd=0.19, Ta_mu=29.0, Ta_sd=7.0, f_line0=50.0):
    """sim.core.synth_window with condition overrides and concurrent faults.

    `fault_classes` is a tuple of class indices; (0,) is normal operation.
    """
    N = cfg.window; fs = cfg.fs
    t = np.arange(N) / fs
    snr_db = cfg.snr_db if snr_db is None else snr_db
    sev = severity
    p = cfg.thermal

    f_line = f_line0 + rng.normal(0, 0.12)
    K = float(np.clip(rng.normal(K_mu, K_sd), 0.15, 1.20))
    Ta = float(rng.normal(Ta_mu, Ta_sd))
    cooling = float(np.clip(rng.normal(1.0, 0.035), 0.85, 1.05))

    vib = (0.55 * np.sin(2 * np.pi * 2 * f_line * t)
           + 0.22 * np.sin(2 * np.pi * 4 * f_line * t + 0.7)
           + 0.09 * np.sin(2 * np.pi * 6 * f_line * t + 1.9))
    vib *= (0.6 + 0.8 * K ** 2)
    cur = np.sin(2 * np.pi * f_line * t)
    thd_bg = rng.uniform(0.015, 0.035)
    for h, amp in [(3, 0.55), (5, 0.30), (7, 0.15)]:
        cur += thd_bg * amp * np.sin(2 * np.pi * h * f_line * t + rng.uniform(0, 6.28))
    cur *= K
    vol = np.sin(2 * np.pi * f_line * t + 0.05) * (1.0 - 0.012 * K)
    ae = rng.normal(0, 0.05, N)
    b, a = sps.butter(4, [0.30, 0.92], btype="band")
    ae = sps.filtfilt(b, a, ae)
    pd = np.abs(rng.normal(0, 0.02, N))

    for fc in fault_classes:
        if fc == 1:
            K = float(np.clip(rng.normal(1.02, 0.07) + 0.30 * sev, 0.85, 1.45))
            Ta = float(rng.normal(Ta_mu + 4.0, 6.0))
            cur = np.sin(2 * np.pi * f_line * t) * K
            vib *= (0.6 + 0.8 * K ** 2) / (0.6 + 0.8 * 0.68 ** 2)
            ae *= 1.0 + 0.5 * sev
        elif fc == 2:
            n_pulse = int(6 + 26 * sev)
            idx = rng.integers(0, N, n_pulse)
            pd[idx] += rng.uniform(0.25, 1.1, n_pulse) * sev
            burst = np.zeros(N)
            for i in rng.integers(0, N - 80, max(2, n_pulse // 3)):
                env = np.exp(-np.arange(80) / 14.0)
                burst[i:i + 80] += env * np.sin(2 * np.pi * 340 * np.arange(80) / fs)
            ae += 0.55 * sev * burst
            cooling = (1.0 - 0.05 * sev) if len(fault_classes) == 1 else min(cooling, 1.0 - 0.05 * sev)
            cur += 0.02 * sev * np.sin(2 * np.pi * 5 * f_line * t)
        elif fc == 3:
            f_bpfo = rng.uniform(88.0, 122.0)
            imp = np.zeros(N)
            period = int(fs / f_bpfo)
            for i in range(0, N, max(period, 4)):
                L = min(60, N - i)
                imp[i:i + L] += np.exp(-np.arange(L) / 8.0) * np.sin(
                    2 * np.pi * 320 * np.arange(L) / fs)
            vib += 0.75 * sev * imp
            cooling = (1.0 - 0.22 * sev) if len(fault_classes) == 1 else min(cooling, 1.0 - 0.22 * sev)
            Ta = float(rng.normal(Ta_mu + 2.0, 4.0))
        elif fc == 4:
            n_pulse = int(20 + 90 * sev)
            ph = np.arange(N) * f_line / fs * 2 * np.pi
            w = (np.abs(np.sin(ph)) ** 6); w /= w.sum()
            idx = rng.choice(N, n_pulse, p=w)
            pd[idx] += rng.uniform(0.4, 1.6, n_pulse) * sev
            ae += 0.30 * sev * rng.normal(0, 1, N) * (np.abs(np.sin(ph)) ** 4)
            b2, a2 = sps.butter(4, [0.55, 0.95], btype="band")
            ae = sps.filtfilt(b2, a2, ae)
        elif fc == 5:
            step_at = int(rng.integers(N // 5, 4 * N // 5))
            vol[step_at:] *= (1.0 + rng.choice([-1, 1]) * 0.028 * (0.5 + sev))
            L = min(150, N - step_at)
            cur[step_at:step_at + L] += 0.9 * sev * np.exp(-np.arange(L) / 30.0)
            mech = np.zeros(N)
            mech[step_at:step_at + 40] = np.exp(-np.arange(40) / 9.0) * rng.normal(0, 1, 40)
            vib += 1.1 * sev * mech

    dTO_ss = top_oil_steady(K, p) / cooling
    Tto_ss = Ta + dTO_ss
    temp = Tto_ss + 0.06 * np.sin(2 * np.pi * 0.4 * t)
    chans = np.vstack([vib, temp, cur, vol, ae, pd])
    ac_idx = [0, 2, 3, 4, 5]
    chans[ac_idx] = _awgn_ac(chans[ac_idx], snr_db, rng)
    chans[1] = chans[1] + rng.normal(0, TEMP_NOISE_K, N)
    gain, off = calib["gain"], calib["offset"]
    t_gain, t_off = calib["t_gain"], calib["t_offset"]
    scale = np.abs(chans).max(axis=1, keepdims=True) + 1e-9
    chans = chans * gain + off * scale
    chans[1] = (Tto_ss + 0.06 * np.sin(2 * np.pi * 0.4 * t)
                + rng.normal(0, TEMP_NOISE_K, N)) * t_gain + t_off
    return chans


def build_shift_dataset(cfg, classes, n_per_class, seed, seq_len=16, sev_range=None,
                        calib_gain_sd=None, calib_off_sd=None, **kw):
    """Episodes under shifted conditions. `classes` is a list of tuples."""
    rng = np.random.default_rng(seed)
    lo, hi = sev_range if sev_range else (cfg.sev_lo, cfg.sev_hi)
    gsd = cfg.calib_gain_sd if calib_gain_sd is None else calib_gain_sd
    osd = cfg.calib_off_sd if calib_off_sd is None else calib_off_sd
    Xs, ys, sev_mean = [], [], []
    for ci, fcs in enumerate(classes):
        for _ in range(n_per_class):
            sev0 = rng.uniform(lo, hi * 0.6 if sev_range is None else lo + 0.6 * (hi - lo))
            sev1 = min(hi, sev0 + rng.uniform(0.03, 0.22) * (1 if sev_range is None
                                                              else (hi - lo) / 0.29))
            sevs = np.linspace(sev0, sev1, seq_len) if fcs != (0,) else np.zeros(seq_len)
            calib = dict(gain=1.0 + rng.normal(0, gsd, (cfg.n_channels, 1)),
                         offset=rng.normal(0, osd, (cfg.n_channels, 1)),
                         t_gain=1.0 + rng.normal(0, 0.004),
                         t_offset=rng.normal(0, TEMP_OFFSET_K))
            F = np.empty((seq_len, cfg.n_channels * cfg.feat_per_channel), dtype=np.float32)
            for k in range(seq_len):
                ch = synth_window_shift(fcs, cfg, rng, float(sevs[k]), calib=calib, **kw)
                F[k] = extract_features(ch, cfg)
            Xs.append(F); ys.append(ci); sev_mean.append(float(sevs.mean()))
    return np.asarray(Xs), np.asarray(ys), np.asarray(sev_mean)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def corrected_resampled_t(a, b, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test on paired fold scores."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return float(np.mean(d)), float("nan"), 1.0
    t = np.mean(d) / np.sqrt((1.0 / k + n_test / n_train) * var)
    pval = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(np.mean(d)), float(t), float(pval)


def mcnemar_exact(y, pa, pb):
    """Exact (binomial) McNemar test on paired predictions."""
    ca = pa == y; cb = pb == y
    n01 = int(np.sum(ca & ~cb)); n10 = int(np.sum(~ca & cb))
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    pval = min(1.0, 2 * stats.binom.cdf(min(n01, n10), n, 0.5))
    return n01, n10, float(pval)


def mean_ci(x, conf=0.95):
    x = np.asarray(x, float)
    m = x.mean(); s = x.std(ddof=1)
    h = stats.t.ppf(0.5 + conf / 2, len(x) - 1) * s / np.sqrt(len(x))
    return float(m), float(s), float(m - h), float(m + h)
