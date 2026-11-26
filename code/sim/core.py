"""
core.py -- Physical models, synthetic data generation and edge feature extraction
for the Edge-AI physics-informed digital-twin simulation study.

All thermal dynamics follow IEC 60076-7:2018 "Loading guide for oil-immersed
power transformers" (exponential-equation solution, Clause 8.2.1).

Author: Md Tabrez
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from scipy import signal as sps
from scipy import stats

# --------------------------------------------------------------------------
# 1. GLOBAL CONFIGURATION AND REPRODUCIBILITY
# --------------------------------------------------------------------------

GLOBAL_SEED = 20260905


@dataclass
class ThermalParams:
    """IEC 60076-7 exponential-equation thermal parameters.

    Default values correspond to a medium-power ONAF-cooled oil-immersed
    transformer (IEC 60076-7:2018, Table 4 / Annex E).
    """
    dTheta_or: float = 45.0     # top-oil rise above ambient at rated load [K]
    Hgr: float = 26.0           # hot-spot-to-top-oil gradient at rated current [K]
    R: float = 6.0              # ratio of load losses to no-load losses at rated load
    x: float = 0.8              # oil exponent (ONAF)
    y: float = 1.3              # winding exponent (ONAF); note K^y == K^(2m), m = y/2
    tau_o: float = 150.0        # oil time constant [min]
    tau_w: float = 7.0          # winding time constant [min]
    Theta_h_ref: float = 98.0   # reference hot-spot for unity ageing rate [degC]
    ageing_base: float = 6.0    # degC per doubling of ageing rate (non-upgraded paper)


@dataclass
class SimConfig:
    """Top-level simulation configuration."""
    seed: int = GLOBAL_SEED

    # --- asset definition ---
    n_phases: int = 6                 # 6-phase secondary (3-phase primary)
    rating_MVA: float = 20.0

    # --- sensing ---
    fs: float = 1000.0                # sensor sampling rate [Hz]
    window: int = 1024                # analysis window length [samples]
    overlap: int = 512                # window overlap [samples]
    f_cut: float = 400.0              # anti-alias / low-pass cutoff [Hz]
    n_channels: int = 6               # vib, temp, current, voltage, acoustic, PD
    feat_per_channel: int = 18
    quant_bits: int = 8

    # --- classification dataset ---
    n_windows_per_class: int = 1100
    n_classes: int = 6

    # --- noise and task difficulty ---
    snr_db: float = 8.0              # nominal sensor SNR
    sev_lo: float = 0.03              # incipient-fault severity, lower bound
    sev_hi: float = 0.32              # incipient-fault severity, upper bound
    calib_gain_sd: float = 0.12       # per-node sensor gain error (relative)
    calib_off_sd: float = 0.08        # per-node sensor offset error (relative)

    # --- fleet / RUL study ---
    n_units: int = 160
    horizon_days: int = 2200
    dt_hours: float = 6.0

    thermal: ThermalParams = field(default_factory=ThermalParams)

    def to_dict(self):
        d = asdict(self)
        return d


FAULT_CLASSES = [
    "Normal",
    "Thermal Overload",
    "Insulation Degradation",
    "Bearing / Cooling Fault",
    "Partial Discharge",
    "Tap Changer Malfunction",
]

CHANNEL_NAMES = ["Vibration", "Temperature", "Current", "Voltage",
                 "Acoustic Emission", "Partial Discharge"]


# --------------------------------------------------------------------------
# 2. IEC 60076-7 THERMAL MODEL
# --------------------------------------------------------------------------

def top_oil_steady(K, p: ThermalParams):
    """Steady-state top-oil rise above ambient for per-unit load K (IEC 60076-7)."""
    return p.dTheta_or * ((1.0 + p.R * K ** 2) / (1.0 + p.R)) ** p.x


def hotspot_rise_steady(K, p: ThermalParams):
    """Steady-state hot-spot-to-top-oil gradient for per-unit load K."""
    return p.Hgr * K ** p.y


def simulate_thermal(K_series, Theta_A_series, p: ThermalParams, dt_min,
                     cooling_derate=1.0, theta_to_0=None, theta_h_0=None):
    """Integrate the IEC 60076-7 differential (exponential) thermal equations.

        tau_o * d(dTheta_TO)/dt = dTheta_or*((1+R*K^2)/(1+R))^x - dTheta_TO
        tau_w * d(dTheta_H )/dt = Hgr*K^y                        - dTheta_H
        Theta_H = Theta_A + dTheta_TO + dTheta_H

    `cooling_derate` in (0, 1] scales the effective cooling capability; a value
    below unity models fan/pump degradation and raises the steady-state rise.

    Returns (Theta_H, Theta_TO, dTheta_TO, dTheta_H), all in degC/K.
    """
    K = np.asarray(K_series, dtype=float)
    Ta = np.asarray(Theta_A_series, dtype=float)
    n = K.shape[0]

    to_ss = top_oil_steady(K, p) / cooling_derate
    hs_ss = hotspot_rise_steady(K, p)

    dTO = np.empty(n)
    dH = np.empty(n)
    dTO[0] = to_ss[0] if theta_to_0 is None else theta_to_0
    dH[0] = hs_ss[0] if theta_h_0 is None else theta_h_0

    # exact exponential update over each step (stable for large dt)
    ao = 1.0 - np.exp(-dt_min / p.tau_o)
    aw = 1.0 - np.exp(-dt_min / p.tau_w)
    for i in range(1, n):
        dTO[i] = dTO[i - 1] + ao * (to_ss[i] - dTO[i - 1])
        dH[i] = dH[i - 1] + aw * (hs_ss[i] - dH[i - 1])

    Theta_TO = Ta + dTO
    Theta_H = Ta + dTO + dH
    return Theta_H, Theta_TO, dTO, dH


def relative_ageing_rate(Theta_H, p: ThermalParams):
    """IEC 60076-7 relative thermal ageing rate V = 2^((Theta_H - 98)/6)."""
    return 2.0 ** ((np.asarray(Theta_H) - p.Theta_h_ref) / p.ageing_base)


def simulate_multiphase_thermal(K_phase, Theta_A, p: ThermalParams, dt_min,
                                cooling_derate=1.0):
    """Coupled multi-phase thermal model.

    All phase windings share a single oil volume, so the top-oil rise is driven
    by the mean of the per-phase loss contributions, while each winding has its
    own hot-spot gradient. K_phase has shape (n_steps, n_phases).
    """
    K_phase = np.asarray(K_phase, dtype=float)
    n, nph = K_phase.shape
    Ta = np.asarray(Theta_A, dtype=float)

    # shared oil driven by average per-phase loss
    loss_term = np.mean((1.0 + p.R * K_phase ** 2) / (1.0 + p.R), axis=1)
    to_ss = p.dTheta_or * loss_term ** p.x / cooling_derate
    hs_ss = p.Hgr * K_phase ** p.y                     # (n, nph)

    ao = 1.0 - np.exp(-dt_min / p.tau_o)
    aw = 1.0 - np.exp(-dt_min / p.tau_w)

    dTO = np.empty(n)
    dH = np.empty((n, nph))
    dTO[0] = to_ss[0]
    dH[0] = hs_ss[0]
    for i in range(1, n):
        dTO[i] = dTO[i - 1] + ao * (to_ss[i] - dTO[i - 1])
        dH[i] = dH[i - 1] + aw * (hs_ss[i] - dH[i - 1])

    Theta_H = Ta[:, None] + dTO[:, None] + dH
    Theta_TO = Ta + dTO
    return Theta_H, Theta_TO, dTO, dH


# --------------------------------------------------------------------------
# 3. LOAD AND AMBIENT PROFILE GENERATION
# --------------------------------------------------------------------------

def airport_load_profile(n_steps, dt_min, rng, base=0.62, seasonal=True):
    """Per-unit load profile with airport-characteristic diurnal structure.

    Airport electrical demand shows a twin-peak diurnal pattern (morning and
    evening banks of departures) superimposed on airfield-lighting load which
    is strongly night-biased.
    """
    t_h = np.arange(n_steps) * dt_min / 60.0
    diurnal = (0.16 * np.sin(2 * np.pi * (t_h - 6) / 24.0)
               + 0.10 * np.sin(4 * np.pi * (t_h - 3) / 24.0))
    lighting = 0.08 * (np.cos(2 * np.pi * t_h / 24.0) * 0.5 + 0.5)
    if seasonal:
        seas = 0.07 * np.sin(2 * np.pi * t_h / (24 * 365.0))
    else:
        seas = 0.0
    # correlated stochastic component (AR(1))
    eps = rng.normal(0, 0.030, n_steps)
    ar = np.zeros(n_steps)
    for i in range(1, n_steps):
        ar[i] = 0.88 * ar[i - 1] + eps[i]
    K = base + diurnal + lighting + seas + ar
    return np.clip(K, 0.12, 1.55)


def ambient_profile(n_steps, dt_min, rng, mean_C=29.0, diurnal_amp=6.0,
                    seasonal_amp=7.0):
    """Ambient temperature profile for a tropical/sub-tropical airport site."""
    t_h = np.arange(n_steps) * dt_min / 60.0
    daily = diurnal_amp * np.sin(2 * np.pi * (t_h - 9) / 24.0)
    seas = seasonal_amp * np.sin(2 * np.pi * (t_h - 24 * 110) / (24 * 365.0))
    eps = rng.normal(0, 0.8, n_steps)
    ar = np.zeros(n_steps)
    for i in range(1, n_steps):
        ar[i] = 0.9 * ar[i - 1] + eps[i]
    return mean_C + daily + seas + ar


def phase_imbalance(K, n_phases, rng, imbalance=0.03):
    """Distribute a total per-unit load across phases with realistic imbalance."""
    n = len(K)
    dev = rng.normal(0, imbalance, (n, n_phases))
    # smooth the imbalance in time (imbalance drifts slowly)
    b, a = sps.butter(2, 0.02)
    dev = sps.filtfilt(b, a, dev, axis=0)
    dev -= dev.mean(axis=1, keepdims=True)
    return np.clip(K[:, None] * (1.0 + dev), 0.05, 1.9)


# --------------------------------------------------------------------------
# 4. SENSOR WAVEFORM SYNTHESIS AND FAULT INJECTION
# --------------------------------------------------------------------------

def _awgn_ac(x, snr_db, rng):
    """Add white Gaussian noise referenced to the AC (zero-mean) signal power.

    Referencing the noise to the total power would be wrong for a channel that
    carries a large DC offset: a top-oil temperature of 45 degC would then be
    assigned tens of kelvin of noise. Only the fluctuating component is used.
    """
    ac = x - x.mean(axis=-1, keepdims=True)
    p_sig = np.mean(ac ** 2, axis=-1, keepdims=True) + 1e-12
    p_noi = p_sig / (10 ** (snr_db / 10.0))
    return x + rng.normal(0, np.sqrt(p_noi), x.shape)


# Absolute measurement uncertainty of the temperature channel. A digital
# temperature sensor of the class assumed here has an absolute accuracy of a
# few tenths of a kelvin, which is a property of the instrument and not a
# fraction of the measured value.
TEMP_NOISE_K = 0.3
TEMP_OFFSET_K = 1.0


def synth_window(fault_class, cfg: SimConfig, rng, severity=None,
                 snr_db=None, return_state=False, calib=None):
    """Synthesise one multi-channel sensor observation window.

    Channel order: [vibration, temperature, current, voltage, acoustic, PD].
    Every fault signature is derived from the corresponding physical mechanism
    rather than from an arbitrary statistical template.

    Fault severities are drawn from the INCIPIENT range: the objective of
    predictive maintenance is to recognise a developing fault long before it
    becomes self-evident, so the discriminative task is deliberately posed in
    the low-severity regime where signatures are weak and partially
    overlapping.

    `calib` supplies per-node sensor calibration errors (gain and offset per
    channel), representing the unit-to-unit variability of low-cost sensors.
    """
    N = cfg.window
    fs = cfg.fs
    t = np.arange(N) / fs
    snr_db = cfg.snr_db if snr_db is None else snr_db
    sev = rng.uniform(cfg.sev_lo, cfg.sev_hi) if severity is None else severity
    p = cfg.thermal

    # nuisance variation: line frequency drift and operating-point spread
    f_line = 50.0 + rng.normal(0, 0.12)
    K = float(np.clip(rng.normal(0.68, 0.19), 0.15, 1.20))
    Ta = float(rng.normal(29.0, 7.0))
    cooling = float(np.clip(rng.normal(1.0, 0.035), 0.85, 1.05))
    imbal = 0.03

    # ---- vibration: core magnetostriction at 2f and harmonics ----
    vib = (0.55 * np.sin(2 * np.pi * 2 * f_line * t)
           + 0.22 * np.sin(2 * np.pi * 4 * f_line * t + 0.7)
           + 0.09 * np.sin(2 * np.pi * 6 * f_line * t + 1.9))
    vib *= (0.6 + 0.8 * K ** 2)

    # ---- current: fundamental + background harmonic content ----
    cur = np.sin(2 * np.pi * f_line * t)
    thd_bg = rng.uniform(0.015, 0.035)
    for h, amp in [(3, 0.55), (5, 0.30), (7, 0.15)]:
        cur += thd_bg * amp * np.sin(2 * np.pi * h * f_line * t + rng.uniform(0, 6.28))
    cur *= K

    # ---- voltage ----
    vol = np.sin(2 * np.pi * f_line * t + 0.05) * (1.0 - 0.012 * K)

    # ---- acoustic emission: broadband floor ----
    ae = rng.normal(0, 0.05, N)
    b, a = sps.butter(4, [0.30, 0.92], btype="band")
    ae = sps.filtfilt(b, a, ae)

    # ---- partial-discharge channel: envelope of UHF pulse activity ----
    pd = np.abs(rng.normal(0, 0.02, N))

    # ---- fault-specific mechanisms ----
    if fault_class == 1:      # Thermal overload
        K = float(np.clip(rng.normal(1.02, 0.07) + 0.30 * sev, 0.85, 1.45))
        Ta = float(rng.normal(33.0, 6.0))
        cur *= K / max(K, 1e-6)
        cur = np.sin(2 * np.pi * f_line * t) * K
        vib *= (0.6 + 0.8 * K ** 2) / (0.6 + 0.8 * 0.68 ** 2)
        ae *= 1.0 + 0.5 * sev

    elif fault_class == 2:    # Insulation degradation
        # elevated partial-discharge activity + AE bursts + slight thermal rise
        n_pulse = int(6 + 26 * sev)
        idx = rng.integers(0, N, n_pulse)
        pd[idx] += rng.uniform(0.25, 1.1, n_pulse) * sev
        burst = np.zeros(N)
        for i in rng.integers(0, N - 80, max(2, n_pulse // 3)):
            env = np.exp(-np.arange(80) / 14.0)
            burst[i:i + 80] += env * np.sin(2 * np.pi * 340 * np.arange(80) / fs)
        ae += 0.55 * sev * burst
        cooling = 1.0 - 0.05 * sev
        thd_extra = 0.02 * sev
        cur += thd_extra * np.sin(2 * np.pi * 5 * f_line * t)

    elif fault_class == 3:    # Bearing / cooling-fan fault
        # fan bearing outer-race defect -> impulse train; degraded cooling
        f_bpfo = rng.uniform(88.0, 122.0)
        imp = np.zeros(N)
        period = int(fs / f_bpfo)
        for i in range(0, N, max(period, 4)):
            L = min(60, N - i)
            imp[i:i + L] += np.exp(-np.arange(L) / 8.0) * np.sin(
                2 * np.pi * 320 * np.arange(L) / fs)
        vib += 0.75 * sev * imp
        cooling = 1.0 - 0.22 * sev
        Ta = float(rng.normal(31.0, 4.0))

    elif fault_class == 4:    # Partial discharge (void/surface discharge)
        n_pulse = int(20 + 90 * sev)
        # PD pulses phase-correlated with the AC voltage peaks
        ph = np.arange(N) * f_line / fs * 2 * np.pi
        w = (np.abs(np.sin(ph)) ** 6)
        w /= w.sum()
        idx = rng.choice(N, n_pulse, p=w)
        pd[idx] += rng.uniform(0.4, 1.6, n_pulse) * sev
        ae += 0.30 * sev * rng.normal(0, 1, N) * (np.abs(np.sin(ph)) ** 4)
        b2, a2 = sps.butter(4, [0.55, 0.95], btype="band")
        ae = sps.filtfilt(b2, a2, ae)

    elif fault_class == 5:    # Tap changer malfunction
        # contact bounce / motor stall -> voltage step + inrush + mech. impulse
        step_at = int(rng.integers(N // 5, 4 * N // 5))
        vol[step_at:] *= (1.0 + rng.choice([-1, 1]) * 0.028 * (0.5 + sev))
        L = min(150, N - step_at)
        cur[step_at:step_at + L] += 0.9 * sev * np.exp(-np.arange(L) / 30.0)
        mech = np.zeros(N)
        mech[step_at:step_at + 40] = np.exp(-np.arange(40) / 9.0) * rng.normal(0, 1, 40)
        vib += 1.1 * sev * mech
        imbal = 0.03 + 0.05 * sev

    # ---- temperature channel: TOP-OIL temperature (the measurable quantity).
    # The hot-spot temperature is NOT directly observable and must be inferred
    # by the physics model; exposing it here would leak the latent state.
    dTO_ss = top_oil_steady(K, p) / cooling
    Tto_ss = Ta + dTO_ss
    Th_ss = Tto_ss + hotspot_rise_steady(K, p)
    temp = Tto_ss + 0.06 * np.sin(2 * np.pi * 0.4 * t)

    chans = np.vstack([vib, temp, cur, vol, ae, pd])

    # AC channels: noise referenced to the fluctuating signal power
    ac_idx = [0, 2, 3, 4, 5]
    chans[ac_idx] = _awgn_ac(chans[ac_idx], snr_db, rng)
    # temperature channel: absolute instrument noise
    chans[1] = chans[1] + rng.normal(0, TEMP_NOISE_K, N)

    # per-node sensor calibration error (gain and offset)
    if calib is None:
        gain = 1.0 + rng.normal(0, cfg.calib_gain_sd, (cfg.n_channels, 1))
        off = rng.normal(0, cfg.calib_off_sd, (cfg.n_channels, 1))
        t_gain = 1.0 + rng.normal(0, 0.004)
        t_off = rng.normal(0, TEMP_OFFSET_K)
    else:
        gain, off = calib["gain"], calib["offset"]
        t_gain, t_off = calib["t_gain"], calib["t_offset"]
    scale = np.abs(chans).max(axis=1, keepdims=True) + 1e-9
    chans = chans * gain + off * scale
    # the temperature channel is re-derived with an absolute calibration error
    chans[1] = (Tto_ss + 0.06 * np.sin(2 * np.pi * 0.4 * t)
                + rng.normal(0, TEMP_NOISE_K, N)) * t_gain + t_off

    if return_state:
        state = dict(K=K, Theta_A=Ta, cooling=cooling, severity=sev,
                     Theta_H=float(Th_ss), Theta_TO=float(Tto_ss),
                     imbalance=imbal)
        return chans, state
    return chans


# --------------------------------------------------------------------------
# 5. EDGE PREPROCESSING PIPELINE  (manuscript Algorithm 1)
# --------------------------------------------------------------------------

# db4 (Daubechies-4) decomposition filters, used for the wavelet-packet energies
_DB4_LO = np.array([-0.010597401784997278, 0.032883011666982945,
                    0.030841381835986965, -0.18703481171888114,
                    -0.02798376941698385, 0.6308807679295904,
                    0.7148465705525415, 0.23037781330885523])
_DB4_HI = _DB4_LO[::-1] * np.array([1, -1] * 4)


def _dwt_level(x):
    """One level of a db4 discrete wavelet transform (periodic extension)."""
    lo = sps.convolve(np.concatenate([x[-7:], x]), _DB4_LO, mode="valid")[::2]
    hi = sps.convolve(np.concatenate([x[-7:], x]), _DB4_HI, mode="valid")[::2]
    return lo, hi


def wavelet_packet_energy(x, levels=3):
    """Normalised energies of a `levels`-deep db4 wavelet packet decomposition."""
    nodes = [x]
    for _ in range(levels):
        nxt = []
        for nd in nodes:
            lo, hi = _dwt_level(nd)
            nxt += [lo, hi]
        nodes = nxt
    e = np.array([np.sum(n ** 2) for n in nodes])
    return e / (e.sum() + 1e-12)


def channel_features(x, fs):
    """Extract the 18-dimensional per-channel feature vector of Algorithm 1.

    6 time-domain + 4 frequency-domain + 8 time-frequency (wavelet packet).
    """
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sd = x.std() + 1e-12
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    crest = peak / (rms + 1e-12)
    skew = stats.skew(x)
    kurt = stats.kurtosis(x)

    f, P = sps.welch(x, fs=fs, nperseg=min(256, len(x)))
    P = P + 1e-18
    Pn = P / P.sum()
    dom_f = f[np.argmax(P)]
    centroid = float(np.sum(f * Pn))
    spread = float(np.sqrt(np.sum(((f - centroid) ** 2) * Pn)))
    entropy = float(-np.sum(Pn * np.log(Pn)) / np.log(len(Pn)))

    wpe = wavelet_packet_energy(x, levels=3)          # 8 values

    return np.concatenate([
        [rms, sd, peak, crest, skew, kurt],       # time domain      (6)
        [dom_f, centroid, spread, entropy],       # frequency domain (4)
        wpe,                                      # wavelet packet   (8)
    ])


def extract_features(chans, cfg: SimConfig):
    """Concatenate per-channel features into the 108-D edge feature vector."""
    feats = [channel_features(chans[c], cfg.fs) for c in range(chans.shape[0])]
    return np.concatenate(feats)


def quantize_minmax(F, bits=8, lo=None, hi=None):
    """Min-max scale to [0,1] then uniformly quantise to `bits` (Algorithm 1)."""
    lo = F.min(axis=0) if lo is None else lo
    hi = F.max(axis=0) if hi is None else hi
    rng_ = np.where((hi - lo) < 1e-12, 1.0, hi - lo)
    Z = np.clip((F - lo) / rng_, 0.0, 1.0)
    L = 2 ** bits - 1
    Zq = np.round(Z * L) / L
    return Zq, lo, hi


# --------------------------------------------------------------------------
# 6. CLASSIFICATION DATASET CONSTRUCTION
# --------------------------------------------------------------------------

def build_classification_dataset(cfg: SimConfig, snr_db=None, seed=None,
                                 n_per_class=None, verbose=True):
    """Generate the labelled fault-classification dataset."""
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    n_pc = cfg.n_windows_per_class if n_per_class is None else n_per_class
    X, y, S = [], [], []
    for c in range(cfg.n_classes):
        for _ in range(n_pc):
            ch, st = synth_window(c, cfg, rng, snr_db=snr_db, return_state=True)
            X.append(extract_features(ch, cfg))
            y.append(c)
            S.append(st)
        if verbose:
            print(f"    class {c} ({FAULT_CLASSES[c]}): {n_pc} windows", flush=True)
    return np.asarray(X), np.asarray(y), S


def bandwidth_analysis(cfg: SimConfig, anomaly_fraction):
    """Upstream bandwidth accounting for raw vs. edge-preprocessed transmission.

    Raw mode transmits every sample of every channel at 16-bit resolution.
    Edge mode transmits a quantised 108-D feature vector per window, and falls
    back to raw transmission only for anomaly-flagged windows.
    """
    hop = cfg.window - cfg.overlap
    windows_per_day = 86400 * cfg.fs / hop

    raw_bytes_per_window = cfg.n_channels * hop * 2          # 16-bit samples
    feat_bytes = cfg.n_channels * cfg.feat_per_channel * cfg.quant_bits / 8
    header = 24                                              # OPC UA PubSub header

    raw_daily = windows_per_day * raw_bytes_per_window
    edge_daily = windows_per_day * (
        (1 - anomaly_fraction) * (feat_bytes + header)
        + anomaly_fraction * (raw_bytes_per_window + header))

    return dict(
        windows_per_day=windows_per_day,
        raw_bytes_per_window=raw_bytes_per_window,
        feature_bytes=feat_bytes + header,
        raw_daily_MB=raw_daily / 1e6,
        edge_daily_MB=edge_daily / 1e6,
        reduction_pct=100.0 * (1 - edge_daily / raw_daily),
        anomaly_fraction=anomaly_fraction,
    )


# --------------------------------------------------------------------------
# 7. EPISODE (SEQUENCE) GENERATION FOR RECURRENT MODELS
# --------------------------------------------------------------------------

def synth_episode(fault_class, cfg: SimConfig, rng, seq_len=16, snr_db=None):
    """Generate a temporally coherent sequence of observation windows.

    Within an episode the underlying asset condition evolves slowly: fault
    severity ramps monotonically (incipient-fault growth) while the load and
    ambient conditions follow the diurnal profile. This yields the temporal
    structure that recurrent models exploit.
    """
    sev0 = rng.uniform(cfg.sev_lo, cfg.sev_hi * 0.6)
    sev1 = min(cfg.sev_hi, sev0 + rng.uniform(0.03, 0.22))
    sevs = np.linspace(sev0, sev1, seq_len) if fault_class != 0 else \
        np.full(seq_len, 0.0)
    calib = dict(gain=1.0 + rng.normal(0, cfg.calib_gain_sd, (cfg.n_channels, 1)),
                 offset=rng.normal(0, cfg.calib_off_sd, (cfg.n_channels, 1)),
                 t_gain=1.0 + rng.normal(0, 0.004),
                 t_offset=rng.normal(0, TEMP_OFFSET_K))
    F = np.empty((seq_len, cfg.n_channels * cfg.feat_per_channel))
    states = []
    for k in range(seq_len):
        ch, st = synth_window(fault_class, cfg, rng, severity=float(sevs[k]),
                              snr_db=snr_db, return_state=True, calib=calib)
        F[k] = extract_features(ch, cfg)
        states.append(st)
    return F, states


def build_sequence_dataset(cfg: SimConfig, n_episodes_per_class=100, seq_len=16,
                           snr_db=None, seed=None, verbose=True):
    """Build the episode-structured fault-classification dataset.

    Returns
    -------
    Xseq : (n_episodes, seq_len, 108) float array of edge feature vectors
    y    : (n_episodes,) integer fault labels
    meta : list of per-episode physical state dictionaries
    """
    rng = np.random.default_rng(cfg.seed + 7 if seed is None else seed)
    Xs, ys, meta = [], [], []
    for c in range(cfg.n_classes):
        for _ in range(n_episodes_per_class):
            F, st = synth_episode(c, cfg, rng, seq_len=seq_len, snr_db=snr_db)
            Xs.append(F)
            ys.append(c)
            meta.append(st)
        if verbose:
            print(f"    class {c} ({FAULT_CLASSES[c]}): "
                  f"{n_episodes_per_class} episodes x {seq_len} windows", flush=True)
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys), meta


def physics_features(F, p: ThermalParams, feat_per_channel=18):
    """Physics-derived features computed from the OBSERVED feature vector.

    The physics module has access only to what the sensors measure: the top-oil
    temperature and the load current. It applies the IEC 60076-7 hot-spot
    equation with datasheet parameters to infer the unobservable hot-spot
    temperature and the resulting relative ageing rate. No latent state of the
    data generator is used, so no information leaks into the classifier.

    F may be (n, d) or (n, T, d).
    """
    F = np.asarray(F)
    flat = F.reshape(-1, F.shape[-1])
    theta_to = flat[:, 1 * feat_per_channel + 0]          # temperature ch. RMS
    i_rms = flat[:, 2 * feat_per_channel + 0]             # current ch. RMS
    K_hat = np.clip(np.sqrt(2.0) * i_rms, 0.02, 2.5)      # per-unit load estimate
    dH_hat = hotspot_rise_steady(K_hat, p)
    Th_hat = theta_to + dH_hat
    V_hat = relative_ageing_rate(Th_hat, p)
    out = np.stack([Th_hat, np.log10(V_hat + 1e-9), K_hat], axis=-1)
    return out.reshape(*F.shape[:-1], 3).astype(np.float32)


# --------------------------------------------------------------------------
# 8. FLEET DEGRADATION SIMULATION FOR RUL STUDY
# --------------------------------------------------------------------------

def simulate_unit_degradation(cfg: SimConfig, rng, unit_id=0):
    """Simulate the life history of one transformer unit.

    Insulation ageing follows the IEC 60076-7 relative ageing rate integrated
    over the hot-spot temperature trajectory. End of life is declared when the
    cumulative loss of life reaches the unit's (randomly drawn) endurance,
    representing manufacturing and material variability across the fleet.
    """
    p = cfg.thermal
    dt_min = cfg.dt_hours * 60.0
    n = int(cfg.horizon_days * 24 / cfg.dt_hours)

    base = float(np.clip(rng.normal(0.66, 0.11), 0.35, 0.95))
    K = airport_load_profile(n, dt_min, rng, base=base)
    Ta = ambient_profile(n, dt_min, rng, mean_C=float(rng.normal(29, 3)))

    # progressive cooling-system degradation (fouling of radiators/fans)
    fouling_rate = float(np.clip(rng.normal(0.06, 0.03), 0.0, 0.20))
    cooling = 1.0 - fouling_rate * np.linspace(0, 1, n) ** 1.5

    to_ss = top_oil_steady(K, p) / cooling
    hs_ss = hotspot_rise_steady(K, p)
    ao = 1.0 - np.exp(-dt_min / p.tau_o)
    aw = 1.0 - np.exp(-dt_min / p.tau_w)
    dTO = np.empty(n); dH = np.empty(n)
    dTO[0], dH[0] = to_ss[0], hs_ss[0]
    for i in range(1, n):
        dTO[i] = dTO[i - 1] + ao * (to_ss[i] - dTO[i - 1])
        dH[i] = dH[i - 1] + aw * (hs_ss[i] - dH[i - 1])
    Th = Ta + dTO + dH

    V = relative_ageing_rate(Th, p)
    lol = np.cumsum(V) * cfg.dt_hours                      # equivalent ageing hours
    endurance = float(rng.lognormal(np.log(lol[-1] * 0.55), 0.26))

    # observable condition indicators driven by the same latent damage state
    d = np.clip(lol / endurance, 0, 3)
    thd = 0.022 + 0.055 * d ** 1.8 + rng.normal(0, 0.004, n)
    vib = 1.0 + 0.85 * d ** 2.1 + rng.normal(0, 0.06, n)
    ae = 0.10 + 0.75 * d ** 2.4 + rng.normal(0, 0.02, n)

    eol_idx = int(np.argmax(lol >= endurance)) if np.any(lol >= endurance) else -1
    return dict(unit_id=unit_id, K=K, Theta_A=Ta, Theta_H=Th, dTO=dTO, dH=dH,
                V=V, lol=lol, endurance=endurance, eol_idx=eol_idx,
                thd=thd, vib=vib, ae=ae, cooling=cooling, n=n)


def health_index(Th, thd, vib, ae, p: ThermalParams,
                 w=(0.4, 0.2, 0.2, 0.2),
                 Th_nom=98.0, Th_max=140.0,
                 thd_lim=0.08, vib_nom=1.0, ae_lim=0.9):
    """Composite health index (manuscript Eq. for HI).

    HI is a weighted, normalised aggregation of thermal, harmonic, vibration
    and acoustic-emission deviation from nominal, clipped to [0, 1].
    """
    t1 = (np.asarray(Th) - Th_nom) / (Th_max - Th_nom)
    t2 = np.asarray(thd) / thd_lim
    t3 = (np.asarray(vib) - vib_nom) / vib_nom
    t4 = np.asarray(ae) / ae_lim
    hi = (w[0] * np.clip(t1, 0, None) + w[1] * np.clip(t2, 0, None)
          + w[2] * np.clip(t3, 0, None) + w[3] * np.clip(t4, 0, None))
    return np.clip(hi, 0.0, 1.0)
