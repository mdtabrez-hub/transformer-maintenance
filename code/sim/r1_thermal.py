"""
r1_thermal.py -- Revision-1 experiments on hot-spot state estimation.

Adds, in response to the reviewers of Access-2026-53764:

  * structurally different ground-truth thermal generators
      S1  IEC 60076-7 two-component hot-spot model (k21, k22 oil-flow lag)
      S2  viscosity-dependent nonlinear model of Susa, Lehtonen and Nordman
          (IEEE Trans. Power Del., 2005): oil viscosity and winding-loss
          temperature dependence, exponent n = 0.25
      S3  cooling-state change (fan outage, ONAF -> ONAN) unknown to estimators
      S4  different design (ONAN distribution-class unit) and loading profile
      S5  sensor calibration bias and unknown initial thermal state
  * conventional estimators that receive the same inputs as the PINN
      IEC forward model, Kalman/RTS smoother, augmented EKF/RTS with joint
      parameter estimation, identified gray-box (MAP least squares)
  * a local identifiability analysis of the lumped thermal model

Author: Md Tabrez
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from scipy import optimize

from .core import (ThermalParams, airport_load_profile, ambient_profile,
                   simulate_thermal)
from .models import PINN, _interp_torch, set_seed, perturb_params

PNAMES = ("dTheta_or", "R", "x", "tau_o", "Hgr", "y", "tau_w")


# --------------------------------------------------------------------------
# 1. GROUND-TRUTH GENERATORS
# --------------------------------------------------------------------------

def _step_first_order(K, Ta, pars, dt, dTO0, dH0):
    """IEC first-order model with possibly time-varying parameter list."""
    n = len(K)
    dTO = np.empty(n); dH = np.empty(n)
    dTO[0], dH[0] = dTO0, dH0
    for i in range(1, n):
        p, kscale = pars[i]
        Ke = K[i] * kscale
        to_ss = p.dTheta_or * ((1 + p.R * Ke ** 2) / (1 + p.R)) ** p.x
        hs_ss = p.Hgr * Ke ** p.y
        ao = 1 - np.exp(-dt / p.tau_o); aw = 1 - np.exp(-dt / p.tau_w)
        dTO[i] = dTO[i - 1] + ao * (to_ss - dTO[i - 1])
        dH[i] = dH[i - 1] + aw * (hs_ss - dH[i - 1])
    return dTO, dH


def ss_first_order(K, p):
    return (p.dTheta_or * ((1 + p.R * K ** 2) / (1 + p.R)) ** p.x,
            p.Hgr * K ** p.y)


def gen_iec_two_component(K, Ta, p, dt, k21=2.0, k22=2.0, dTO0=None, dH0=None):
    """IEC 60076-7 hot-spot model with the oil-flow lag (two-component gradient).

        k22*tau_w dH1/dt = k21*Hgr*K^y - dH1
        (tau_o/k22) dH2/dt = (k21-1)*Hgr*K^y - dH2,     dH = dH1 - dH2
    Top oil follows the same first-order equation as the estimators assume,
    so the structural difference lies in the hot-spot dynamics only.
    """
    n = len(K)
    to_ss, hs_ss = ss_first_order(K, p)
    dTO = np.empty(n); h1 = np.empty(n); h2 = np.empty(n)
    dTO[0] = to_ss[0] if dTO0 is None else dTO0
    h1[0] = k21 * hs_ss[0]; h2[0] = (k21 - 1) * hs_ss[0]
    if dH0 is not None:
        h1[0] += dH0 - hs_ss[0]
    sub = 10; h = dt / sub
    for i in range(1, n):
        a, b, c = dTO[i - 1], h1[i - 1], h2[i - 1]
        for _ in range(sub):
            a += h / p.tau_o * (to_ss[i] - a)
            b += h / (k22 * p.tau_w) * (k21 * hs_ss[i] - b)
            c += h / (p.tau_o / k22) * ((k21 - 1) * hs_ss[i] - c)
        dTO[i], h1[i], h2[i] = a, b, c
    return dTO, h1 - h2


def _visc(theta):
    """Transformer-oil dynamic viscosity (Pa s), Susa et al. (2005)."""
    return 0.0013573 * np.exp(2797.3 / (theta + 273.0))


def gen_susa(K, Ta, p, dt, n_exp=0.25, Pdc=0.85, Pec=0.15, theta_k=235.0,
             amb_rated=20.0, dTO0=None, dH0=None):
    """Viscosity-dependent thermal model (Susa, Lehtonen, Nordman, 2005).

    Top oil:  q * dT_or = tau_o dT_o/dt * mu^n + dT_o^(1+n)/dT_or^n   (per mu^n)
    Hot spot: K^2 P_w(theta_h) * dT_hr = tau_w dT_h/dt * mu^n + dT_h^(1+n)/dT_hr^n
    with q = (1+R K^2)/(1+R), mu the per-unit oil viscosity and P_w the
    winding-loss temperature correction (dc and eddy components).
    """
    n = len(K)
    th_o_r = amb_rated + p.dTheta_or
    th_h_r = th_o_r + p.Hgr
    mu_r = _visc(th_o_r)
    dTO = np.empty(n); dH = np.empty(n)
    q0 = (1 + p.R * K[0] ** 2) / (1 + p.R)
    dTO[0] = p.dTheta_or * q0 ** (1 / (1 + n_exp)) if dTO0 is None else dTO0
    dH[0] = p.Hgr * (K[0] ** 2) ** (1 / (1 + n_exp)) if dH0 is None else dH0
    sub = 20; h = dt / sub
    for i in range(1, n):
        a, b = dTO[i - 1], dH[i - 1]
        q = (1 + p.R * K[i] ** 2) / (1 + p.R)
        for _ in range(sub):
            th_o = Ta[i] + a
            mu = (_visc(th_o) / mu_r) ** n_exp
            a += h / p.tau_o * (q * p.dTheta_or - max(a, 0.0) ** (1 + n_exp)
                                / p.dTheta_or ** n_exp / mu)
            th_h = th_o + b
            Pw = Pdc * (theta_k + th_h) / (theta_k + th_h_r) \
                + Pec * (theta_k + th_h_r) / (theta_k + th_h)
            b += h / p.tau_w * (K[i] ** 2 * Pw * p.Hgr - max(b, 0.0) ** (1 + n_exp)
                                / p.Hgr ** n_exp / mu)
        dTO[i], dH[i] = a, b
    return dTO, dH


def industrial_profile(n, dt, rng, day_level=0.85, night_level=0.42,
                       overload_day=3, overload_K=1.25, overload_h=2.0):
    """Two-shift industrial load with a planned emergency overload."""
    t_h = np.arange(n) * dt / 60.0
    hod = t_h % 24
    K = np.where((hod >= 6) & (hod < 22), day_level, night_level).astype(float)
    # smooth the shift transitions over ~30 min
    w = max(1, int(30 / dt))
    K = np.convolve(np.pad(K, (w, w), mode="edge"), np.ones(2 * w + 1) / (2 * w + 1),
                    mode="same")[w:-w]
    i0 = int((overload_day * 24 + 14) * 60 / dt); i1 = i0 + int(overload_h * 60 / dt)
    K[i0:i1] = overload_K
    eps = rng.normal(0, 0.02, n); ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.85 * ar[i - 1] + eps[i]
    return np.clip(K + ar, 0.1, 1.6)


ONAF = ThermalParams()
ONAN_DIST = ThermalParams(dTheta_or=55.0, Hgr=23.0, R=5.0, x=0.8, y=1.6,
                          tau_o=180.0, tau_w=4.0)


def make_scenario(name, seed, days=6, dt=5.0):
    """Return a ground-truth trajectory and its nominal (datasheet) parameters."""
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60 / dt)
    if name == "S4":
        K = industrial_profile(n, dt, rng)
        Ta = ambient_profile(n, dt, rng, mean_C=22.0)
        nominal = ONAN_DIST
    else:
        K = airport_load_profile(n, dt, rng)
        Ta = ambient_profile(n, dt, rng)
        nominal = ONAF
    p = nominal
    to_ss, hs_ss = ss_first_order(K, p)
    if name in ("S0", "S4", "S5"):
        dTO0, dH0 = to_ss[0], hs_ss[0]
        if name == "S5":                       # unknown initial thermal state
            dTO0, dH0 = dTO0 + 6.0, dH0 + 4.0
        dTO, dH = _step_first_order(K, Ta, [(p, 1.0)] * n, dt, dTO0, dH0)
    elif name == "S1":
        dTO, dH = gen_iec_two_component(K, Ta, p, dt)
    elif name == "S2":
        dTO, dH = gen_susa(K, Ta, p, dt)
    elif name == "S3":                         # fan outage on day 2.5-4.0
        onan = ThermalParams(dTheta_or=p.dTheta_or, Hgr=p.Hgr, R=p.R, x=p.x,
                             y=p.y, tau_o=210.0, tau_w=10.0)
        t_d = np.arange(n) * dt / 1440.0
        pars = [((onan, 1 / 0.8) if 2.5 <= td < 4.0 else (p, 1.0)) for td in t_d]
        dTO, dH = _step_first_order(K, Ta, pars, dt, to_ss[0], hs_ss[0])
    else:
        raise ValueError(name)
    return dict(name=name, t_min=np.arange(n) * dt, K=K, Theta_A=Ta, dTO=dTO,
                dH=dH, Theta_H=Ta + dTO + dH, nominal=nominal, dt=dt)


def make_observations(traj, seed, n_obs_to=180, n_obs_hs=6, noise_K=0.5,
                      gain=1.0, offset=0.0):
    """Same random-number ordering as sim.models.train_pinn."""
    rng = np.random.default_rng(seed)
    n = len(traj["t_min"])
    i_to = np.sort(rng.choice(n, size=min(n_obs_to, n), replace=False))
    Tto = traj["Theta_A"][i_to] + traj["dTO"][i_to]
    meas = gain * Tto + offset
    obs_to = meas - traj["Theta_A"][i_to] + rng.normal(0, noise_K, len(i_to))
    if n_obs_hs > 0:
        i_hs = np.sort(rng.choice(n, size=n_obs_hs, replace=False))
        obs_hs = traj["dH"][i_hs] + rng.normal(0, noise_K, n_obs_hs)
    else:
        i_hs = np.array([], dtype=int); obs_hs = np.array([])
    return dict(i_to=i_to, obs_to=obs_to, i_hs=i_hs, obs_hs=obs_hs,
                noise_K=noise_K)


def initial_condition(traj, p_assumed, mode):
    """'exact' (original protocol) or 'assumed' (steady state of assumed model)."""
    if mode == "exact":
        return float(traj["dTO"][0]), float(traj["dH"][0])
    a, b = ss_first_order(np.array([traj["K"][0]]), p_assumed)
    return float(a[0]), float(b[0])


def metrics(traj, dTO_hat, dH_hat):
    Th_hat = traj["Theta_A"] + dTO_hat + dH_hat
    e = Th_hat - traj["Theta_H"]
    eto = dTO_hat - traj["dTO"]
    eh = dH_hat - traj["dH"]
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                maxerr=float(np.max(np.abs(e))),
                r2=float(1 - np.sum(e ** 2) / np.sum((traj["Theta_H"] - traj["Theta_H"].mean()) ** 2)),
                rmse_to=float(np.sqrt(np.mean(eto ** 2))),
                rmse_grad=float(np.sqrt(np.mean(eh ** 2))))


# --------------------------------------------------------------------------
# 2. ESTIMATORS
# --------------------------------------------------------------------------

def est_pinn(traj, obs, tp, ic, lam_p=10.0, lam_d=1.0, lam_bc=0.1, n_coll=1500,
             epochs=1800, lr=2e-3, seed=0, hidden=64, layers=5, inverse=False,
             prior_sd=0.10):
    """PINN identical to sim.models.train_pinn, with inputs supplied explicitly.

    With inverse=True the seven thermal constants in the residual are
    multiplied by trainable log-scale factors z, penalized by the same 10 %
    log-normal prior that regularizes the gray-box model (inverse PINN).
    """
    set_seed(seed)
    t = traj["t_min"]; T = float(t[-1]); tn = t / T
    tg = torch.tensor(tn, dtype=torch.float32)
    Kg = torch.tensor(traj["K"], dtype=torch.float32)
    t_to = torch.tensor(tn[obs["i_to"]], dtype=torch.float32).view(-1, 1)
    y_to = torch.tensor(obs["obs_to"], dtype=torch.float32).view(-1, 1)
    if len(obs["i_hs"]) > 0:
        t_hs = torch.tensor(tn[obs["i_hs"]], dtype=torch.float32).view(-1, 1)
        y_hs = torch.tensor(obs["obs_hs"], dtype=torch.float32).view(-1, 1)
    else:
        t_hs = y_hs = None
    t_coll = torch.rand(n_coll, 1, requires_grad=True)
    model = PINN(hidden=hidden, layers=layers, max_freq=max(8.0, T / 240.0))
    z = torch.zeros(7, requires_grad=inverse)
    params = list(model.parameters()) + ([z] if inverse else [])
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 5), gamma=0.5)
    # prior weight equivalent to the gray-box MAP objective divided by N_d
    w_prior = obs["noise_K"] ** 2 / max(1, len(obs["i_to"])) / prior_sd ** 2
    for ep in range(epochs):
        opt.zero_grad()
        s = torch.exp(z)
        dor, R_, x_, to_, hg, y_, tw = (tp.dTheta_or * s[0], tp.R * s[1], tp.x * s[2],
                                        tp.tau_o * s[3], tp.Hgr * s[4], tp.y * s[5],
                                        tp.tau_w * s[6])
        out_to = model(t_to)
        L_data = F.mse_loss(out_to[:, 0:1], y_to)
        if t_hs is not None:
            L_data = L_data + F.mse_loss(model(t_hs)[:, 1:2], y_hs)
        if lam_p > 0:
            out_c = model(t_coll)
            dTO_c, dH_c = out_c[:, 0:1], out_c[:, 1:2]
            g = torch.ones_like(dTO_c)
            d_dTO = torch.autograd.grad(dTO_c, t_coll, grad_outputs=g, create_graph=True)[0]
            d_dH = torch.autograd.grad(dH_c, t_coll, grad_outputs=g, create_graph=True)[0]
            Kc = _interp_torch(t_coll, tg, Kg)
            to_ss = dor * ((1 + R_ * Kc ** 2) / (1 + R_)) ** x_
            hs_ss = hg * torch.clamp(Kc, min=1e-3) ** y_
            r1 = (to_ / T) * d_dTO - (to_ss - dTO_c)
            r2 = (tw / T) * d_dH - (hs_ss - dH_c)
            L_phys = (r1 ** 2).mean() / 100.0 + (r2 ** 2).mean() / 100.0
        else:
            L_phys = torch.tensor(0.0)
        out0 = model(torch.zeros(1, 1))
        L_bc = (out0[0, 0] - ic[0]) ** 2 + (out0[0, 1] - ic[1]) ** 2
        loss = lam_d * L_data + lam_p * L_phys + lam_bc * L_bc
        if inverse:
            loss = loss + w_prior * (z ** 2).sum()
        loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        pred = model(torch.tensor(tn, dtype=torch.float32).view(-1, 1)).numpy()
    if inverse:
        return pred[:, 0].astype(float), pred[:, 1].astype(float), torch.exp(z).detach().numpy().tolist()
    return pred[:, 0].astype(float), pred[:, 1].astype(float)


def est_iec_forward(traj, obs, tp, ic):
    n = len(traj["K"])
    return _step_first_order(traj["K"], traj["Theta_A"], [(tp, 1.0)] * n,
                             traj["dt"], ic[0], ic[1])


def est_kalman_rts(traj, obs, tp, ic, q_to=0.2, q_h=0.2, p0=4.0):
    """Linear time-varying Kalman filter + Rauch-Tung-Striebel smoother.

    State x = [dTheta_TO, dTheta_H]; the IEC model is linear in x once the
    measured load is given. Top-oil and (when available) hot-spot observations
    update the state at their sampling instants.
    """
    K = traj["K"]; n = len(K); dt = traj["dt"]
    to_ss, hs_ss = ss_first_order(K, tp)
    ao = 1 - np.exp(-dt / tp.tau_o); aw = 1 - np.exp(-dt / tp.tau_w)
    A = np.diag([1 - ao, 1 - aw])
    Q = np.diag([q_to ** 2, q_h ** 2])
    Rm = obs["noise_K"] ** 2
    mt = {int(i): v for i, v in zip(obs["i_to"], obs["obs_to"])}
    mh = {int(i): v for i, v in zip(obs["i_hs"], obs["obs_hs"])}
    xf = np.zeros((n, 2)); Pf = np.zeros((n, 2, 2))
    xp = np.zeros((n, 2)); Pp = np.zeros((n, 2, 2))
    x = np.array(ic, float); P = np.eye(2) * p0 ** 2
    for k in range(n):
        if k > 0:
            u = np.array([ao * to_ss[k], aw * hs_ss[k]])
            x = A @ x + u; P = A @ P @ A.T + Q
        xp[k], Pp[k] = x, P
        for idx, val in ((0, mt.get(k)), (1, mh.get(k))):
            if val is None:
                continue
            H = np.zeros(2); H[idx] = 1.0
            S = H @ P @ H + Rm
            Kg = P @ H / S
            x = x + Kg * (val - H @ x)
            P = P - np.outer(Kg, H @ P)
        xf[k], Pf[k] = x, P
    xs = xf.copy(); Ps = Pf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ A.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs[:, 0], xs[:, 1]


def est_ekf_joint(traj, obs, tp, ic, q_to=0.2, q_h=0.2, q_par=2e-3,
                  prior_sd=0.10, p0=4.0):
    """Augmented-state EKF + RTS smoother with joint parameter estimation.

    x = [dTheta_TO, dTheta_H, a, b], with the rated top-oil rise and rated
    hot-spot gradient scaled by exp(a) and exp(b); a and b follow random
    walks with a 10 % prior. Smoothed parameter estimates are then used to
    re-simulate the trajectory from the smoothed initial state.
    """
    K = traj["K"]; n = len(K); dt = traj["dt"]
    to_ss, hs_ss = ss_first_order(K, tp)
    ao = 1 - np.exp(-dt / tp.tau_o); aw = 1 - np.exp(-dt / tp.tau_w)
    Q = np.diag([q_to ** 2, q_h ** 2, q_par ** 2, q_par ** 2])
    Rm = obs["noise_K"] ** 2
    mt = {int(i): v for i, v in zip(obs["i_to"], obs["obs_to"])}
    mh = {int(i): v for i, v in zip(obs["i_hs"], obs["obs_hs"])}
    x = np.array([ic[0], ic[1], 0.0, 0.0])
    P = np.diag([p0 ** 2, p0 ** 2, prior_sd ** 2, prior_sd ** 2])
    xf = np.zeros((n, 4)); Pf = np.zeros((n, 4, 4))
    xp = np.zeros((n, 4)); Pp = np.zeros((n, 4, 4)); Fs = np.zeros((n, 4, 4))
    for k in range(n):
        if k > 0:
            ea, eb = np.exp(x[2]), np.exp(x[3])
            xn = np.array([(1 - ao) * x[0] + ao * ea * to_ss[k],
                           (1 - aw) * x[1] + aw * eb * hs_ss[k], x[2], x[3]])
            Fk = np.array([[1 - ao, 0, ao * ea * to_ss[k], 0],
                           [0, 1 - aw, 0, aw * eb * hs_ss[k]],
                           [0, 0, 1, 0], [0, 0, 0, 1]])
            x = xn; P = Fk @ P @ Fk.T + Q; Fs[k] = Fk
        xp[k], Pp[k] = x, P
        for idx, val in ((0, mt.get(k)), (1, mh.get(k))):
            if val is None:
                continue
            H = np.zeros(4); H[idx] = 1.0
            S = H @ P @ H + Rm
            Kg = P @ H / S
            x = x + Kg * (val - H @ x)
            P = P - np.outer(Kg, H @ P)
        xf[k], Pf[k] = x, P
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ Fs[k + 1].T @ np.linalg.pinv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
    return xs[:, 0], xs[:, 1], dict(a=float(xs[0, 2]), b=float(xs[0, 3]))


def est_graybox(traj, obs, tp, ic, prior_sd=0.10):
    """Identified gray-box IEC model: MAP fit of log-scale parameter corrections.

    Top-oil parameters (dTheta_or, R, x, tau_o) are fitted to the top-oil
    observations; hot-spot parameters (Hgr, y, tau_w) can only be fitted when
    hot-spot labels exist and are otherwise held at their datasheet values.
    The initial top-oil state is estimated jointly. A 10 % log-normal prior
    regularizes every scale factor, which is the same calibration information
    the PINN receives through its assumed parameters.
    """
    K = traj["K"]; Ta = traj["Theta_A"]; n = len(K); dt = traj["dt"]
    has_hs = len(obs["i_hs"]) > 0
    free = ["dTheta_or", "R", "x", "tau_o"] + (["Hgr", "y", "tau_w"] if has_hs else [])
    sig = obs["noise_K"]

    def params(z):
        q = copy.deepcopy(tp)
        for nm, v in zip(free, z[:len(free)]):
            setattr(q, nm, getattr(tp, nm) * float(np.exp(v)))
        return q

    def sim(z):
        q = params(z)
        dTO0 = ic[0] + z[len(free)]
        _, hs0 = ss_first_order(np.array([K[0]]), q)
        return _step_first_order(K, Ta, [(q, 1.0)] * n, dt, dTO0, float(hs0[0]))

    def resid(z):
        a, b = sim(z)
        r = [(a[obs["i_to"]] - obs["obs_to"]) / sig]
        if has_hs:
            r.append((b[obs["i_hs"]] - obs["obs_hs"]) / sig)
        r.append(z[:len(free)] / prior_sd)
        r.append(np.array([z[len(free)] / 5.0]))
        return np.concatenate(r)

    z0 = np.zeros(len(free) + 1)
    sol = optimize.least_squares(resid, z0, method="trf", x_scale=0.1, max_nfev=400)
    a, b = sim(sol.x)
    return a, b, dict(zip(free, np.exp(sol.x[:len(free)]).tolist()))


# --------------------------------------------------------------------------
# 3. IDENTIFIABILITY
# --------------------------------------------------------------------------

def identifiability(traj, obs, tp, prior_sd=None, h=1e-4):
    """Local identifiability from the output sensitivity matrix.

    Rows: observed outputs (top-oil at its sampling instants, plus hot-spot
    gradient at the label instants if any) divided by the noise level.
    Columns: log-scale perturbations of the seven thermal parameters.
    Returns singular values and the Cramer-Rao standard deviation of each
    log-parameter (infinite if the parameter is not identifiable).
    """
    K = traj["K"]; Ta = traj["Theta_A"]; n = len(K); dt = traj["dt"]

    def out(z):
        q = copy.deepcopy(tp)
        for nm, v in zip(PNAMES, z):
            setattr(q, nm, getattr(tp, nm) * float(np.exp(v)))
        a0, b0 = ss_first_order(np.array([K[0]]), q)
        a, b = _step_first_order(K, Ta, [(q, 1.0)] * n, dt, float(a0[0]), float(b0[0]))
        o = [a[obs["i_to"]]]
        if len(obs["i_hs"]) > 0:
            o.append(b[obs["i_hs"]])
        return np.concatenate(o) / obs["noise_K"], a, b

    z0 = np.zeros(len(PNAMES))
    y0, a0, b0 = out(z0)
    J = np.zeros((len(y0), len(PNAMES)))
    for j in range(len(PNAMES)):
        z = z0.copy(); z[j] = h
        J[:, j] = (out(z)[0] - y0) / h
    s = np.linalg.svd(J, compute_uv=False)
    FIM = J.T @ J
    if prior_sd:
        FIM = FIM + np.eye(len(PNAMES)) / prior_sd ** 2
    try:
        cov = np.linalg.pinv(FIM, rcond=1e-12)
        crb = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        crb = np.full(len(PNAMES), np.inf)
    # columns with (numerically) zero sensitivity are not identifiable
    colnorm = np.linalg.norm(J, axis=0)
    return dict(singular_values=s.tolist(), col_norm=colnorm.tolist(),
                crb_logsd=crb.tolist(), rank=int(np.sum(s > 1e-6 * s[0])))
