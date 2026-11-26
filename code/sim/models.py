"""
models.py -- Learning models for the physics-informed edge digital-twin study.

Contains:
  * PINN            : physics-informed reconstruction of transformer hot-spot
                      temperature from sparse top-oil observations (IEC 60076-7)
  * Teacher/Student : Bi-GRU + Random Forest ensemble and its knowledge-distilled
                      lightweight counterpart
  * VAE             : variational autoencoder for sensor-integrity validation
  * Baselines       : Random Forest, GRU, CNN-LSTM
  * WeibullRUL      : two-parameter Weibull RUL estimator with confidence bounds
  * Edge cost model : FLOP counting, INT8 quantisation, memory footprint

Author: Md Tabrez
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import optimize, stats
from sklearn.ensemble import RandomForestClassifier

torch.set_num_threads(1)


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


# ==========================================================================
# 1. PHYSICS-INFORMED NEURAL NETWORK
# ==========================================================================

class FourierFeatures(nn.Module):
    """Deterministic Fourier encoding of the (scalar, normalised) time input.

    A plain tanh multilayer perceptron exhibits strong spectral bias and cannot
    resolve the multi-scale diurnal forcing of the thermal problem within a
    practical training budget. Encoding the input as
    [t, sin(2*pi*k*t), cos(2*pi*k*t)] for a geometric set of frequencies k
    removes this limitation while leaving the network differentiable in t, so
    the automatic-differentiation residual remains exact.
    """

    def __init__(self, n_freq=12, max_freq=64.0):
        super().__init__()
        k = torch.exp(torch.linspace(0, np.log(max_freq), n_freq))
        self.register_buffer("k", k)
        self.out_dim = 1 + 2 * n_freq

    def forward(self, t):
        a = 2.0 * np.pi * t * self.k
        return torch.cat([t, torch.sin(a), torch.cos(a)], dim=-1)


class PINN(nn.Module):
    """Fully connected network mapping normalised time to thermal state.

    Outputs the top-oil rise above ambient and the hot-spot-to-top-oil
    gradient, from which the hot-spot temperature follows as
    Theta_H = Theta_A + dTheta_TO + dTheta_H.
    """

    def __init__(self, hidden=64, layers=5, n_out=2, n_freq=12, max_freq=64.0):
        super().__init__()
        self.ff = FourierFeatures(n_freq=n_freq, max_freq=max_freq)
        seq = [nn.Linear(self.ff.out_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.Tanh()]
        seq += [nn.Linear(hidden, n_out)]
        self.net = nn.Sequential(*seq)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        # output scaling keeps the network in a well-conditioned range
        self.scale = nn.Parameter(torch.tensor([50.0, 30.0][:n_out]),
                                  requires_grad=False)

    def forward(self, t):
        return self.net(self.ff(t)) * self.scale


def _interp_torch(t_query, t_grid, values):
    """Linear interpolation of a forcing signal at arbitrary query times."""
    idx = torch.clamp(torch.searchsorted(t_grid, t_query.squeeze(-1)), 1,
                      len(t_grid) - 1)
    t0, t1 = t_grid[idx - 1], t_grid[idx]
    v0, v1 = values[idx - 1], values[idx]
    w = (t_query.squeeze(-1) - t0) / (t1 - t0 + 1e-12)
    return (v0 + w * (v1 - v0)).unsqueeze(-1)


def perturb_params(tp, rel=0.10, seed=0):
    """Return thermal parameters perturbed by a relative error.

    In practice the thermal constants embedded in the physics loss come from
    the manufacturer test report or from IEC 60076-7 default tables, and
    therefore differ from the true parameters of the individual unit. Modelling
    this mismatch is essential: without it the physics term is exact and the
    physics weight can be raised without limit, which would not hold in a real
    deployment.
    """
    import copy
    rng = np.random.default_rng(seed)
    q = copy.deepcopy(tp)
    for name in ("dTheta_or", "Hgr", "R", "tau_o", "tau_w", "x", "y"):
        v = getattr(q, name)
        setattr(q, name, float(v * (1.0 + rng.normal(0, rel))))
    return q


def train_pinn(traj, tp, n_obs_to=180, n_obs_hs=6, lam_p=1.0, lam_d=1.0,
               lam_bc=0.1, n_coll=2000, epochs=6000, lr=2e-3, seed=0,
               hidden=64, layers=5, noise_K=0.5, verbose=False, device="cpu",
               param_mismatch=0.10):
    """Train a PINN to reconstruct the hot-spot temperature trajectory.

    Parameters
    ----------
    traj : dict with keys t_min, K, Theta_A, dTO, dH (ground truth)
    tp   : ThermalParams
    n_obs_to : number of noisy top-oil observations available
    n_obs_hs : number of (scarce) direct hot-spot labels available
    lam_p    : physics-loss weight; lam_p = 0 gives the purely data-driven model
    noise_K  : standard deviation of temperature sensor noise [K]

    Returns dict with the trained model, loss history and reconstruction error.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # the physics loss uses the ASSUMED (datasheet) parameters, which differ
    # from the true parameters that generated the trajectory
    tp_true = tp
    tp = perturb_params(tp, rel=param_mismatch, seed=seed + 991) \
        if param_mismatch > 0 else tp

    t = traj["t_min"]
    T = float(t[-1])
    tn = t / T                                            # normalised time
    Ta, K = traj["Theta_A"], traj["K"]
    dTO_true, dH_true = traj["dTO"], traj["dH"]

    tg = torch.tensor(tn, dtype=torch.float32)
    Kg = torch.tensor(K, dtype=torch.float32)
    Tag = torch.tensor(Ta, dtype=torch.float32)

    # ---- observation sets ----
    i_to = np.sort(rng.choice(len(t), size=min(n_obs_to, len(t)), replace=False))
    obs_to = dTO_true[i_to] + rng.normal(0, noise_K, len(i_to))
    t_to = torch.tensor(tn[i_to], dtype=torch.float32).view(-1, 1)
    y_to = torch.tensor(obs_to, dtype=torch.float32).view(-1, 1)

    if n_obs_hs > 0:
        i_hs = np.sort(rng.choice(len(t), size=n_obs_hs, replace=False))
        obs_hs = dH_true[i_hs] + rng.normal(0, noise_K, n_obs_hs)
        t_hs = torch.tensor(tn[i_hs], dtype=torch.float32).view(-1, 1)
        y_hs = torch.tensor(obs_hs, dtype=torch.float32).view(-1, 1)
    else:
        t_hs = y_hs = None

    t_coll = torch.rand(n_coll, 1, requires_grad=True)

    model = PINN(hidden=hidden, layers=layers, max_freq=max(8.0, T / 240.0)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 5),
                                            gamma=0.5)

    # IEC 60076-7 constants, expressed on the normalised time axis
    tau_o = tp.tau_o / T
    tau_w = tp.tau_w / T
    hist = []

    for ep in range(epochs):
        opt.zero_grad()

        # ---- data loss ----
        out_to = model(t_to)
        L_data = F.mse_loss(out_to[:, 0:1], y_to)
        if t_hs is not None:
            out_hs = model(t_hs)
            L_data = L_data + F.mse_loss(out_hs[:, 1:2], y_hs)

        # ---- physics loss (autodiff residual of the IEC thermal ODEs) ----
        if lam_p > 0:
            out_c = model(t_coll)
            dTO_c, dH_c = out_c[:, 0:1], out_c[:, 1:2]
            g = torch.ones_like(dTO_c)
            d_dTO = torch.autograd.grad(dTO_c, t_coll, grad_outputs=g,
                                        create_graph=True)[0]
            d_dH = torch.autograd.grad(dH_c, t_coll, grad_outputs=g,
                                       create_graph=True)[0]
            Kc = _interp_torch(t_coll, tg, Kg)
            to_ss = tp.dTheta_or * ((1 + tp.R * Kc ** 2) / (1 + tp.R)) ** tp.x
            hs_ss = tp.Hgr * torch.clamp(Kc, min=1e-3) ** tp.y
            r1 = tau_o * d_dTO - (to_ss - dTO_c)
            r2 = tau_w * d_dH - (hs_ss - dH_c)
            L_phys = (r1 ** 2).mean() / 100.0 + (r2 ** 2).mean() / 100.0
        else:
            L_phys = torch.tensor(0.0)

        # ---- initial condition ----
        t0 = torch.zeros(1, 1)
        out0 = model(t0)
        L_bc = ((out0[0, 0] - float(dTO_true[0])) ** 2
                + (out0[0, 1] - float(dH_true[0])) ** 2)

        loss = lam_d * L_data + lam_p * L_phys + lam_bc * L_bc
        loss.backward()
        opt.step()
        sched.step()

        if ep % 50 == 0 or ep == epochs - 1:
            hist.append((ep, float(loss.detach()), float(L_data.detach()), float(L_phys.detach())))
        if verbose and ep % 1000 == 0:
            print(f"      ep {ep:5d}  L={float(loss):.4f} "
                  f"Ld={float(L_data):.4f} Lp={float(L_phys):.4f}", flush=True)

    # ---- evaluation on the full (unseen) hot-spot trajectory ----
    with torch.no_grad():
        pred = model(torch.tensor(tn, dtype=torch.float32).view(-1, 1)).numpy()
    Th_pred = Ta + pred[:, 0] + pred[:, 1]
    Th_true = Ta + dTO_true + dH_true
    rmse = float(np.sqrt(np.mean((Th_pred - Th_true) ** 2)))
    mae = float(np.mean(np.abs(Th_pred - Th_true)))
    r2 = float(1 - np.sum((Th_pred - Th_true) ** 2)
               / np.sum((Th_true - Th_true.mean()) ** 2))

    return dict(model=model, hist=hist, Th_pred=Th_pred, Th_true=Th_true,
                dTO_pred=pred[:, 0], dH_pred=pred[:, 1],
                rmse=rmse, mae=mae, r2=r2, i_to=i_to,
                n_obs_hs=n_obs_hs, lam_p=lam_p,
                param_mismatch=param_mismatch)


def train_pinn_multiphase(traj, tp, n_phases=6, lam_p=1.0, lam_couple=0.0,
                          n_obs_hs=6, epochs=5000, lr=2e-3, seed=0, n_coll=2000,
                          noise_K=0.5, param_mismatch=0.10):
    """Multi-phase PINN with an optional inter-phase thermal coupling term.

    The coupling loss enforces that all phase windings share a single oil
    volume, i.e. that the per-phase hot-spot gradients are consistent with a
    common top-oil rise driven by the mean per-phase loss.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    tp = perturb_params(tp, rel=param_mismatch, seed=seed + 991) \
        if param_mismatch > 0 else tp
    t = traj["t_min"]; T = float(t[-1]); tn = t / T
    Ta = traj["Theta_A"]; Kph = traj["K_phase"]
    dTO_true = traj["dTO"]; dH_true = traj["dH_phase"]

    tg = torch.tensor(tn, dtype=torch.float32)
    Kg = torch.tensor(Kph, dtype=torch.float32)

    n_out = 1 + n_phases
    model = PINN(hidden=64, layers=5, n_out=1)
    model.net[-1] = nn.Linear(64, n_out)
    nn.init.xavier_normal_(model.net[-1].weight); nn.init.zeros_(model.net[-1].bias)
    model.scale = nn.Parameter(torch.tensor([50.0] + [30.0] * n_phases),
                               requires_grad=False)

    i_to = np.sort(rng.choice(len(t), size=180, replace=False))
    y_to = torch.tensor(dTO_true[i_to] + rng.normal(0, noise_K, len(i_to)),
                        dtype=torch.float32).view(-1, 1)
    t_to = torch.tensor(tn[i_to], dtype=torch.float32).view(-1, 1)

    i_hs = np.sort(rng.choice(len(t), size=n_obs_hs, replace=False))
    y_hs = torch.tensor(dH_true[i_hs] + rng.normal(0, noise_K, dH_true[i_hs].shape),
                        dtype=torch.float32)
    t_hs = torch.tensor(tn[i_hs], dtype=torch.float32).view(-1, 1)

    t_coll = torch.rand(n_coll, 1, requires_grad=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 5), gamma=0.5)
    tau_o, tau_w = tp.tau_o / T, tp.tau_w / T

    for ep in range(epochs):
        opt.zero_grad()
        o_to = model(t_to); o_hs = model(t_hs)
        L_data = F.mse_loss(o_to[:, 0:1], y_to) + F.mse_loss(o_hs[:, 1:], y_hs)

        oc = model(t_coll)
        g = torch.ones_like(oc[:, 0:1])
        d_dTO = torch.autograd.grad(oc[:, 0:1], t_coll, grad_outputs=g,
                                    create_graph=True)[0]
        Kc = torch.stack([_interp_torch(t_coll, tg, Kg[:, i]).squeeze(-1)
                          for i in range(n_phases)], dim=1)
        loss_term = ((1 + tp.R * Kc ** 2) / (1 + tp.R)).mean(dim=1, keepdim=True)
        to_ss = tp.dTheta_or * loss_term ** tp.x
        r1 = tau_o * d_dTO - (to_ss - oc[:, 0:1])
        L_phys = (r1 ** 2).mean() / 100.0
        for i in range(n_phases):
            d_i = torch.autograd.grad(oc[:, i + 1:i + 2], t_coll,
                                      grad_outputs=g, create_graph=True)[0]
            hs_ss = tp.Hgr * torch.clamp(Kc[:, i:i + 1], min=1e-3) ** tp.y
            r2 = tau_w * d_i - (hs_ss - oc[:, i + 1:i + 2])
            L_phys = L_phys + (r2 ** 2).mean() / 100.0 / n_phases

        # inter-phase coupling: deviation of each phase gradient from the mean
        # must match the loss-driven expectation from the shared oil volume
        L_cpl = torch.tensor(0.0)
        if lam_couple > 0:
            hs_pred = oc[:, 1:]
            hs_mean = hs_pred.mean(dim=1, keepdim=True)
            expect = tp.Hgr * (torch.clamp(Kc, min=1e-3) ** tp.y
                               - (torch.clamp(Kc, min=1e-3) ** tp.y).mean(dim=1, keepdim=True))
            L_cpl = ((hs_pred - hs_mean - expect) ** 2).mean() / 100.0

        loss = L_data + lam_p * L_phys + lam_couple * L_cpl
        loss.backward(); opt.step(); sched.step()

    with torch.no_grad():
        pr = model(torch.tensor(tn, dtype=torch.float32).view(-1, 1)).numpy()
    Th_pred = Ta[:, None] + pr[:, 0:1] + pr[:, 1:]
    Th_true = Ta[:, None] + dTO_true[:, None] + dH_true
    rmse = float(np.sqrt(np.mean((Th_pred - Th_true) ** 2)))
    # per-phase imbalance reconstruction error
    imb_pred = Th_pred - Th_pred.mean(axis=1, keepdims=True)
    imb_true = Th_true - Th_true.mean(axis=1, keepdims=True)
    imb_rmse = float(np.sqrt(np.mean((imb_pred - imb_true) ** 2)))
    return dict(rmse=rmse, imbalance_rmse=imb_rmse, Th_pred=Th_pred,
                Th_true=Th_true, lam_couple=lam_couple)


# ==========================================================================
# 2. SEQUENCE CLASSIFIERS (TEACHER / STUDENT / BASELINES)
# ==========================================================================

class GRUClassifier(nn.Module):
    """GRU sequence classifier; optionally bidirectional (teacher configuration)."""

    def __init__(self, d_in, n_cls, hidden=32, layers=1, bidir=False, feat_dim=32):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, num_layers=layers, batch_first=True,
                          bidirectional=bidir)
        d = hidden * (2 if bidir else 1)
        self.proj = nn.Linear(d, feat_dim)     # common feature space for distillation
        self.head = nn.Linear(feat_dim, n_cls)

    def forward(self, x, return_feat=False):
        o, _ = self.gru(x)
        h = torch.tanh(self.proj(o[:, -1]))
        z = self.head(h)
        return (z, h) if return_feat else z


class CNNLSTM(nn.Module):
    """Convolutional front end followed by an LSTM (state-of-the-art baseline)."""

    def __init__(self, d_in, n_cls, ch=64, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_in, ch, 3, padding=1), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU())
        self.lstm = nn.LSTM(ch, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_cls)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        o, _ = self.lstm(h)
        return self.head(o[:, -1])


def train_torch_classifier(model, Xtr, ytr, Xva, yva, epochs=60, lr=3e-3,
                           bs=64, seed=0, weight_decay=1e-5, verbose=False):
    """Standard supervised training loop with best-validation checkpointing."""
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva); yva_t = torch.tensor(yva, dtype=torch.long)
    n = len(Xtr_t)
    best, best_state, hist = 0.0, None, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            z = model(Xtr_t[idx])
            loss = F.cross_entropy(z, ytr_t[idx])
            loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            acc = float((model(Xva_t).argmax(1) == yva_t).float().mean())
        hist.append((ep, tot / n, acc))
        if acc > best:
            best = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and ep % 10 == 0:
            print(f"      ep {ep:3d} loss={tot/n:.4f} val={acc:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best


def distill(student, teacher_gru, Xtr, ytr, teacher_soft, Xva, yva,
            T=4.0, alpha=0.3, beta=0.6, gamma=0.1, epochs=60, lr=3e-3, bs=64,
            seed=0, verbose=False):
    """Knowledge distillation from the Bi-GRU teacher to the Uni-GRU student.

    L = alpha*CE(y, student) + beta*T^2*KL(teacher||student) + gamma*MSE(h_t, h_s)
    """
    set_seed(seed)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva); yva_t = torch.tensor(yva, dtype=torch.long)
    teacher_gru.eval()
    with torch.no_grad():
        zt_all, ht_all = teacher_gru(Xtr_t, return_feat=True)
    n = len(Xtr_t)
    best, best_state, hist = 0.0, None, []
    for ep in range(epochs):
        student.train()
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            zs, hs = student(Xtr_t[idx], return_feat=True)
            L_ce = F.cross_entropy(zs, ytr_t[idx])
            L_kl = F.kl_div(F.log_softmax(zs / T, dim=1),
                            F.softmax(zt_all[idx] / T, dim=1),
                            reduction="batchmean") * (T ** 2)
            L_mse = F.mse_loss(hs, ht_all[idx])
            loss = alpha * L_ce + beta * L_kl + gamma * L_mse
            loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        student.eval()
        with torch.no_grad():
            acc = float((student(Xva_t).argmax(1) == yva_t).float().mean())
        hist.append((ep, tot / n, acc))
        if acc > best:
            best = acc
            best_state = {k: v.clone() for k, v in student.state_dict().items()}
        if verbose and ep % 10 == 0:
            print(f"      ep {ep:3d} L={tot/n:.4f} val={acc:.4f}", flush=True)
    if best_state is not None:
        student.load_state_dict(best_state)
    return student, hist, best


class Ensemble:
    """Averaging ensemble of a Random Forest and a recurrent classifier."""

    def __init__(self, rf, gru, w_rf=0.5, n_classes=None):
        self.rf, self.gru, self.w_rf = rf, gru, w_rf
        self.n_classes = n_classes or gru.head.out_features

    def predict_proba(self, Xseq, Xflat):
        with torch.no_grad():
            p_gru = F.softmax(self.gru(torch.tensor(Xseq)), dim=1).numpy()
        p_rf_partial = self.rf.predict_proba(Xflat)
        # a Random Forest fitted on a subsample may not have observed every
        # class; scatter its columns back into the full class set
        p_rf = np.zeros((len(Xflat), self.n_classes))
        p_rf[:, self.rf.classes_.astype(int)] = p_rf_partial
        return self.w_rf * p_rf + (1 - self.w_rf) * p_gru

    def predict(self, Xseq, Xflat):
        return self.predict_proba(Xseq, Xflat).argmax(1)


# ==========================================================================
# 3. VARIATIONAL AUTOENCODER FOR SENSOR-INTEGRITY VALIDATION
# ==========================================================================

class VAE(nn.Module):
    """Compact VAE: d -> 32 -> 16 -> 8 (latent) -> 16 -> 32 -> d."""

    def __init__(self, d_in, latent=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU())
        self.mu = nn.Linear(16, latent)
        self.lv = nn.Linear(16, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 16), nn.ReLU(),
                                 nn.Linear(16, 32), nn.ReLU(),
                                 nn.Linear(32, d_in))

    def forward(self, x):
        h = self.enc(x)
        mu, lv = self.mu(h), torch.clamp(self.lv(h), -8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(z), mu, lv

    def score(self, x, beta_kl=1e-3):
        """Anomaly score = reconstruction error + beta * KL divergence."""
        with torch.no_grad():
            xh, mu, lv = self.forward(x)
            rec = ((x - xh) ** 2).sum(dim=1)
            kl = -0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(dim=1)
            return (rec + beta_kl * kl).numpy()


def train_vae(Xn, epochs=200, lr=2e-3, bs=128, beta_kl=1e-3, seed=0, latent=8):
    """Train the VAE on normal-operation feature vectors only."""
    set_seed(seed)
    vae = VAE(Xn.shape[1], latent=latent)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    X = torch.tensor(Xn, dtype=torch.float32)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            xh, mu, lv = vae(X[idx])
            rec = ((X[idx] - xh) ** 2).sum(dim=1).mean()
            kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(dim=1)).mean()
            (rec + beta_kl * kl).backward()
            opt.step()
    vae.eval()
    return vae


# ==========================================================================
# 4. WEIBULL REMAINING-USEFUL-LIFE ESTIMATOR
# ==========================================================================

class WeibullRUL:
    """Two-parameter Weibull RUL model fitted to health-index trajectories.

    The health index is interpreted as an unreliability estimate, i.e.
    HI(t) ~ F(t; lambda, beta) = 1 - exp(-(t/lambda)^beta), so that
    RUL(t) = lambda * (-ln(1 - HI_eol))^(1/beta) - t
    with HI_eol the end-of-life health-index threshold.
    """

    def __init__(self, hi_eol=0.85, min_points=12):
        self.hi_eol = hi_eol
        self.min_points = min_points

    # physically admissible parameter ranges for insulation degradation
    BETA_BOUNDS = (0.6, 8.0)

    @staticmethod
    def _fit_mle(t, hi, t_ref=None):
        """Fit the two-parameter Weibull curve to a health-index history.

        The classical log-log linearisation is used only for initialisation: it
        is badly conditioned when the health index spans a narrow range, and can
        return shape parameters near zero that make the extrapolated end-of-life
        time diverge. The parameters are therefore refined by bounded nonlinear
        least squares on the untransformed curve.
        """
        m = (hi > 1e-4) & (hi < 0.999) & (t > 0)
        if m.sum() < 8:
            return None
        tm, hm = t[m], hi[m]
        t_ref = float(tm[-1]) if t_ref is None else t_ref

        # --- initialisation by linear regression in Weibull coordinates ---
        try:
            x = np.log(tm)
            yv = np.log(-np.log(1.0 - hm))
            A = np.vstack([x, np.ones_like(x)]).T
            b0, c0 = np.linalg.lstsq(A, yv, rcond=None)[0]
            b0 = float(np.clip(b0, *WeibullRUL.BETA_BOUNDS))
            l0 = float(np.clip(np.exp(-c0 / b0), 0.2 * t_ref, 20.0 * t_ref))
        except Exception:
            b0, l0 = 2.0, 2.0 * t_ref

        def model(tt, beta, lam):
            return 1.0 - np.exp(-np.power(np.maximum(tt, 1e-9) / lam, beta))

        try:
            popt, _ = optimize.curve_fit(
                model, tm, hm, p0=[b0, l0],
                bounds=([WeibullRUL.BETA_BOUNDS[0], 0.2 * t_ref],
                        [WeibullRUL.BETA_BOUNDS[1], 20.0 * t_ref]),
                maxfev=4000)
            beta, lam = float(popt[0]), float(popt[1])
        except Exception:
            beta, lam = b0, l0
        if not (np.isfinite(beta) and np.isfinite(lam)) or lam <= 0:
            return None
        return beta, lam

    @staticmethod
    def _damage(h, h0):
        """Normalised damage fraction u in (0,1) from a health index history.

        A new asset has a non-zero health index because the nominal harmonic,
        vibration and acoustic levels are not zero. The Weibull cumulative
        distribution vanishes at t = 0, so the baseline must be removed before
        the health index can be interpreted as an unreliability.
        """
        return np.clip((h - h0) / max(1.0 - h0, 1e-6), 1e-4, 0.999)

    def fit_predict(self, t_days, hi, t_now_idx):
        """Fit on history up to `t_now_idx` and extrapolate to end of life."""
        t = t_days[:t_now_idx + 1]
        h = hi[:t_now_idx + 1]
        if len(t) < self.min_points:
            return None
        h0 = float(np.percentile(h, 5))          # as-new baseline, causal estimate
        u = self._damage(h, h0)
        u_eol = float(np.clip((self.hi_eol - h0) / max(1.0 - h0, 1e-6),
                              1e-3, 0.999))
        fit = self._fit_mle(t, u, t_ref=float(t_days[t_now_idx]))
        if fit is None:
            return None
        beta, lam = fit
        t_eol = lam * (-np.log(1.0 - u_eol)) ** (1.0 / beta)
        rul = t_eol - t_days[t_now_idx]
        # reject physically implausible extrapolations rather than reporting them
        if not np.isfinite(rul) or t_eol > 6.0 * t_days[t_now_idx]:
            return None
        return dict(beta=beta, lam=lam, t_eol=float(t_eol),
                    h0=h0, u_eol=u_eol, rul=float(max(rul, 0.0)))

    @staticmethod
    def bootstrap_ci(t_days, hi, t_now_idx, hi_eol=0.85, n_boot=60, seed=0):
        """Bootstrap confidence interval for the RUL point estimate."""
        rng = np.random.default_rng(seed)
        t = t_days[:t_now_idx + 1]; h = hi[:t_now_idx + 1]
        h0 = float(np.percentile(h, 5))
        u = WeibullRUL._damage(h, h0)
        u_eol = float(np.clip((hi_eol - h0) / max(1 - h0, 1e-6), 1e-3, 0.999))
        n = len(t); out = []
        for _ in range(n_boot):
            idx = np.sort(rng.choice(n, n, replace=True))
            f = WeibullRUL._fit_mle(t[idx], u[idx], t_ref=float(t_days[t_now_idx]))
            if f is None:
                continue
            b, l = f
            v = l * (-np.log(1 - u_eol)) ** (1 / b) - t_days[t_now_idx]
            if np.isfinite(v) and 0 <= v <= 6 * t_days[t_now_idx]:
                out.append(v)
        if len(out) < 8:
            return None
        return float(np.percentile(out, 5)), float(np.percentile(out, 95))


# ==========================================================================
# 5. EDGE DEPLOYMENT COST MODEL
# ==========================================================================

def count_params_bytes(module, dtype_bytes=4):
    n = sum(p.numel() for p in module.parameters())
    return n, n * dtype_bytes


def gru_flops(d_in, hidden, seq_len, bidir=False):
    """Multiply-accumulate count for a single-layer GRU forward pass."""
    dirs = 2 if bidir else 1
    per_step = 3 * (d_in * hidden + hidden * hidden)
    return 2 * dirs * seq_len * per_step          # 2 FLOPs per MAC


def linear_flops(d_in, d_out):
    return 2 * d_in * d_out


def rf_flops(n_trees, max_depth):
    """Comparison count for a Random Forest traversal (one per tree per level)."""
    return n_trees * max_depth


def fake_quantize_int8(model):
    """Simulate post-training INT8 weight quantisation (per-tensor, symmetric).

    Weights are quantised to 8-bit integers and de-quantised in place, so the
    accuracy impact of the quantisation error can be measured directly.
    """
    import copy
    q = copy.deepcopy(model)
    with torch.no_grad():
        for p in q.parameters():
            s = p.abs().max() / 127.0
            if float(s) == 0:
                continue
            p.copy_(torch.round(p / s).clamp(-127, 127) * s)
    return q


def project_latency(flops, throughput_mflops):
    """Project inference latency from FLOP count and sustained throughput.

    This is an analytical projection, not a hardware measurement.
    """
    return flops / (throughput_mflops * 1e6)


class LossOfLifeRUL:
    """Physics-informed RUL estimator fusing thermal ageing with condition data.

    Two complementary sources of information are combined.

    1. The IEC 60076-7 relative ageing rate V(t) = 2^((Theta_H - 98)/6),
       obtained from the estimated hot-spot temperature, gives the *rate* at
       which insulation life is consumed. Integrating it yields the accumulated
       equivalent ageing L(t), which is directly observable.

    2. The health index gives the *fraction* of life already consumed, through a
       monotone relation HI(d) calibrated once on a disjoint set of units.

    Inverting the calibrated relation gives the damage fraction d(t), and the
    per-unit endurance follows as L_end = L(t)/d(t). This is essential because
    endurance varies over orders of magnitude across a fleet, so a single
    population-level prior is not usable. Remaining useful life is then the
    remaining ageing budget divided by the projected future ageing rate.
    """

    def __init__(self, hi_grid, d_grid, window_days=180.0):
        """hi_grid, d_grid: the calibrated, monotone HI-versus-damage curve."""
        self.hi_grid = np.asarray(hi_grid, dtype=float)
        self.d_grid = np.asarray(d_grid, dtype=float)
        self.window_days = float(window_days)

    @staticmethod
    def calibrate(hi_list, d_list, n_bins=24):
        """Build the median HI-versus-damage-fraction curve from calibration units."""
        hi = np.concatenate(hi_list)
        d = np.concatenate(d_list)
        m = np.isfinite(hi) & np.isfinite(d) & (d >= 0) & (d <= 1.2)
        hi, d = hi[m], d[m]
        edges = np.linspace(0, 1.0, n_bins + 1)
        dc, hc = [], []
        for i in range(n_bins):
            sel = (d >= edges[i]) & (d < edges[i + 1])
            if sel.sum() < 20:
                continue
            dc.append(0.5 * (edges[i] + edges[i + 1]))
            hc.append(float(np.median(hi[sel])))
        dc, hc = np.asarray(dc), np.asarray(hc)
        order = np.argsort(hc)
        return hc[order], dc[order]

    def damage_from_hi(self, hi_now):
        return float(np.clip(np.interp(hi_now, self.hi_grid, self.d_grid),
                             1e-3, 1.0))

    def predict(self, t_days, V_series, hi_series, t_now_idx, dt_hours):
        """Project the damage fraction forward to unity.

        Inverting the health index to a per-unit endurance (L_end = L/d) was
        found to be unusable: the estimate of d is noisy and the reciprocal
        amplifies that noise hyperbolically at exactly the early inspection
        times where a prediction is most valuable. Extrapolating the damage
        fraction itself is far better conditioned, because d is bounded in
        [0, 1] and its rate of change is estimated by ordinary least squares
        over a long observation window.
        """
        d_hat = np.clip(np.interp(hi_series[:t_now_idx + 1],
                                  self.hi_grid, self.d_grid), 1e-3, 1.0)
        t = t_days[:t_now_idx + 1]

        w = max(8, int(self.window_days / (dt_hours / 24.0)))
        lo = max(0, t_now_idx + 1 - w)
        tt = t[lo:] - t[t_now_idx]
        dd = d_hat[lo:]
        if len(tt) < 8:
            return None

        A = np.vstack([tt, np.ones_like(tt)]).T
        slope, inter = np.linalg.lstsq(A, dd, rcond=None)[0]
        slope = max(float(slope), 1e-6)          # damage does not decrease
        d_now = float(np.clip(inter, 1e-3, 0.999))

        L_now = float(np.sum(V_series[:t_now_idx + 1]) * dt_hours)
        rul = (1.0 - d_now) / slope
        return dict(rul=float(np.clip(rul, 0.0, 4000.0)), d_now=d_now,
                    d_rate=slope, L_now=L_now)
