#!/usr/bin/env python3
"""
make_equations.py -- Render every displayed equation of the manuscript as a
transparent, high-resolution PNG for embedding in the Word document.

Matplotlib's mathtext engine is used so that no external LaTeX installation is
required. Each equation is written to eqs/eqNN.png together with a JSON index
giving its rendered width and height in points.

Author: Md Tabrez
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "eqs"
DPI = 420

EQUATIONS = {
    # ---- Section III: problem formulation ----
    1: r"$\mathcal{S}(t)=\{\Theta_H^{(i)}(t),\,\mathrm{HI}^{(i)}(t),\,"
       r"\mathrm{RUL}^{(i)}(t)\}_{i=1}^{N_a}$",
    2: r"$\min_{\theta,\phi}\ \ \mathrm{E}\left[\,c_f\,\mathrm{I}[\hat{y}\neq y]"
       r"\;+\;c_r\,|\widehat{\mathrm{RUL}}-\mathrm{RUL}|\,\right]$",
    3: r"$\mathrm{s.t.}\quad \mathcal{T}_{inf}(\phi)\leq\mathcal{T}_{max},"
       r"\quad \mathcal{M}(\phi)\leq\mathcal{M}_{max},"
       r"\quad \mathcal{B}(\phi)\leq\mathcal{B}_{max}$",

    # ---- Section IV-A: sensor placement ----
    4: r"$\min_{S\subseteq V}|S|\ \ \mathrm{s.t.}\ \ \forall v\in V,\ \ "
       r"v\in S\ \vee\ \exists u\in S:\ d(u,v)\leq r$",

    # ---- Section IV-B: IEC 60076-7 thermal model ----
    5: r"$\tau_{o}\frac{d\,\Delta\Theta_{TO}}{dt}"
       r"=\Delta\Theta_{or}\left(\frac{1+RK^{2}}{1+R}\right)^{x}-\Delta\Theta_{TO}$",
    6: r"$\tau_{w}\frac{d\,\Delta\Theta_{H}}{dt}=H\cdot g_{r}K^{\,y}-\Delta\Theta_{H}$",
    7: r"$\Theta_{H}(t)=\Theta_{A}(t)+\Delta\Theta_{TO}(t)+\Delta\Theta_{H}(t)$",
    8: r"$\tau_{o}\frac{d\,\Delta\Theta_{TO}}{dt}=\Delta\Theta_{or}"
       r"\left(\frac{1}{N_{ph}}\sum_{i=1}^{N_{ph}}\frac{1+RK_{i}^{2}}{1+R}\right)^{x}"
       r"-\Delta\Theta_{TO}$",
    9: r"$\tau_{w}\frac{d\,\Delta\Theta_{H,i}}{dt}=H\cdot g_{r}K_{i}^{\,y}"
       r"-\Delta\Theta_{H,i},\qquad i=1,\dots,N_{ph}$",
    10: r"$V(t)=2^{\,(\Theta_{H}(t)-98)/6},\qquad "
        r"L=\int_{0}^{T}V(t)\,dt$",

    # ---- Section IV-B: PINN ----
    11: r"$\gamma(t)=\left[\,t,\ \sin(2\pi k_{1}t),\cos(2\pi k_{1}t),\ \dots,\ "
        r"\sin(2\pi k_{M}t),\cos(2\pi k_{M}t)\,\right]$",
    12: r"$\left[\widehat{\Delta\Theta}_{TO}(t;\theta),\ "
        r"\widehat{\Delta\Theta}_{H}(t;\theta)\right]"
        r"=\mathcal{N}_{\theta}\left(\gamma(t)\right)$",
    13: r"$\mathcal{L}(\theta)=\lambda_{d}\mathcal{L}_{data}"
        r"+\lambda_{p}\mathcal{L}_{phys}+\lambda_{bc}\mathcal{L}_{bc}$",
    14: r"$\mathcal{L}_{data}=\frac{1}{N_{d}}\sum_{j=1}^{N_{d}}"
        r"\left(\widehat{\Delta\Theta}_{TO}(t_{j};\theta)-\Delta\Theta_{TO}^{obs}(t_{j})\right)^{2}"
        r"+\frac{1}{N_{hs}}\sum_{l=1}^{N_{hs}}"
        r"\left(\widehat{\Delta\Theta}_{H}(t_{l};\theta)-\Delta\Theta_{H}^{obs}(t_{l})\right)^{2}$",
    15: r"$r_{1}(t_{k})=\tilde{\tau}_{o}\frac{\partial\widehat{\Delta\Theta}_{TO}}"
        r"{\partial t}_{\,t=t_{k}}-\left[\Delta\tilde{\Theta}_{or}"
        r"\left(\frac{1+\tilde{R}K^{2}(t_{k})}{1+\tilde{R}}\right)^{\tilde{x}}"
        r"-\widehat{\Delta\Theta}_{TO}(t_{k})\right]$",
    16: r"$r_{2}(t_{k})=\tilde{\tau}_{w}\frac{\partial\widehat{\Delta\Theta}_{H}}"
        r"{\partial t}_{\,t=t_{k}}-\left[\widetilde{H\cdot g_{r}}\,"
        r"K^{\tilde{y}}(t_{k})-\widehat{\Delta\Theta}_{H}(t_{k})\right]$",
    17: r"$\mathcal{L}_{phys}=\frac{1}{N_{c}}\sum_{k=1}^{N_{c}}"
        r"\left(r_{1}^{2}(t_{k})+r_{2}^{2}(t_{k})\right)$",
    18: r"$\mathcal{L}_{bc}=\left(\widehat{\Delta\Theta}_{TO}(0)-\Delta\Theta_{TO,0}\right)^{2}"
        r"+\left(\widehat{\Delta\Theta}_{H}(0)-\Delta\Theta_{H,0}\right)^{2}$",
    19: r"$\mathcal{L}_{couple}=\frac{1}{N_{c}N_{ph}}\sum_{k}\sum_{i}"
        r"\left(\widehat{\Delta\Theta}_{H,i}(t_{k})-\overline{\widehat{\Delta\Theta}}_{H}(t_{k})"
        r"-\delta_{i}(t_{k})\right)^{2}$",
    20: r"$\delta_{i}(t)=\widetilde{H\cdot g_{r}}\left(K_{i}^{\tilde{y}}(t)"
        r"-\frac{1}{N_{ph}}\sum_{j}K_{j}^{\tilde{y}}(t)\right)$",

    # ---- Section IV-C: HI, RUL, distillation ----
    21: r"$\mathrm{HI}(t)=w_{1}\left[\frac{\Theta_{H}(t)-\Theta_{H,nom}}"
        r"{\Theta_{H,max}-\Theta_{H,nom}}\right]_{+}"
        r"+w_{2}\frac{\mathrm{THD}_{I}(t)}{\mathrm{THD}_{I,lim}}"
        r"+w_{3}\left[\frac{\sigma_{v}(t)-\sigma_{v,nom}}{\sigma_{v,nom}}\right]_{+}"
        r"+w_{4}\frac{\mathrm{AE}_{rms}(t)}{\mathrm{AE}_{lim}}$",
    22: r"$F(t;\lambda,\beta)=1-\exp\left[-\left(\frac{t}{\lambda}\right)^{\beta}\right]$",
    23: r"$\ln\left[-\ln(1-\mathrm{HI}(t))\right]=\beta\ln t-\beta\ln\lambda$",
    24: r"$\widehat{\mathrm{RUL}}(t)=\lambda\left[-\ln(1-\mathrm{HI}_{eol})\right]^{1/\beta}-t$",
    25: r"$\mathcal{L}_{distill}=\alpha\,\mathcal{L}_{CE}(y,z_{s})"
        r"+\beta T^{2}\,\mathcal{L}_{KL}\left(\sigma(z_{t}/T)\,\|\,\sigma(z_{s}/T)\right)"
        r"+\gamma\,\mathcal{L}_{MSE}(h_{t},h_{s})$",
    26: r"$p_{ens}=w_{rf}\,p_{RF}+(1-w_{rf})\,p_{GRU},\qquad "
        r"w_{rf}=\arg\max_{w}\ \mathrm{Acc}_{val}(w)$",
    27: r"$W_{q}=s\cdot\mathrm{clip}\left(\mathrm{round}(W/s),-127,127\right),"
        r"\qquad s=\frac{\max|W|}{127}$",

    # ---- Section IV-E: VAE security ----
    28: r"$\mathcal{L}_{VAE}=\|\mathbf{f}-\hat{\mathbf{f}}\|_{2}^{2}"
        r"+\beta_{KL}\,D_{KL}\left(q_{\varphi}(z|\mathbf{f})\,\|\,p(z)\right)$",
    29: r"$D_{KL}=-\frac{1}{2}\sum_{j=1}^{d_{z}}"
        r"\left(1+\log\sigma_{z,j}^{2}-\mu_{z,j}^{2}-\sigma_{z,j}^{2}\right)$",
    30: r"$v(\mathbf{f})=\mathrm{I}\left[\,s(\mathbf{f})\leq\mu_{s}+k\sigma_{s}\,\right],"
        r"\qquad k=3$",

    # ---- Section V: bandwidth ----
    31: r"$\rho=1-\frac{(1-p_{a})(b_{f}+b_{h})+p_{a}(b_{r}+b_{h})}{b_{r}}$",
}


def render(idx, tex):
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, tex, fontsize=13, color="black")
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    w, h = bb.width / fig.dpi, bb.height / fig.dpi
    fig.set_size_inches(w * 1.03 + 0.04, h * 1.03 + 0.04)
    path = os.path.join(OUT, f"eq{idx:02d}.png")
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight",
                pad_inches=0.012)
    plt.close(fig)
    from PIL import Image
    im = Image.open(path)
    return dict(path=path, px_w=im.width, px_h=im.height,
                pt_w=im.width / DPI * 72.0, pt_h=im.height / DPI * 72.0)


def main():
    os.makedirs(OUT, exist_ok=True)
    index = {}
    for i, tex in EQUATIONS.items():
        try:
            index[str(i)] = render(i, tex)
        except Exception as e:
            print(f"  !! eq {i}: {e}")
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"  rendered {len(index)} equations -> {OUT}/")


if __name__ == "__main__":
    main()
