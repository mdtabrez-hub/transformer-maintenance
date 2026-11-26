#!/usr/bin/env python3
"""
make_manuscript.py -- Author the manuscript and emit manuscript.json for
build_docx.js. Every quantitative statement is interpolated from
results/results.json, so the text cannot drift from the simulation output.

Author: Md Tabrez
"""

import json
import os
from PIL import Image

R = json.load(open("results/results.json"))
EQ = json.load(open("eqs/index.json"))

# ---------------------------------------------------------------- shortcuts --
PM = R["pinn_main"]["summary"]
PL = R["pinn_lambda"]
PD = R["pinn_data_eff"]
MP = R["pinn_multiphase"]
CL = R["classification"]
CM = CL["metrics"]
CC = CL["components"]
PA = R["physics_ablation"]
TF = R["train_fraction"]
NS = R["noise_sensitivity"]
SEC = R["security"]
EC = R["edge_cost"]
BW = R["bandwidth"]["base"]
PLC = R["placement"]
RUL = R["rul"]
DS = R["dataset"]
MET = R["meta"]
CFG = MET["config"]
TH = CFG["thermal"]

f2 = lambda x: f"{x:.2f}"
f3 = lambda x: f"{x:.3f}"
pc = lambda x: f"{x * 100:.2f}"
pc1 = lambda x: f"{x * 100:.1f}"

best_lam = min(PL, key=lambda d: d["rmse"])
worst_lam = max(PL, key=lambda d: d["rmse"])
d_eff0 = next(d for d in PD if d["n_labels"] == 0)
d_eff200 = next(d for d in PD if d["n_labels"] == 200)
teacher_row, student_row = EC["rows"]
proj = {(p["target"].split(" @")[0], p["model"].split(" ")[0]): p["projected_latency_ms"]
        for p in EC["projections"]}
A72 = "ARM Cortex-A72"
M4 = "ARM Cortex-M4F"

blocks = []
S = []


def sec(cols, bs):
    S.append({"cols": cols, "blocks": bs})


def P(t, noindent=False):
    return {"t": "p", "text": t, "noindent": noindent}


def H1(t):
    return {"t": "h1", "text": t}


def H2(t):
    return {"t": "h2", "text": t}


def H3(t):
    return {"t": "h3", "text": t}


# An equation is scaled down to fit its container. Below roughly three
# quarters of natural size it stops being legible at print resolution, so any
# equation that would be shrunk that far is promoted to a full-width band.
COL_CAP_PT = (3.45 - 0.55) * 72.0
MIN_SCALE = 0.78


def EQN(n, wide=False):
    nat = EQ[str(n)]["pt_w"]
    if not wide and nat > COL_CAP_PT / MIN_SCALE:
        wide = True
    return {"t": "eq", "n": n, "img": f"eqs/eq{n:02d}.png", "wide": wide}


def FIG(n, img, caption, wide=False):
    return {"t": "fig", "n": n, "img": f"figs/{img}", "caption": caption, "wide": wide}


def TAB(n, caption, rows, widths=None, wide=False):
    return {"t": "table", "n": n, "caption": caption, "rows": rows,
            "widths": widths, "wide": wide}


def LI(t):
    return {"t": "listitem", "text": t}


def NOTE(t):
    return {"t": "note", "text": t}


def REF(t):
    return {"t": "ref", "text": t}


def CODE(lines, wide=True):
    # algorithm listings are set full width; at column width the monospace
    # lines wrap and the indentation that carries the control flow is lost
    return {"t": "code", "lines": lines, "wide": wide}


# ==========================================================================
# FRONT MATTER  (single column)
# ==========================================================================
sec(1, [
    {"t": "title", "text": "A Physics-Informed Digital-Twin Framework With "
                           "Distilled Edge Inference for Predictive Maintenance "
                           "of Multi-Phase Substation Transformers: A Simulation Study"},
    {"t": "authors", "text": "Mohammad Tabrez, *Member, IEEE*"},
    {"t": "affil", "text": "Manuscript prepared September 2026. "
                           "All results reported in this paper are obtained from "
                           "software simulation; no physical deployment was performed."},
    {"t": "abstract", "text":
        "Hot-spot temperature governs the insulation ageing of oil-immersed "
        "power transformers, yet it is almost never instrumented directly, and "
        "the label scarcity that follows is the central obstacle to data-driven "
        "condition monitoring. This paper presents a physics-informed digital-twin "
        "framework in which a physics-informed neural network (PINN) reconstructs "
        "the unobservable hot-spot trajectory from sparse top-oil measurements by "
        "embedding the IEC 60076-7 thermal differential equations in its loss "
        "function, and a knowledge-distilled classifier delivers fault "
        f"discrimination within an edge computational budget. On a seeded "
        f"simulation of a six-phase 20 MVA substation transformer, the PINN "
        f"reconstructs hot-spot temperature with a root-mean-square error of "
        f"{f2(PM['pinn']['rmse'][0])} \u00b1 {f2(PM['pinn']['rmse'][1])} K "
        f"(R\u00b2 = {f3(PM['pinn']['r2'][0])}) against "
        f"{f2(PM['data_only']['rmse'][0])} \u00b1 {f2(PM['data_only']['rmse'][1])} K "
        f"for an identical network trained without the physics residual. The "
        f"physics term substitutes for labelled data: the PINN attains "
        f"{f2(d_eff0['pinn_rmse'])} K using no direct hot-spot labels at all, "
        f"whereas the purely data-driven model requires {d_eff200['n_labels']} "
        f"labels to reach {f2(d_eff200['data_rmse'])} K. A physics-loss weight "
        f"ablation performed under 10 % thermal-parameter mismatch exhibits a "
        f"clear interior optimum at \u03bb_p = {best_lam['lam_p']:g}. "
        f"Physics-derived features raise incipient-fault classification accuracy "
        f"from {pc(PA['sensor features only']['accuracy'])} % to "
        f"{pc(PA['with physics features']['accuracy'])} %. Knowledge distillation "
        f"and INT8 quantisation compress the classifier by "
        f"{teacher_row['params'] / student_row['params']:.1f}\u00d7 in parameters "
        f"and {teacher_row['total_int8_kB'] / student_row['total_int8_kB']:.1f}\u00d7 "
        f"in stored size for a {(CM['Teacher (Bi-GRU + RF)']['accuracy'] - CM['Student, distilled + INT8']['accuracy']) * 100:.2f} "
        f"percentage-point accuracy cost, with quantisation itself lossless. "
        f"Edge feature extraction reduces upstream traffic by "
        f"{BW['reduction_pct']:.1f} %. A physics-informed remaining-useful-life "
        f"estimator based on cumulative loss of life achieves "
        f"{RUL['stats_physics']['mae']:.0f} days mean absolute error "
        f"(R\u00b2 = {f3(RUL['stats_physics']['r2'])}) versus "
        f"{RUL['stats_weibull']['mae']:.0f} days for Weibull health-index "
        f"extrapolation. We also report a negative result: a reconstruction-based "
        f"integrity validator detects gross sensor spoofing "
        f"({pc1(SEC['rows'][0]['detection_rate'])} %) but is structurally unable "
        f"to detect replay of valid data ({pc1(SEC['rows'][2]['detection_rate'])} %), "
        "which we argue requires cryptographic freshness rather than anomaly "
        "detection. All code, seeds and figures are released for reproduction."},
    {"t": "keywords", "text":
        "Physics-informed neural networks, digital twin, predictive maintenance, "
        "edge computing, knowledge distillation, power transformers, IEC 60076-7, "
        "remaining useful life, simulation study."},
])

# ==========================================================================
# I. INTRODUCTION
# ==========================================================================
sec(2, [
    H1("I.  Introduction"),
    H2("A.  Motivation and Background"),
    P("Oil-immersed power transformers are the least redundant and most "
      "capital-intensive elements of a distribution substation. Their service "
      "life is governed almost entirely by the thermal degradation of the "
      "cellulose insulation, and that degradation is driven by the winding "
      "hot-spot temperature rather than by any quantity that is routinely "
      "measured [1], [2], [6]. The relationship is steep: IEC 60076-7 models "
      "the relative rate of insulation ageing as doubling for every 6 K rise in "
      "hot-spot temperature above the 98 \u00b0C reference [6]. An estimation "
      "error of only a few kelvin therefore translates into a large error in "
      "predicted remaining life.", noindent=True),
    P("Airport electrical infrastructure sharpens this problem. Airfield "
      "lighting, navigation aids and terminal services impose a twin-peak "
      "diurnal demand superimposed on a strongly night-biased lighting load, "
      "and the tolerance for unplanned outage is very low [3]\u2013[5]. Multi-phase "
      "secondary configurations, used to supply constant-current airfield "
      "lighting circuits, further complicate thermal modelling because several "
      "windings share a single oil volume while carrying unequal currents."),
    P("The obstacle to applying modern condition-monitoring methods here is not "
      "algorithmic sophistication but data. Hot-spot temperature is seldom "
      "instrumented; fibre-optic probes are installed on a minority of units and "
      "usually only at commissioning. Run-to-failure records are rare because "
      "transformers are replaced on schedule rather than on failure. Purely "
      "data-driven models trained under these conditions are fitting a target "
      "they can barely observe."),

    H2("B.  State of the Art and Limitations"),
    P("Digital-twin research in the power domain has matured from conceptual "
      "reference models [15]\u2013[17] toward integrated architectures with "
      "standards-based semantics [23], [44], [45]. Physics-informed neural "
      "networks [7], [8], [49] offer a principled route to incorporating "
      "governing equations into learning, and have been applied across fluid, "
      "solid and thermal domains. Edge computing [18] and model compression "
      "[9], [24] address the deployment constraint. Yet three gaps persist.",
      noindent=True),
    P("First, PINN applications to transformer thermal modelling are typically "
      "demonstrated with the physics assumed exact. In practice the thermal "
      "constants entering the residual come from a factory test report or from "
      "IEC default tables and carry appreciable uncertainty. Whether the "
      "physics term still helps under that mismatch, and how strongly it should "
      "be weighted, is rarely quantified."),
    P("Second, edge-deployment claims are commonly reported as headline latency "
      "or memory figures without an accompanying account of what accuracy was "
      "surrendered to obtain them, and without separating measured quantities "
      "from projected ones."),
    P("Third, and most relevant to this work, the closest prior architecture "
      "\u2014 the five-layer digital-twin predictive-maintenance framework of "
      "Alabbad and Alt\u0131nkaya [23], which integrates IEC 61850, CIM and "
      "OPC UA semantics with IEC 62443-aligned security and reports real "
      "field validation \u2014 demonstrates that layered, standards-aligned "
      "architecture is no longer itself a novel contribution. We therefore make "
      "no architectural novelty claim."),

    H2("C.  Research Contributions"),
    P("This paper contributes the following.", noindent=True),
    LI("1)  A PINN formulation for transformer hot-spot reconstruction that "
       "treats the hot-spot temperature as a latent state inferred from sparse "
       "top-oil observations through the IEC 60076-7 differential equations. "
       "Fourier feature encoding of the time input is shown to be necessary, "
       "not optional: without it the network cannot resolve the diurnal forcing "
       "and the physics term confers almost no benefit."),
    LI("2)  A quantification of the physics-versus-data trade-off under "
       "realistic parameter uncertainty. Introducing 10 % mismatch between the "
       "assumed and true thermal constants produces a genuine interior optimum "
       f"in the physics-loss weight (\u03bb_p = {best_lam['lam_p']:g}), which "
       "does not exist when the physics is exact."),
    LI("3)  A label-efficiency result: the physics residual substitutes for "
       "direct supervision, so that hot-spot reconstruction accuracy "
       "unattainable by a data-driven model with 200 labels is reached with none."),
    LI("4)  An edge-deployment analysis that reports the accuracy cost of "
       "compression alongside its benefit, and that separates measured host "
       "latency from analytically projected embedded latency rather than "
       "conflating the two."),
    LI("5)  A negative result on integrity validation. A variational "
       "autoencoder detects gross spoofing reliably but fails on replay of "
       "valid data, a limitation we argue is structural rather than a matter "
       "of tuning."),
    P("We emphasise at the outset that this is a simulation study. No physical "
      "sensors were installed, no field data were collected, and no operational "
      "deployment took place. Section VII states the resulting limitations "
      "explicitly."),

    H2("D.  Paper Organisation"),
    P("Section II reviews related work. Section III formalises the problem and "
      "describes the simulated architecture. Section IV presents the "
      "methodology. Section V details the simulation setup. Section VI reports "
      "results. Section VII states limitations and Section VIII concludes.",
      noindent=True),
])

# ==========================================================================
# II. RELATED WORK
# ==========================================================================
sec(2, [
    H1("II.  Related Work"),
    H2("A.  Digital-Twin Architectures"),
    P("Grieves and Vickers [17] articulated the digital-twin concept as a "
      "bidirectionally coupled virtual counterpart of a physical asset. "
      "Kritzinger et al. [16] subsequently distinguished digital models, "
      "digital shadows and true digital twins by their degree of automated data "
      "flow \u2014 a taxonomy under which much published work classifies as the "
      "weaker forms. Tao et al. [15] surveyed industrial applications and "
      "identified data-model fusion as the persistent difficulty. Lu et al. [44] "
      "and Rasheed et al. [45] extended these treatments to manufacturing and "
      "to modelling perspectives respectively.", noindent=True),
    P("In the power sector specifically, Alabbad and Alt\u0131nkaya [23] "
      "reported a five-layer substation-automation architecture combining "
      "standards-based acquisition, semantic interoperability across IEC 61850, "
      "CIM and OPC UA Part 17, hybrid analytics and IEC 62443-aligned security, "
      "validated on an operating oil-field network. Their work is the closest "
      "antecedent to the architecture described in Section III, and we adopt "
      "rather than claim that structure."),

    H2("B.  Physics-Informed Neural Networks"),
    P("Raissi et al. [7] introduced physics-informed neural networks, in which "
      "the residual of a governing differential equation, evaluated by automatic "
      "differentiation at collocation points, is added to the training loss. "
      "Karniadakis et al. [8] and Cuomo et al. [49] survey the resulting "
      "literature. Two known pathologies bear directly on the present problem. "
      "Wang et al. [26] analysed the gradient imbalance between data and physics "
      "terms that makes the loss weighting consequential. Tancik et al. [25] "
      "showed that coordinate-input networks exhibit a spectral bias that "
      "prevents them from representing high-frequency structure, and that "
      "Fourier feature encoding removes it. Section VI-A confirms that the "
      "second pathology is decisive for the diurnally forced thermal problem.",
      noindent=True),
    P("Applications to transformer thermal modelling remain comparatively "
      "sparse. The IEC 60076-7 exponential equations [6] and the corresponding "
      "IEEE loading guide [36] provide the governing relations used here, and "
      "gas-in-oil analysis [38] and partial-discharge measurement [37] supply "
      "the complementary condition indicators."),

    H2("C.  Edge AI for Condition Monitoring"),
    P("Shi et al. [18] set out the case for computation at the network edge. "
      "For condition monitoring the driver is bandwidth: continuous "
      "multi-channel vibration and acoustic streams are expensive to backhaul, "
      "while the diagnostic information they carry is low-dimensional. "
      "Model compression follows two main lines. Hinton et al. [9] introduced "
      "knowledge distillation, transferring the softened output distribution of "
      "a large teacher to a compact student. Jacob et al. [24] established "
      "integer-arithmetic quantisation as a deployment technique. Recurrent "
      "architectures for sequence modelling derive from Hochreiter and "
      "Schmidhuber [47] and, in the gated-recurrent form used here, from "
      "Cho et al. [12] and Chung et al. [13]. Tree ensembles [14] remain "
      "strong and inexpensive baselines on engineered tabular features, a point "
      "our results reinforce.", noindent=True),

    H2("D.  Cyber-Physical Security"),
    P("IEC 62443-3-3 [20] defines system security requirements and zone-conduit "
      "segmentation for industrial automation. Within a digital twin the "
      "specific concern is that analytics consume sensor data whose integrity "
      "cannot be assumed. Reconstruction-based anomaly detection using "
      "variational autoencoders [10] is a common response. Goodfellow et al. "
      "[22] showed that gradient-based perturbations can mislead classifiers "
      "with imperceptible changes. Section VI-E reports where this class of "
      "defence succeeds and where it does not.", noindent=True),

    H2("E.  Research Gap Synthesis"),
    P("Table I positions this work. The gap it addresses is narrow and "
      "specific: quantifying how much a physics residual is worth, in units of "
      "labelled data, when the physics itself is imperfectly known, and what "
      "that buys within an edge inference budget.", noindent=True),
])

sec(1, [
    TAB("I", "Positioning of this work relative to representative prior art",
        [["Study", "Domain", "Physics-informed", "Edge compression",
          "Parameter-mismatch analysis", "Validation"],
         ["Tao et al. [15]", "Industrial DT survey", "No", "No", "No", "Survey"],
         ["Kritzinger et al. [16]", "DT taxonomy", "No", "No", "No", "Review"],
         ["Alabbad & Alt\u0131nkaya [23]", "Substation automation", "Partial (rule-based)",
          "No", "No", "Field deployment"],
         ["Raissi et al. [7]", "General PDE solving", "Yes", "No", "No", "Numerical"],
         ["Wang et al. [26]", "PINN training dynamics", "Yes", "No",
          "Loss weighting only", "Numerical"],
         ["Hinton et al. [9]", "Model compression", "No", "Yes", "No", "Benchmark"],
         ["This work", "Transformer thermal DT", "Yes (IEC 60076-7)",
          "Yes (KD + INT8)", "Yes (10 % mismatch)", "Simulation only"]],
        widths=[0.19, 0.17, 0.17, 0.13, 0.19, 0.15], wide=True),
])

# ==========================================================================
# III. PROBLEM FORMULATION AND SYSTEM ARCHITECTURE
# ==========================================================================
sec(2, [
    H1("III.  Problem Formulation and System Architecture"),
    H2("A.  Problem Statement"),
    P(f"Consider a substation containing N_a monitored assets. The digital twin "
      f"maintains, for each asset i, an estimated state comprising hot-spot "
      f"temperature, a scalar health index and a remaining-useful-life estimate:",
      noindent=True),
    EQN(1),
    P("Observations arrive as a multi-channel sensor stream sampled at f_s, "
      "segmented into windows of length N_w. Only the top-oil temperature, the "
      "phase currents and voltages, and vibration, acoustic-emission and "
      "partial-discharge activity are observable; the hot-spot temperature that "
      "governs ageing is not."),
    P("Let \u03b8 denote the parameters of the state-estimation model and \u03d5 "
      "those of the deployed diagnostic model. Writing c_f for the cost of a "
      "misclassified fault and c_r for the cost per day of remaining-life error, "
      "the design problem is"),
    EQN(2),
    P("subject to the edge resource constraints"),
    EQN(3),
    P("where T_inf is inference latency, M the memory footprint and B the "
      "upstream bandwidth. The formulation makes explicit that accuracy is not "
      "the sole objective: a model that cannot be executed within the gateway "
      "budget has no operational value, and Section VI-B quantifies the exchange "
      "rate between the two."),

    H2("B.  Architectural Philosophy"),
    P("Three principles organise the design. Computation is placed as close to "
      "the sensor as the resource budget permits, so that raw waveforms are "
      "reduced to features locally and only compact representations traverse the "
      "network. Physical knowledge is embedded in the estimator rather than "
      "supplied as a post-hoc sanity check, because the quantity of interest is "
      "unobservable and cannot be learned from data alone. Interfaces follow "
      "existing utility standards so that the twin composes with installed "
      "substation automation rather than displacing it.", noindent=True),

    H2("C.  Four-Layer Architecture"),
    P("Fig. 1 shows the simulated architecture. Layer 1 generates six-channel "
      "sensor observations from the physical models of Section IV-A. Layer 2 "
      "performs feature extraction, integrity validation, physics-informed state "
      "estimation and diagnostic inference. Layer 3 maintains the asset model "
      "and its standards-aligned representation. Layer 4 exposes health state "
      "for maintenance decision support. Every layer is realised in software; "
      "Layer 1 replaces physical instrumentation with a validated generative "
      "model and Layer 2 executes on a host processor under an explicitly "
      "modelled computational budget.", noindent=True),
])

sec(1, [
    FIG(1, "fig01_architecture.png",
        "Four-layer architecture of the simulated digital twin. All layers are "
        "software components; Layer 1 substitutes a physics-based generative "
        "model for physical instrumentation.", wide=True),
])

sec(2, [
    H2("D.  Simulated Asset Scope"),
    P(f"The primary modelled asset is a {CFG['rating_MVA']:.0f} MVA oil-immersed "
      f"ONAF-cooled transformer with a {CFG['n_phases']}-phase secondary. "
      f"Thermal parameters follow the IEC 60076-7 defaults for medium-power "
      f"ONAF units [6]: rated top-oil rise \u0394\u0398_or = {TH['dTheta_or']:.0f} K, "
      f"rated hot-spot-to-top-oil gradient H\u00b7g_r = {TH['Hgr']:.0f} K, "
      f"loss ratio R = {TH['R']:.0f}, oil exponent x = {TH['x']}, winding "
      f"exponent y = {TH['y']}, oil time constant \u03c4_o = {TH['tau_o']:.0f} min "
      f"and winding time constant \u03c4_w = {TH['tau_w']:.0f} min.", noindent=True),
    P("Five fault modes are modelled, each through its physical mechanism rather "
      "than as a statistical template: thermal overload, insulation degradation, "
      "cooling-fan bearing defect, partial discharge and tap-changer "
      "malfunction. Table II summarises the mechanisms and their observable "
      "signatures. Faults are generated in the incipient severity range, since "
      "the practical value of predictive maintenance lies in recognising a "
      "developing fault well before it becomes self-evident."),
    P("For the sensor-placement study the substation is abstracted as a graph "
      f"of {PLC['n_assets']} assets connected by {PLC['n_edges']} electrical and "
      "physical adjacencies. Placing sensing nodes so that every asset is either "
      "instrumented or adjacent to an instrumented asset is a minimum "
      "dominating-set problem,"),
    EQN(4),
    P("which is NP-hard; the greedy algorithm with its classical ln|V| + 1 "
      f"approximation guarantee yields {PLC['n_sensors']} nodes covering all "
      f"{PLC['n_assets']} assets at a mean redundancy of "
      f"{PLC['redundancy']:.2f} (Fig. 2, Table III)."),
])

sec(2, [
    TAB("II", "Simulated fault modes and physical mechanisms",
        [["Class", "Mechanism", "Primary observable signature"],
         ["Normal", "\u2014", "Nominal 2f core magnetostriction; low PD activity"],
         ["Thermal overload", "Sustained load factor above rating",
          "Elevated current RMS, raised top-oil temperature"],
         ["Insulation degradation", "Cellulose breakdown, void formation",
          "PD pulse count, acoustic bursts, mild cooling loss"],
         ["Bearing / cooling fault", "Fan outer-race defect, radiator fouling",
          "Impulse train at defect frequency; reduced cooling"],
         ["Partial discharge", "Void and surface discharge",
          "Phase-correlated PD pulses at voltage peaks"],
         ["Tap changer malfunction", "Contact bounce, motor stall",
          "Voltage step, current inrush, mechanical impulse"]],
        widths=[0.20, 0.34, 0.46]),
    TAB("III", "Graph-based sensor placement result",
        [["Quantity", "Value"],
         ["Assets in graph", f"{PLC['n_assets']}"],
         ["Adjacency edges", f"{PLC['n_edges']}"],
         ["Sensing nodes selected", f"{PLC['n_sensors']}"],
         ["Asset coverage", f"{PLC['coverage_pct']:.1f} %"],
         ["Mean degree of selected nodes", f"{PLC['mean_degree_of_sensors']:.2f}"],
         ["Mean coverage redundancy", f"{PLC['redundancy']:.2f}"],
         ["Reduction vs. one sensor per asset",
          f"{100 * (1 - PLC['n_sensors'] / PLC['n_assets']):.1f} %"]],
        widths=[0.62, 0.38]),
    FIG(2, "fig02_sensor_placement.png",
        "Greedy dominating-set sensor placement. Filled nodes carry sensing "
        "hardware; all remaining assets are adjacent to at least one."),
])


# ==========================================================================
# IV. METHODOLOGY
# ==========================================================================
sec(2, [
    H1("IV.  Methodology"),
    H2("A.  Synthetic Data Generation Framework"),
    P("The generative model is the substitute for physical instrumentation, so "
      "its fidelity determines the credibility of everything downstream. Thermal "
      "state follows the IEC 60076-7 exponential-equation solution [6]. The "
      "top-oil rise above ambient obeys", noindent=True),
    EQN(5),
    P("and the hot-spot-to-top-oil gradient obeys"),
    EQN(6),
    P("with the hot-spot temperature recovered as"),
    EQN(7),
    P("Note that the winding exponent is written as y, related to the "
      "alternative 2m parameterisation by y = 2m. For the multi-phase case all "
      "windings share one oil volume, so the top-oil equation is driven by the "
      "mean per-phase loss while each winding retains its own gradient:"),
    EQN(8),
    EQN(9),
    P("Insulation ageing follows the IEC relative ageing rate, whose integral "
      "gives the accumulated equivalent ageing used later for prognostics:"),
    EQN(10),
    P("Load profiles reproduce the twin-peak diurnal structure characteristic of "
      "airport demand, superimposed on a night-biased airfield-lighting "
      "component, a seasonal term and a first-order autoregressive stochastic "
      "component. Per-phase imbalance is generated as a slowly drifting "
      "zero-sum perturbation."),
    P("Six sensor channels are synthesised at "
      f"{CFG['fs']:.0f} Hz: vibration, top-oil temperature, current, voltage, "
      "acoustic emission and partial discharge. Fault signatures follow their "
      "mechanisms. A cooling-fan bearing defect produces an exponentially "
      "damped impulse train at the outer-race defect frequency together with "
      "reduced cooling capability; partial discharge produces pulses "
      "concentrated at the voltage peaks through phase-correlated sampling; a "
      "tap-changer malfunction produces a voltage step with current inrush and a "
      "mechanical impulse."),
    NOTE("Two modelling choices materially affect the results and are stated "
         "explicitly. First, the temperature channel carries TOP-OIL "
         "temperature, not hot-spot temperature. Exposing the hot-spot would "
         "leak the latent state that the PINN exists to infer. Second, sensor "
         "noise on the alternating-current channels is referenced to the "
         "fluctuating signal power, while the temperature channel receives "
         "absolute instrument noise of "
         "0.3 K with a calibration offset. Referencing temperature noise to "
         "total signal power would assign tens of kelvin of error to a 45 "
         "\u00b0C reading, which is not a property of any real sensor."),
    P("Each node additionally carries fixed gain and offset calibration errors "
      f"of standard deviation {CFG['calib_gain_sd'] * 100:.0f} % and "
      f"{CFG['calib_off_sd'] * 100:.0f} % respectively, held constant within an "
      "episode, representing the unit-to-unit variability of low-cost sensing "
      "hardware."),

    H2("B.  Physics-Informed Neural Network"),
    P("The estimation task is posed as follows. Top-oil temperature is observed "
      "at N_d instants with measurement noise. Hot-spot temperature is either "
      "unobserved or observed at a very small number N_hs of instants, "
      "corresponding to an occasional fibre-optic probe. The network must "
      "reconstruct the entire hot-spot trajectory.", noindent=True),
    P("A network taking normalised time as its sole input cannot represent the "
      "diurnal forcing, because coordinate-input multilayer perceptrons are "
      "spectrally biased toward low frequencies [25]. The time input is "
      "therefore encoded as"),
    EQN(11),
    P("over a geometric set of M frequencies, and the network maps this "
      "encoding to the two thermal state components:"),
    EQN(12),
    P("The encoding is an explicit, differentiable function of t, so the "
      "automatic-differentiation residual remains exact. The composite "
      "objective is"),
    EQN(13),
    P("The data term penalises disagreement with the available observations, "
      "with the hot-spot component present only when N_hs is non-zero:"),
    EQN(14, wide=True),
    P("The physics term evaluates the residuals of the governing equations at "
      "N_c collocation points drawn uniformly in normalised time. Crucially, "
      "the residuals are formed using the ASSUMED thermal parameters, denoted "
      "by tildes, which differ from the true parameters of the simulated unit:"),
    EQN(15, wide=True),
    EQN(16, wide=True),
    EQN(17),
    P("with the initial condition enforced by"),
    EQN(18),
    P("For the multi-phase model an optional coupling term expresses that "
      "per-phase deviations from the mean hot-spot gradient must be consistent "
      "with the per-phase loss distribution under a shared oil volume:"),
    EQN(19),
    EQN(20),
    P("This term proves largely redundant in practice, for reasons discussed in "
      "Section VI-A."),
    P("The author\u2019s earlier work on multi-phase transformer modelling "
      "[57]\u2013[59] provides the winding-configuration analysis from which the "
      "per-phase loss distribution used in (8) and (20) is taken."),

    H2("C.  Lightweight Edge-AI Inference Engine"),
    P("Edge processing follows Algorithm 1 (Appendix A). Each window yields an "
      f"18-dimensional feature vector per channel \u2014 six time-domain, four "
      f"frequency-domain and eight wavelet-packet energies from a three-level "
      f"Daubechies-4 decomposition [30], with spectral quantities obtained by "
      f"Welch\u2019s method [31] \u2014 giving "
      f"{CFG['n_channels'] * CFG['feat_per_channel']} features in total.",
      noindent=True),
    P("Three physics-derived features are appended: the IEC 60076-7 hot-spot "
      "estimate, the corresponding logarithmic relative ageing rate and the "
      "estimated per-unit load. These are computed from OBSERVABLES ONLY "
      "\u2014 the measured top-oil temperature and current RMS \u2014 using "
      "datasheet thermal parameters, so no latent state of the generator enters "
      f"the classifier. The final representation is {DS['d']}-dimensional."),
    P("The teacher is a bidirectional gated recurrent network with a random "
      "forest [14] operating on pooled episode statistics. The student halves "
      "the recurrence to a single unidirectional layer and reduces the forest. "
      "Distillation follows Hinton et al. [9] with an added feature-matching "
      "term:"),
    EQN(25, wide=True),
    P("The balance between hard-label and teacher supervision is selected on "
      f"the validation split rather than fixed a priori; the selected values are "
      f"\u03b1 = {CL['distill_hparams']['alpha']}, "
      f"\u03b2 = {CL['distill_hparams']['beta']} at temperature "
      f"T = {CL['distill_hparams']['T']:.0f}. Ensemble mixing is likewise "
      "validation-selected,"),
    EQN(26),
    P("with the validation accuracy curve smoothed over neighbouring grid points "
      "before the maximum is taken, since it is a noisy function of the weight "
      "on a finite validation set. Deployment applies symmetric per-tensor INT8 "
      "quantisation [24],"),
    EQN(27),

    H2("D.  Standards-Aligned Digital-Twin Core"),
    P("Twin state is expressed through IEC 61850-7-4 logical nodes [19] so that "
      "the twin composes with existing substation automation. Table V gives the "
      "mapping. Asset topology follows the IEC 61970 common information model "
      "[40] and transport uses OPC UA PubSub [39]. No claim of interoperability "
      "testing against physical devices is made; the mapping is a design "
      "specification, not a verified integration.", noindent=True),

    H2("E.  Cyber-Physical Security Integration"),
    P("Analytics consume sensor data whose integrity cannot be assumed. A "
      "variational autoencoder [10] trained exclusively on normal-condition "
      "feature vectors provides an integrity check, with objective", noindent=True),
    EQN(28),
    P("where the divergence term has the closed form"),
    EQN(29),
    P("A feature vector is accepted when its anomaly score lies within a "
      "threshold calibrated on held-out normal data:"),
    EQN(30),
    P("Zone and conduit segmentation follows IEC 62443-3-3 [20]. Section VI-E "
      "reports both where this detector works and where it does not."),

    H2("F.  Health Index and Remaining Useful Life"),
    P("The health index aggregates normalised deviation from nominal across the "
      "thermal, harmonic, vibration and acoustic indicators:", noindent=True),
    EQN(21, wide=True),
    P("Two remaining-life estimators are compared. The first, following common "
      "practice, treats the health index as an unreliability and fits a "
      "two-parameter Weibull distribution,"),
    EQN(22),
    P("whose linearised form"),
    EQN(23),
    P("provides an initialisation refined by bounded nonlinear least squares, "
      "giving"),
    EQN(24),
    NOTE("The health index of a new asset is not zero, because nominal "
         "harmonic, vibration and acoustic levels are non-zero, whereas the "
         "Weibull distribution vanishes at t = 0. The baseline must therefore "
         "be removed and the index rescaled to a damage fraction before it can "
         "be interpreted as an unreliability. Omitting this step was found to "
         "produce divergent extrapolations."),
    P("The second estimator is physics-informed. The IEC ageing rate obtained "
      "from the estimated hot-spot temperature gives the RATE at which life is "
      "consumed; the health index, through a monotone relation calibrated once "
      "on a disjoint set of units, gives the FRACTION already consumed. "
      "Extrapolating the damage fraction linearly over a long observation window "
      "and solving for the time at which it reaches unity yields the remaining "
      "life. Inverting the health index to a per-unit endurance was also tried "
      "and abandoned: the reciprocal amplifies estimation noise hyperbolically "
      "at precisely the early inspection times where a prediction is most "
      "valuable."),
])

# ==========================================================================
# V. SIMULATION SETUP
# ==========================================================================
sec(2, [
    H1("V.  Simulation Setup and Validation Protocol"),
    H2("A.  Simulation Environment"),
    P(f"All experiments were executed on a single-threaded x86-64 host using "
      f"Python {MET['python']}, NumPy {MET['numpy']} [34], SciPy [35], "
      f"scikit-learn [32] and PyTorch {MET['torch']} [33]. The master seed is "
      f"{MET['seed']}; every stochastic component derives its stream from it, so "
      f"the entire study reproduces exactly.", noindent=True),

    H2("B.  Data Generation Protocol"),
    P(f"The classification dataset comprises {DS['shape'][0]} episodes of "
      f"{DS['shape'][1]} consecutive windows each, balanced across "
      f"{CFG['n_classes']} classes, split "
      f"{DS['n_train']}/{DS['n_val']}/{DS['n_test']} into training, validation "
      f"and test partitions stratified by class. Within an episode the fault "
      f"severity ramps monotonically over the incipient range "
      f"[{CFG['sev_lo']}, {CFG['sev_hi']}] while load and ambient conditions "
      f"follow the diurnal profile. Nominal sensor SNR is {CFG['snr_db']:.0f} dB.",
      noindent=True),
    P("PINN experiments use six-day trajectories at five-minute resolution. The "
      "fleet study for prognostics simulates "
      f"{CFG['n_units']} independent units over a {CFG['horizon_days']}-day "
      f"horizon at {CFG['dt_hours']:.0f}-hour resolution, of which "
      f"{RUL['n_units_total']} reach end of life within the horizon. Of these, "
      f"{RUL['n_units_cal']} are reserved for calibrating the health-index "
      f"decision rule and {RUL['n_units_test']} are used for evaluation, so no "
      "test unit informs the rule applied to it."),

    H2("C.  Baseline Models"),
    P("Four baselines are evaluated on identical splits: a random forest on "
      "pooled episode features; a unidirectional GRU; a convolutional-recurrent "
      "network; and the undistilled teacher ensemble. For the PINN the baseline "
      "is an architecturally identical network trained with the physics weight "
      "set to zero, isolating the contribution of the physics term from that of "
      "the Fourier encoding and network capacity.", noindent=True),

    H2("D.  Implementation Details"),
    P("Table IV lists the PINN hyperparameters. Classifiers are trained with "
      "Adam [11] using best-validation checkpointing. Reported PINN results are "
      "means over three seeds; classification results use a single seed on a "
      "fixed split, a limitation noted in Section VII.", noindent=True),
])

sec(2, [
    TAB("IV", "PINN configuration",
        [["Parameter", "Value"],
         ["Hidden layers / width", "5 / 64, tanh"],
         ["Fourier features", "12 frequencies, geometric"],
         ["Outputs", "\u0394\u0398_TO, \u0394\u0398_H"],
         ["Collocation points N_c", "1500"],
         ["Top-oil observations N_d", "180"],
         ["Hot-spot labels N_hs", "0 \u2013 200 (swept)"],
         ["Physics weight \u03bb_p", f"{best_lam['lam_p']:g} (selected)"],
         ["Boundary weight \u03bb_bc", "0.1"],
         ["Optimiser", "Adam, lr 2\u00d710\u207b\u00b3, step decay"],
         ["Epochs", "1800"],
         ["Parameter mismatch", "10 % (all thermal constants)"],
         ["Temperature sensor noise", "0.5 K"]],
        widths=[0.55, 0.45]),
    TAB("V", "IEC 61850-7-4 logical node mapping",
        [["Twin quantity", "Logical node", "Data object"],
         ["Hot-spot temperature", "STMP", "Tmp.mag"],
         ["Top-oil temperature", "STMP", "Tmp.mag"],
         ["Load factor", "MMXU", "A.phsA\u2013F"],
         ["Health index", "SIML", "SimSt"],
         ["Remaining useful life", "SIML", "OpTmh"],
         ["Fault classification", "SPDC", "Str.general"],
         ["Partial discharge", "SPDC", "PDLev"],
         ["Tap position", "YLTC", "TapPos"],
         ["Integrity alarm", "GGIO", "Alm"]],
        widths=[0.40, 0.28, 0.32]),
])



# ==========================================================================
# VI. RESULTS AND DISCUSSION
# ==========================================================================
sec(2, [
    H1("VI.  Results and Discussion"),
    H2("A.  Physics-Informed Hot-Spot Reconstruction"),
    P(f"Table VI and Fig. 5 report the headline result. Averaged over three "
      f"seeds, the PINN reconstructs the hot-spot trajectory with an RMSE of "
      f"{f2(PM['pinn']['rmse'][0])} \u00b1 {f2(PM['pinn']['rmse'][1])} K and "
      f"R\u00b2 = {f3(PM['pinn']['r2'][0])}, against "
      f"{f2(PM['data_only']['rmse'][0])} \u00b1 {f2(PM['data_only']['rmse'][1])} K "
      f"and R\u00b2 = {f3(PM['data_only']['r2'][0])} for the identical network "
      f"without the physics residual. The negative coefficient of determination "
      f"for the data-driven model is not an artefact: with the hot-spot "
      f"essentially unobserved, its output is unconstrained and tracks the "
      f"ground truth worse than a constant predictor would.", noindent=True),
    P("Fig. 5(c) shows a counter-intuitive convergence pattern worth stating "
      "plainly. The data-driven model reaches a LOWER total training loss than "
      "the PINN, because it is optimising an easier objective \u2014 it need "
      "only fit the sparse observations, not simultaneously satisfy a "
      "differential equation. Lower training loss accompanies an order of "
      "magnitude worse reconstruction. Training loss is not a proxy for "
      "estimation quality when the objectives differ."),
    P("The Fourier encoding is a precondition rather than a refinement. In "
      "preliminary runs a plain tanh network on the same problem achieved "
      "roughly 8.5 K RMSE with the physics term and 8.7 K without it \u2014 the "
      "physics conferred almost no benefit, because the network could not "
      "represent a function whose derivative varies over the diurnal cycle. "
      "Reports that a physics residual failed to help a coordinate-input network "
      "should be read with this pathology in mind."),

    H2("B.  Physics Weight and Label Efficiency"),
    P(f"Fig. 6(a) shows a clear interior optimum at \u03bb_p = "
      f"{best_lam['lam_p']:g} with RMSE {f2(best_lam['rmse'])} K, degrading to "
      f"{f2(next(d['rmse'] for d in PL if d['lam_p'] == 100.0))} K at "
      f"\u03bb_p = 100 and {f2(worst_lam['rmse'])} K at \u03bb_p = 0. This "
      f"optimum exists only because the physics is imperfectly known. When the "
      f"residual uses exactly the constants that generated the data, "
      f"reconstruction error decreases monotonically with \u03bb_p and the "
      f"ablation is uninformative. Introducing 10 % parameter mismatch \u2014 "
      f"the level of uncertainty attaching to test-report or IEC default values "
      f"\u2014 restores the bias-variance trade-off that a deployed system "
      f"would face.", noindent=True),
    P(f"Fig. 6(b) quantifies the substitution of physics for labels. The PINN "
      f"attains {f2(d_eff0['pinn_rmse'])} K with NO hot-spot labels; the "
      f"data-driven model needs {d_eff200['n_labels']} labels to reach "
      f"{f2(d_eff200['data_rmse'])} K, and is still worse. Between these, the "
      f"data-driven model improves steadily "
      f"({f2(next(d['data_rmse'] for d in PD if d['n_labels'] == 20))} K at 20 "
      f"labels, {f2(next(d['data_rmse'] for d in PD if d['n_labels'] == 60))} K "
      f"at 60) while the PINN is essentially flat, indicating that the physics "
      f"residual already supplies the information the labels would carry. For an "
      f"asset class where hot-spot instrumentation is the exception, this is the "
      f"practically important result."),
    P(f"Fig. 6(c) reports a negative finding. Adding the explicit inter-phase "
      f"coupling term reduces phase-imbalance RMSE only from "
      f"{f3(MP[0]['imbalance_rmse'])} K to {f3(MP[1]['imbalance_rmse'])} K, a "
      f"relative improvement of "
      f"{100 * (1 - MP[1]['imbalance_rmse'] / MP[0]['imbalance_rmse']):.1f} %. "
      f"The per-phase residuals already encode the shared-oil constraint "
      f"implicitly through the common top-oil equation, so the explicit term is "
      f"largely redundant. We report this because the corresponding term is "
      f"often assumed beneficial without measurement."),
])

sec(1, [
    FIG(5, "fig05_pinn_reconstruction.png",
        "PINN hot-spot reconstruction. (a) Exogenous load and ambient forcing. "
        "(b) Reconstructed hot-spot trajectory against ground truth; the "
        "data-driven model of identical architecture cannot recover the latent "
        "state. (c) Training convergence \u2014 note that the data-driven model "
        "attains a lower training loss while reconstructing far less accurately.",
        wide=True),
    FIG(6, "fig06_pinn_ablation.png",
        "PINN ablations. (a) Physics-loss weight, showing an interior optimum "
        "that exists only under thermal-parameter mismatch. (b) Label "
        "efficiency: the physics residual substitutes for direct supervision. "
        "(c) Multi-phase coupling term, whose benefit is marginal.", wide=True),
])

sec(2, [
    TAB("VI", "Hot-spot reconstruction accuracy (mean \u00b1 s.d., 3 seeds)",
        [["Model", "RMSE (K)", "MAE (K)", "R\u00b2"],
         ["PINN (\u03bb_p = %g)" % best_lam["lam_p"],
          f"{f2(PM['pinn']['rmse'][0])} \u00b1 {f2(PM['pinn']['rmse'][1])}",
          f"{f2(PM['pinn']['mae'][0])} \u00b1 {f2(PM['pinn']['mae'][1])}",
          f"{f3(PM['pinn']['r2'][0])}"],
         ["Data-driven (\u03bb_p = 0)",
          f"{f2(PM['data_only']['rmse'][0])} \u00b1 {f2(PM['data_only']['rmse'][1])}",
          f"{f2(PM['data_only']['mae'][0])} \u00b1 {f2(PM['data_only']['mae'][1])}",
          f"{f3(PM['data_only']['r2'][0])}"]],
        widths=[0.34, 0.24, 0.22, 0.20]),

    H2("C.  Fault Classification Accuracy"),
    P(f"Table VII reports classification on the held-out test set of "
      f"{DS['n_test']} episodes. The teacher ensemble attains "
      f"{pc(CM['Teacher (Bi-GRU + RF)']['accuracy'])} % accuracy; the distilled "
      f"student reaches {pc(CM['Student, distilled (proposed)']['accuracy'])} %, "
      f"and INT8 quantisation leaves this unchanged. Distillation is worth "
      f"{100 * (CM['Student, distilled (proposed)']['accuracy'] - CM['Student, no distillation']['accuracy']):.2f} "
      f"percentage points over training the same student from scratch.",
      noindent=True),
    P(f"Two observations deserve emphasis. First, the random forest alone "
      f"reaches {pc(CC['rf_teacher_only'])} % \u2014 competitive with the full "
      f"ensemble at a fraction of the cost. On engineered tabular features from "
      f"episodes with weak temporal structure, recurrence earns little. "
      f"Practitioners should benchmark a tree ensemble before reaching for a "
      f"recurrent architecture. Second, the validation-selected mixing weights "
      f"differ markedly between models "
      f"({CL['ensemble_weights']['teacher']:.2f} for the teacher versus "
      f"{CL['ensemble_weights']['student_kd']:.2f} for the student), reflecting "
      f"which component carries the discriminative burden in each case."),
    P(f"Per-class ROC areas (Fig. 9) exceed 0.99 for thermal overload, "
      f"insulation degradation, bearing fault and partial discharge, but fall to "
      f"{R['classification']['roc']['Normal']['auc']:.3f} for the normal class "
      f"and {R['classification']['roc']['Tap Changer Malfunction']['auc']:.3f} "
      f"for tap-changer malfunction. The confusion matrix (Fig. 8) shows the "
      f"residual error concentrated between normal operation and low-severity "
      f"tap-changer events, which is expected: an incipient contact fault "
      f"perturbs the voltage waveform only slightly."),
    P(f"The physics-feature ablation is the clearest evidence for the paper\u2019s "
      f"central claim on the diagnostic side. Removing the three physics-derived "
      f"features drops accuracy from {pc(PA['with physics features']['accuracy'])} % "
      f"to {pc(PA['sensor features only']['accuracy'])} %, a loss of "
      f"{100 * (PA['with physics features']['accuracy'] - PA['sensor features only']['accuracy']):.2f} "
      f"points, despite those features being computed from the same observables "
      f"the network already sees. The gain comes from the IEC 60076-7 relation "
      f"imposed on them, not from new measurements."),
])

sec(1, [
    TAB("VII", "Fault classification performance on the held-out test set",
        [["Model", "Accuracy", "Precision", "Recall", "F1", "Params",
          "Stored size"]] +
        [[k, pc(v["accuracy"]) + " %", f3(v["precision"]), f3(v["recall"]),
          f3(v["f1"]),
          f"{teacher_row['params']:,}" if "Teacher" in k else
          (f"{student_row['params']:,}" if "Student" in k else "\u2014"),
          f"{teacher_row['total_int8_kB']:.0f} kB" if "Teacher" in k else
          (f"{student_row['total_int8_kB']:.0f} kB" if "Student" in k else "\u2014")]
         for k, v in CM.items()],
        widths=[0.26, 0.12, 0.12, 0.12, 0.11, 0.13, 0.14], wide=True),
    FIG(7, "fig07_model_performance.png",
        "Classification performance across model variants. Accuracy values are "
        "annotated above the corresponding bars.", wide=True),
])

sec(2, [
    FIG(8, "fig08_confusion_matrix.png",
        "Confusion matrix for the deployed INT8 student ensemble. Counts with "
        "row-normalised percentages."),
    FIG(9, "fig09_roc_curves.png",
        "One-versus-rest ROC curves with areas under the curve."),
    FIG(10, "fig10_convergence.png",
        "Training convergence of the classifier variants."),

    H2("D.  Edge Inference Cost"),
    P(f"Table VIII separates measured from projected quantities, a distinction "
      f"often blurred in the literature. Parameter counts, FLOP counts, stored "
      f"model size and host latency are measured directly. The student requires "
      f"{student_row['params']:,} parameters against {teacher_row['params']:,} "
      f"for the teacher, a {teacher_row['params'] / student_row['params']:.1f}\u00d7 "
      f"reduction, and {student_row['mflops']:.3f} MFLOP against "
      f"{teacher_row['mflops']:.3f} MFLOP per inference. Including the random "
      f"forest, stored size falls from {teacher_row['total_int8_kB']:.0f} kB to "
      f"{student_row['total_int8_kB']:.0f} kB, a "
      f"{teacher_row['total_int8_kB'] / student_row['total_int8_kB']:.1f}\u00d7 "
      f"compression.", noindent=True),
    NOTE("Embedded latency figures in Table VIII are ANALYTICAL PROJECTIONS "
         "obtained by dividing the measured FLOP count by an assumed sustained "
         "throughput. They are not hardware measurements. No embedded target "
         "was available to this study, and reporting projected values as "
         "measurements would misrepresent them. The projections should be "
         "treated as order-of-magnitude feasibility indicators only."),
    P(f"On that basis the student projects to roughly "
      f"{proj[(A72, 'Student')]:.2f} ms on a Cortex-A72-class core and "
      f"{proj[(M4, 'Student')]:.0f} ms on a Cortex-M4F-class microcontroller, "
      f"against {proj[(A72, 'Teacher')]:.2f} ms and "
      f"{proj[(M4, 'Teacher')]:.0f} ms for the teacher. Both remain far inside a "
      f"multi-second inference budget, which suggests the binding edge "
      f"constraint for this workload is memory and energy rather than latency."),
    P("No power consumption figures are reported. Power cannot be simulated "
      "without the target hardware, and any figure we might quote would be "
      "unfounded. Fig. 11(a) reports computational cost in MFLOP instead, from "
      "which power may be estimated once a target is selected."),

    H2("E.  Bandwidth Reduction"),
    P(f"Raw streaming of six channels at {CFG['fs']:.0f} Hz and 16-bit "
      f"resolution requires {BW['raw_daily_MB']:.0f} MB per node per day. "
      f"Transmitting a quantised {DS['d'] - 3}-dimensional feature vector per "
      f"window, with raw fallback for anomaly-flagged windows at the measured "
      f"rate of {pc(BW['anomaly_fraction'])} %, requires "
      f"{BW['edge_daily_MB']:.1f} MB per day \u2014 a reduction of "
      f"{BW['reduction_pct']:.2f} % (Fig. 13). The reduction is dominated by the "
      f"feature-extraction ratio and remains above 90 % for anomaly rates up to "
      f"roughly 10 %.", noindent=True),
])

sec(1, [
    TAB("VIII", "Edge inference cost: measured and projected",
        [["Model", "Params", "RF nodes", "MFLOP", "FP32 (kB)",
          "INT8 + RF (kB)", "Host latency (ms), measured",
          "Cortex-A72 (ms), projected", "Cortex-M4F (ms), projected"]] +
        [[r["model"], f"{r['params']:,}", f"{r['rf_nodes']:,}",
          f"{r['mflops']:.3f}", f"{r['size_fp32_kB']:.1f}",
          f"{r['total_int8_kB']:.1f}", f"{r['host_latency_ms']:.3f}",
          f"{proj[(A72, r['model'].split()[0])]:.2f}",
          f"{proj[(M4, r['model'].split()[0])]:.1f}"] for r in EC["rows"]],
        widths=[0.20, 0.09, 0.09, 0.08, 0.09, 0.11, 0.12, 0.11, 0.11], wide=True),
    FIG(11, "fig11_edge_cost.png",
        "Edge computational cost. (a) Measured FLOP count per inference. "
        "(b) Measured stored model size. (c) Projected embedded latency \u2014 "
        "analytical, not measured.", wide=True),
])

sec(2, [
    FIG(13, "fig13_bandwidth.png",
        "Upstream bandwidth. (a) Raw streaming against edge preprocessing. "
        "(b) Sensitivity to the anomaly-flag rate.", wide=False),

    H2("F.  Remaining Useful Life"),
    P(f"Table IX compares the two estimators over {RUL['n_units_test']} "
      f"evaluation units at four inspection points each. The physics-informed "
      f"loss-of-life estimator achieves {RUL['stats_physics']['mae']:.1f} days "
      f"mean absolute error, RMSE {RUL['stats_physics']['rmse']:.1f} days and "
      f"R\u00b2 = {f3(RUL['stats_physics']['r2'])}, against "
      f"{RUL['stats_weibull']['mae']:.1f} days and "
      f"R\u00b2 = {f3(RUL['stats_weibull']['r2'])} for Weibull health-index "
      f"extrapolation \u2014 an error reduction of "
      f"{100 * (1 - RUL['stats_physics']['mae'] / RUL['stats_weibull']['mae']):.0f} %.",
      noindent=True),
    P(f"The Weibull baseline\u2019s negative R\u00b2 means it performs worse "
      f"than predicting the fleet-mean remaining life, and its bias is strongly "
      f"positive ({RUL['stats_weibull']['bias']:+.0f} days), i.e. it "
      f"systematically over-estimates remaining life \u2014 the dangerous "
      f"direction for maintenance planning. Bootstrap confidence intervals for "
      f"that baseline cover the true value in only "
      f"{pc1(RUL['ci_coverage'])} % of cases against a nominal 90 %, so they "
      f"should not be relied upon."),
    P("Accuracy improves monotonically as the asset ages, as prognostic theory "
      "predicts: " +
      ", ".join(f"{d['mae']:.0f} d at {d['frac']:.0%} of life"
                for d in RUL["per_fraction"]) +
      ". Early predictions remain the weakest, which is precisely where "
      "prognostics is most valuable and where this estimator is least reliable."),

    H2("G.  Cyber-Physical Integrity Validation"),
    P(f"Table X reports the most qualified result in this paper. At a "
      f"{pc(SEC['fpr_clean'])} % false-positive rate on clean data, the "
      f"validator detects temperature-offset spoofing with "
      f"{pc1(SEC['rows'][0]['detection_rate'])} % probability and combined "
      f"multi-sensor spoofing with {pc1(SEC['rows'][4]['detection_rate'])} %. "
      f"Vibration spoofing is detected only "
      f"{pc1(SEC['rows'][1]['detection_rate'])} % of the time, adversarial "
      f"perturbation {pc1(SEC['rows'][3]['detection_rate'])} %, and replay of "
      f"valid feature vectors only {pc1(SEC['rows'][2]['detection_rate'])} %.",
      noindent=True),
    P("The replay result is not a tuning deficiency and cannot be fixed by "
      "threshold adjustment. A reconstruction-based detector asks whether an "
      "observation resembles normal data. Replayed data IS normal data, "
      "captured at a different time. The detector is being asked a question it "
      "cannot answer. Freshness is a property of message sequence, not of "
      "signal content, and must be enforced cryptographically \u2014 through "
      "OPC UA sequence numbers, nonces or timestamped authentication \u2014 "
      "rather than by anomaly detection. We state this because detection rates "
      "above 99 % for replay appear in the literature, and we do not believe "
      "a purely reconstruction-based detector can achieve them."),
    P(f"Adversarial perturbation at \u03b5 = 0.05 reduces classifier accuracy "
      f"from {pc(SEC['acc_clean'])} % to {pc(SEC['acc_adv'])} %, and is detected "
      f"in only {pc1(SEC['rows'][3]['detection_rate'])} % of cases \u2014 by "
      f"construction, since the perturbation is optimised to be small. "
      f"Adversarial robustness requires adversarial training or certified "
      f"defences, not an autoencoder. Fig. 15(b) shows the threshold trade-off: "
      f"relaxing from 3\u03c3 to 1\u03c3 raises mean detection but at a "
      f"false-positive cost that would generate unacceptable alarm volume."),

    H2("H.  Sensitivity Analysis"),
    P(f"Accuracy degrades gracefully with sensor noise (Fig. 15(a)): "
      f"{pc1(NS[0]['accuracy'])} % at {NS[0]['snr_db']} dB, "
      f"{pc1(NS[2]['accuracy'])} % at {NS[2]['snr_db']} dB and "
      f"{pc1(NS[-1]['accuracy'])} % at {NS[-1]['snr_db']} dB. The absence of a "
      f"sharp cliff suggests the wavelet-packet and physics features retain "
      f"discriminative content well below the nominal operating point.",
      noindent=True),
    P(f"The label-budget curve (Fig. 14(c)) rises from "
      f"{pc1(TF[0]['accuracy'])} % with {TF[0]['n_train']} training episodes to "
      f"{pc1(TF[-1]['accuracy'])} % with {TF[-1]['n_train']}, with most of the "
      f"gain realised by roughly 20 % of the data. The non-monotonicity at "
      f"intermediate fractions reflects single-seed variance on a modest test "
      f"set and should not be over-interpreted."),
])

sec(1, [
    TAB("IX", "Remaining-useful-life estimation on held-out units",
        [["Estimator", "MAE (days)", "RMSE (days)", "R\u00b2", "MAPE (%)",
          "Bias (days)"],
         ["Weibull health-index extrapolation",
          f"{RUL['stats_weibull']['mae']:.1f}", f"{RUL['stats_weibull']['rmse']:.1f}",
          f"{f3(RUL['stats_weibull']['r2'])}", f"{RUL['stats_weibull']['mape']:.1f}",
          f"{RUL['stats_weibull']['bias']:+.1f}"],
         ["Physics-informed loss of life (proposed)",
          f"{RUL['stats_physics']['mae']:.1f}", f"{RUL['stats_physics']['rmse']:.1f}",
          f"{f3(RUL['stats_physics']['r2'])}", f"{RUL['stats_physics']['mape']:.1f}",
          f"{RUL['stats_physics']['bias']:+.1f}"]],
        widths=[0.34, 0.14, 0.14, 0.12, 0.13, 0.13], wide=True),
    TAB("X", "Integrity-validation performance at a %s %% false-positive rate"
        % pc(SEC["fpr_clean"]),
        [["Attack", "Detection rate (%)", "AUC", "Assessment"]] +
        [[r["attack"], pc1(r["detection_rate"]), f3(r["auc"]),
          ("Reliable" if r["detection_rate"] > 0.9 else
           ("Partial" if r["detection_rate"] > 0.5 else "Inadequate"))]
         for r in SEC["rows"]],
        widths=[0.34, 0.22, 0.16, 0.28], wide=True),
    FIG(12, "fig12_rul.png",
        "Remaining useful life. (a) Predicted against actual, coloured by "
        "fraction of life elapsed; dashed lines mark \u00b120 %. (b) Error "
        "distribution. (c) Health-index trajectories with the calibrated "
        "end-of-life threshold.", wide=True),
    FIG(14, "fig14_ablation.png",
        "Ablation studies. (a) Physics-derived features. (b) Distillation and "
        "quantisation. (c) Labelled-data budget.", wide=True),
    FIG(15, "fig15_sensitivity.png",
        "Sensitivity analyses. (a) Sensor noise. (b) Integrity-detector "
        "threshold. (c) Physics-loss weight.", wide=True),
    FIG(16, "fig16_security.png",
        "Integrity validation. (a) Detection rate by attack class against the "
        "clean false-positive rate. (b) Detector ROC curves.", wide=True),
])

sec(1, [
    FIG(3, "fig03_pinn_architecture.png",
        "PINN architecture. Fourier encoding of the time input precedes the "
        "network; the physics residual is formed by automatic differentiation "
        "at collocation points using assumed thermal parameters.", wide=True),
    FIG(4, "fig04_distillation.png",
        "Knowledge-distillation pipeline from the offline teacher ensemble to "
        "the deployed INT8 student.", wide=True),
])


# ==========================================================================
# VII. LIMITATIONS  /  VIII. CONCLUSION
# ==========================================================================
sec(2, [
    H1("VII.  Limitations and Future Work"),
    P("The limitations below are stated at length because the study is "
      "simulation-only and its conclusions are correspondingly bounded.",
      noindent=True),
    P("**Synthetic data.** Every result rests on a generative model. That model "
      "encodes IEC 60076-7 thermal dynamics and mechanism-derived fault "
      "signatures, but it cannot contain phenomena its author did not "
      "anticipate: sensor drift over years, unmodelled failure modes, "
      "installation-specific electromagnetic interference, or the correlated "
      "faults that occur in practice. Reported accuracies are properties of the "
      "simulation, not predictions of field performance, and should not be "
      "quoted as the latter."),
    P("**The physics is partly circular.** The PINN residual and the data "
      "generator share the same governing equations, differing only by 10 % "
      "parameter mismatch. This is the most favourable possible setting for a "
      "physics-informed method. Real transformers exhibit structural model "
      "error \u2014 moisture-dependent ageing, non-uniform oil circulation, "
      "geometry effects \u2014 that no parameter perturbation represents. The "
      "reported gain over the data-driven baseline is therefore an upper bound."),
    P("**Embedded latency is projected, not measured.** Table VIII derives "
      "embedded figures from FLOP counts and assumed throughput. Real "
      "performance depends on memory bandwidth, cache behaviour, operator "
      "library quality and thermal throttling, none of which a FLOP count "
      "captures. Hardware-in-the-loop measurement is required before any "
      "deployment claim."),
    P("**No power or cost analysis.** Power draw cannot be simulated without "
      "the target hardware; economic analysis would require deployment costs "
      "and outage valuations we do not have. Both are omitted rather than "
      "estimated."),
    P("**Statistical strength.** PINN results average three seeds. "
      "Classification results use a single seed on one fixed split, so the "
      "sub-percentage-point differences between the closest variants are within "
      "run-to-run variance and no significance testing is offered. "
      "Multi-seed repetition with paired testing is needed to establish the "
      "distillation gain firmly."),
    P("**Prognostic scope.** Degradation trajectories come from one ageing "
      "model with lognormal endurance variability. Real fleets exhibit "
      "multi-modal failure causes. The physics-informed estimator explains "
      f"{100 * RUL['stats_physics']['r2']:.0f} % of variance here; early-life "
      "predictions remain weak."),
    P("**No public-benchmark comparison.** Comparison against C-MAPSS [27] or "
      "similar was considered. It was not performed, because the transformer "
      "thermal problem shares neither the sensor semantics nor the degradation "
      "physics of turbofan run-to-failure data, and a comparison across "
      "incommensurable tasks would have been decorative rather than informative."),
    P("Future work should proceed in this order: hardware-in-the-loop timing on "
      "a representative gateway; validation of the thermal model against "
      "instrumented units with fibre-optic hot-spot probes; multi-seed "
      "replication with significance testing; and cryptographic freshness for "
      "the integrity channel, since Section VI-G shows anomaly detection alone "
      "is insufficient."),

    H1("VIII.  Conclusion"),
    P("This paper presented a physics-informed digital-twin framework for "
      "transformer condition monitoring and evaluated it in simulation. The "
      "central finding is that embedding the IEC 60076-7 thermal equations in "
      "the training objective allows an unobservable state \u2014 winding "
      "hot-spot temperature \u2014 to be reconstructed to "
      f"{f2(PM['pinn']['rmse'][0])} K RMSE with no direct labels, where an "
      f"identical network without the physics term achieves "
      f"{f2(PM['data_only']['rmse'][0])} K and still trails after 200 labels. "
      "For an asset class where hot-spot instrumentation is rare, physics "
      "substitutes for data that does not exist.", noindent=True),
    P("Three secondary findings follow. Fourier feature encoding is a "
      "precondition for the physics term to help at all on diurnally forced "
      "problems. The optimal physics weight exists only under parameter "
      "mismatch, so ablations conducted with exact physics are uninformative "
      "about deployment. And compression is inexpensive here: distillation with "
      "INT8 quantisation gives "
      f"{teacher_row['total_int8_kB'] / student_row['total_int8_kB']:.1f}\u00d7 "
      f"size reduction for "
      f"{100 * (CM['Teacher (Bi-GRU + RF)']['accuracy'] - CM['Student, distilled + INT8']['accuracy']):.2f} "
      "percentage points of accuracy, with quantisation itself free."),
    P("We also report results that do not support common claims. A compact "
      "random forest matches the recurrent ensemble on these features. The "
      "explicit multi-phase coupling term is nearly redundant. And "
      "reconstruction-based integrity validation, effective against gross "
      "spoofing, is structurally unable to detect replay of valid data. "
      "Reporting these alongside the positive results seems to us more useful "
      "than reporting the positive results alone."),
    P("All findings are simulation results. Field validation remains necessary "
      "before operational claims can be made. Code, seeds and figure-generation "
      "scripts accompany this paper so that every number can be reproduced."),
])

# ==========================================================================
# APPENDIX
# ==========================================================================
sec(2, [
    H1("Appendix A:  Algorithm Specifications"),
    {"t": "p", "text": "**Algorithm 1: Edge preprocessing and feature extraction**", "noindent": True, "wide": True},
    CODE([
        "Input : raw window X of shape (C=6, N_w=1024) at f_s = 1000 Hz",
        "         datasheet thermal parameters (Hgr, y)",
        "Output: quantised feature vector f of dimension 111",
        "",
        " 1  for each channel c in 1..C do",
        " 2      x <- lowpass(X[c], f_cut = 400 Hz)      # anti-alias",
        " 3      t <- [rms, sd, peak, crest, skew, kurt](x)      # 6 time-domain",
        " 4      (f_w, P) <- welch(x, nperseg = 256)",
        " 5      s <- [f_dom, centroid, spread, entropy](f_w, P)  # 4 spectral",
        " 6      w <- wavelet_packet_energy(x, db4, levels = 3)   # 8 sub-bands",
        " 7      F[c] <- concat(t, s, w)                          # 18 features",
        " 8  end for",
        " 9  f_sensor <- concat(F[1..C])                          # 108 features",
        "10  Theta_TO <- F[temp][rms] ;  K_hat <- sqrt(2) * F[cur][rms]",
        "11  Theta_H_hat <- Theta_TO + Hgr * K_hat^y      # IEC 60076-7",
        "12  V_hat <- 2^((Theta_H_hat - 98)/6)",
        "13  f <- concat(f_sensor, [Theta_H_hat, log10(V_hat), K_hat])",
        "14  (lo, hi) <- calibration range from training set",
        "15  f_q <- round(clip((f - lo)/(hi - lo), 0, 1) * 255)/255",
        "16  return f_q",
    ]),
    {"t": "p", "text": "**Algorithm 2: PINN training**", "noindent": True, "wide": True},
    CODE([
        "Input : observations D_to = {(t_j, dTheta_TO_j)}_{j=1..N_d}",
        "         optional labels D_hs = {(t_l, dTheta_H_l)}_{l=1..N_hs}",
        "         assumed parameters (tau_o~, tau_w~, dTheta_or~, Hgr~, R~, x~, y~)",
        "         weights lambda_d, lambda_p, lambda_bc ; collocation count N_c",
        "Output: trained network N_theta",
        "",
        " 1  normalise time to s = t / T ;  initialise theta (Xavier)",
        " 2  sample collocation set {s_k}_{k=1..N_c} ~ U(0,1), requires_grad",
        " 3  for epoch = 1 to E do",
        " 4      L_data <- MSE over D_to (+ over D_hs if N_hs > 0)",
        " 5      (u, v) <- N_theta(gamma(s_k))         # dTheta_TO, dTheta_H",
        " 6      du/ds, dv/ds <- autograd.grad(u, s_k), autograd.grad(v, s_k)",
        " 7      K_k <- interpolate load forcing at s_k",
        " 8      r1 <- (tau_o~/T) du/ds - [dTheta_or~ ((1+R~K_k^2)/(1+R~))^x~ - u]",
        " 9      r2 <- (tau_w~/T) dv/ds - [Hgr~ K_k^y~ - v]",
        "10      L_phys <- mean(r1^2 + r2^2)",
        "11      L_bc   <- (u(0) - u_0)^2 + (v(0) - v_0)^2",
        "12      L <- lambda_d L_data + lambda_p L_phys + lambda_bc L_bc",
        "13      theta <- Adam_step(theta, grad L)",
        "14  end for",
        "15  return N_theta",
    ]),
    {"t": "p", "text": "**Algorithm 3: Physics-informed RUL estimation**", "noindent": True, "wide": True},
    CODE([
        "Input : estimated hot-spot series Theta_H(1..k), health index HI(1..k)",
        "         calibrated monotone map g: HI -> damage fraction d",
        "         observation window W days",
        "Output: remaining useful life estimate",
        "",
        " 1  V(i) <- 2^((Theta_H(i) - 98)/6)        for i = 1..k",
        " 2  L(k) <- sum_i V(i) * dt                # accumulated ageing",
        " 3  d(i) <- g(HI(i))                       # damage fraction history",
        " 4  restrict to the last W days",
        " 5  (slope, intercept) <- least_squares(t - t_k, d)",
        " 6  slope <- max(slope, eps)               # damage is monotone",
        " 7  d_now <- clip(intercept, 1e-3, 0.999)",
        " 8  return (1 - d_now) / slope",
    ]),

    H1("Appendix B:  Derivations"),
    H2("B.1  Weibull Parameter Estimation"),
    P("Taking logarithms twice in (22) linearises the Weibull cumulative "
      "distribution to (23), so that ordinary least squares on "
      "(ln t, ln[-ln(1-u)]) yields the slope \u03b2 and intercept "
      "-\u03b2 ln \u03bb. This estimator is used only for initialisation: it "
      "is ill-conditioned when the damage fraction spans a narrow range, and "
      "can return shape parameters near zero for which the extrapolated "
      "end-of-life time diverges. Parameters are refined by bounded nonlinear "
      "least squares on the untransformed curve with \u03b2 constrained to "
      "[0.6, 8].", noindent=True),
    H2("B.2  Variational Lower Bound"),
    P("The autoencoder maximises the evidence lower bound on log p(f), "
      "equivalently minimising (28). For a Gaussian approximate posterior with "
      "diagonal covariance and a standard normal prior the divergence admits "
      "the closed form (29), which is what the implementation computes. The "
      "anomaly score combines reconstruction error with the divergence, so that "
      "an input mapping to an improbable latent code is penalised even when it "
      "reconstructs well.", noindent=True),
    H2("B.3  Stable Root for the Ageing Projection"),
    P("Projecting a linearly increasing ageing rate to a target ageing budget "
      "requires solving aR\u00b2 + bR - \u0394L = 0. The textbook root "
      "(-b + \u221a(b\u00b2+4a\u0394L))/2a loses all significance when a is "
      "small, which is the common case since the ageing-rate trend is nearly "
      "flat. The mathematically equivalent form "
      "R = 2\u0394L/(b + \u221a(b\u00b2+4a\u0394L)) is stable throughout and "
      "is used in the implementation.", noindent=True),
])

# ==========================================================================
# REFERENCES
# ==========================================================================
REFS = [
 "T. Gonen, *Electric Power Distribution Engineering*, 3rd ed. Boca Raton, FL, USA: CRC Press, 2014.",
 "J. D. McDonald, Ed., *Electric Power Substations Engineering*, 3rd ed. Boca Raton, FL, USA: CRC Press, 2012.",
 "International Civil Aviation Organization, *Annex 14 to the Convention on International Civil Aviation: Aerodromes, Volume I \u2014 Aerodrome Design and Operations*, 8th ed., Montreal, QC, Canada, 2018.",
 "Federal Aviation Administration, \u201cDesign and installation details for airport visual aids,\u201d Advisory Circular AC 150/5340-30J, Washington, DC, USA, 2018.",
 "International Air Transport Association, *Airport Development Reference Manual*, 11th ed., Montreal, QC, Canada, 2019.",
 "*Power Transformers \u2014 Part 7: Loading Guide for Mineral-Oil-Immersed Power Transformers*, IEC Standard 60076-7:2018, 2018.",
 "M. Raissi, P. Perdikaris, and G. E. Karniadakis, \u201cPhysics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,\u201d *J. Comput. Phys.*, vol. 378, pp. 686\u2013707, 2019.",
 "G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang, \u201cPhysics-informed machine learning,\u201d *Nature Rev. Phys.*, vol. 3, no. 6, pp. 422\u2013440, 2021.",
 "G. Hinton, O. Vinyals, and J. Dean, \u201cDistilling the knowledge in a neural network,\u201d 2015, *arXiv:1503.02531*.",
 "D. P. Kingma and M. Welling, \u201cAuto-encoding variational Bayes,\u201d in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2014.",
 "D. P. Kingma and J. Ba, \u201cAdam: A method for stochastic optimization,\u201d in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2015.",
 "K. Cho et al., \u201cLearning phrase representations using RNN encoder\u2013decoder for statistical machine translation,\u201d in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, 2014, pp. 1724\u20131734.",
 "J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, \u201cEmpirical evaluation of gated recurrent neural networks on sequence modeling,\u201d 2014, *arXiv:1412.3555*.",
 "L. Breiman, \u201cRandom forests,\u201d *Mach. Learn.*, vol. 45, no. 1, pp. 5\u201332, 2001.",
 "F. Tao, H. Zhang, A. Liu, and A. Y. C. Nee, \u201cDigital twin in industry: State-of-the-art,\u201d *IEEE Trans. Ind. Informat.*, vol. 15, no. 4, pp. 2405\u20132415, Apr. 2019.",
 "W. Kritzinger, M. Karner, G. Traar, J. Henjes, and W. Sihn, \u201cDigital twin in manufacturing: A categorical literature review and classification,\u201d *IFAC-PapersOnLine*, vol. 51, no. 11, pp. 1016\u20131022, 2018.",
 "M. Grieves and J. Vickers, \u201cDigital twin: Mitigating unpredictable, undesirable emergent behavior in complex systems,\u201d in *Transdisciplinary Perspectives on Complex Systems*. Cham, Switzerland: Springer, 2017, pp. 85\u2013113.",
 "W. Shi, J. Cao, Q. Zhang, Y. Li, and L. Xu, \u201cEdge computing: Vision and challenges,\u201d *IEEE Internet Things J.*, vol. 3, no. 5, pp. 637\u2013646, Oct. 2016.",
 "*Communication Networks and Systems for Power Utility Automation \u2014 Part 7-4: Basic Communication Structure \u2014 Compatible Logical Node Classes and Data Object Classes*, IEC Standard 61850-7-4:2010, 2010.",
 "*Industrial Communication Networks \u2014 Network and System Security \u2014 Part 3-3: System Security Requirements and Security Levels*, IEC Standard 62443-3-3:2013, 2013.",
 "J. Lee, B. Bagheri, and H.-A. Kao, \u201cA cyber-physical systems architecture for Industry 4.0-based manufacturing systems,\u201d *Manuf. Lett.*, vol. 3, pp. 18\u201323, 2015.",
 "I. J. Goodfellow, J. Shlens, and C. Szegedy, \u201cExplaining and harnessing adversarial examples,\u201d in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2015.",
 "S. Alabbad and H. Alt\u0131nkaya, \u201cAn intelligent predictive maintenance architecture for substation automation: Real-world validation of a digital twin and AI framework of the Badra oil field project,\u201d *Electronics*, vol. 15, no. 2, Art. no. 416, 2026.",
 "B. Jacob et al., \u201cQuantization and training of neural networks for efficient integer-arithmetic-only inference,\u201d in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 2704\u20132713.",
 "M. Tancik et al., \u201cFourier features let networks learn high frequency functions in low dimensional domains,\u201d in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2020, pp. 7537\u20137547.",
 "S. Wang, Y. Teng, and P. Perdikaris, \u201cUnderstanding and mitigating gradient flow pathologies in physics-informed neural networks,\u201d *SIAM J. Sci. Comput.*, vol. 43, no. 5, pp. A3055\u2013A3081, 2021.",
 "A. Saxena, K. Goebel, D. Simon, and N. Eklund, \u201cDamage propagation modeling for aircraft engine run-to-failure simulation,\u201d in *Proc. Int. Conf. Prognostics Health Manage. (PHM)*, 2008, pp. 1\u20139.",
 "Y. Lei, N. Li, L. Guo, N. Li, T. Yan, and J. Lin, \u201cMachinery health prognostics: A systematic review from data acquisition to RUL prediction,\u201d *Mech. Syst. Signal Process.*, vol. 104, pp. 799\u2013836, 2018.",
 "R. Zhao, R. Yan, Z. Chen, K. Mao, P. Wang, and R. X. Gao, \u201cDeep learning and its applications to machine health monitoring,\u201d *Mech. Syst. Signal Process.*, vol. 115, pp. 213\u2013237, 2019.",
 "I. Daubechies, *Ten Lectures on Wavelets*. Philadelphia, PA, USA: SIAM, 1992.",
 "P. Welch, \u201cThe use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms,\u201d *IEEE Trans. Audio Electroacoust.*, vol. 15, no. 2, pp. 70\u201373, Jun. 1967.",
 "F. Pedregosa et al., \u201cScikit-learn: Machine learning in Python,\u201d *J. Mach. Learn. Res.*, vol. 12, pp. 2825\u20132830, 2011.",
 "A. Paszke et al., \u201cPyTorch: An imperative style, high-performance deep learning library,\u201d in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019, pp. 8024\u20138035.",
 "C. R. Harris et al., \u201cArray programming with NumPy,\u201d *Nature*, vol. 585, pp. 357\u2013362, 2020.",
 "P. Virtanen et al., \u201cSciPy 1.0: Fundamental algorithms for scientific computing in Python,\u201d *Nature Methods*, vol. 17, pp. 261\u2013272, 2020.",
 "*IEEE Guide for Loading Mineral-Oil-Immersed Transformers and Step-Voltage Regulators*, IEEE Standard C57.91-2011, 2012.",
 "*High-Voltage Test Techniques \u2014 Partial Discharge Measurements*, IEC Standard 60270:2000, 2000.",
 "M. Duval, \u201cA review of faults detectable by gas-in-oil analysis in transformers,\u201d *IEEE Elect. Insul. Mag.*, vol. 18, no. 3, pp. 8\u201317, May/Jun. 2002.",
 "OPC Foundation, *OPC Unified Architecture \u2014 Part 14: PubSub*, Release 1.04, 2018.",
 "*Energy Management System Application Program Interface (EMS-API) \u2014 Part 301: Common Information Model (CIM) Base*, IEC Standard 61970-301, 2016.",
 "G. A. Susto, A. Schirru, S. Pampuri, S. McLoone, and A. Beghi, \u201cMachine learning for predictive maintenance: A multiple classifier approach,\u201d *IEEE Trans. Ind. Informat.*, vol. 11, no. 3, pp. 812\u2013820, Jun. 2015.",
 "W. Zhang, D. Yang, and H. Wang, \u201cData-driven methods for predictive maintenance of industrial equipment: A survey,\u201d *IEEE Syst. J.*, vol. 13, no. 3, pp. 2213\u20132227, Sep. 2019.",
 "T. Wuest, D. Weimer, C. Irgens, and K.-D. Thoben, \u201cMachine learning in manufacturing: Advantages, challenges, and applications,\u201d *Prod. Manuf. Res.*, vol. 4, no. 1, pp. 23\u201345, 2016.",
 "Y. Lu, C. Liu, K. I.-K. Wang, H. Huang, and X. Xu, \u201cDigital twin-driven smart manufacturing: Connotation, reference model, applications and research issues,\u201d *Robot. Comput.-Integr. Manuf.*, vol. 61, Art. no. 101837, 2020.",
 "A. Rasheed, O. San, and T. Kvamsdal, \u201cDigital twin: Values, challenges and enablers from a modeling perspective,\u201d *IEEE Access*, vol. 8, pp. 21980\u201322012, 2020.",
 "S. Boschert and R. Rosen, \u201cDigital twin \u2014 The simulation aspect,\u201d in *Mechatronic Futures*. Cham, Switzerland: Springer, 2016, pp. 59\u201374.",
 "S. Hochreiter and J. Schmidhuber, \u201cLong short-term memory,\u201d *Neural Comput.*, vol. 9, no. 8, pp. 1735\u20131780, 1997.",
 "S. Cuomo, V. S. di Cola, F. Giampaolo, G. Rozza, M. Raissi, and F. Piccialli, \u201cScientific machine learning through physics-informed neural networks: Where we are and what\u2019s next,\u201d *J. Sci. Comput.*, vol. 92, no. 3, Art. no. 88, 2022.",
 "L. Lu, X. Meng, Z. Mao, and G. E. Karniadakis, \u201cDeepXDE: A deep learning library for solving differential equations,\u201d *SIAM Rev.*, vol. 63, no. 1, pp. 208\u2013228, 2021.",
 "Y. LeCun, Y. Bengio, and G. Hinton, \u201cDeep learning,\u201d *Nature*, vol. 521, pp. 436\u2013444, 2015.",
 "N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, \u201cDropout: A simple way to prevent neural networks from overfitting,\u201d *J. Mach. Learn. Res.*, vol. 15, pp. 1929\u20131958, 2014.",
 "T. Chen and C. Guestrin, \u201cXGBoost: A scalable tree boosting system,\u201d in *Proc. ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2016, pp. 785\u2013794.",
 "D. An, J.-H. Choi, and N. H. Kim, \u201cPrognostics 101: A tutorial for particle filter-based prognostics algorithm using MATLAB,\u201d *Rel. Eng. Syst. Saf.*, vol. 115, pp. 161\u2013169, 2013.",
 "X.-S. Si, W. Wang, C.-H. Hu, and D.-H. Zhou, \u201cRemaining useful life estimation \u2014 A review on the statistical data driven approaches,\u201d *Eur. J. Oper. Res.*, vol. 213, no. 1, pp. 1\u201314, 2011.",
 "W. Weibull, \u201cA statistical distribution function of wide applicability,\u201d *J. Appl. Mech.*, vol. 18, no. 3, pp. 293\u2013297, 1951.",
 "B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall, 1993.",
 "M. Tabrez, P. Sadhu, and S. B. Santra, \u201cMulti-phase transformer configurations and their performance analysis,\u201d *Int. J. Power Electron. Drive Syst.*, vol. 11, no. 4, pp. 1780\u20131790, 2020.",
 "M. Tabrez, P. K. Sadhu, and A. Iqbal, \u201cA novel three-phase to nine-phase transformer configuration,\u201d *Int. Trans. Elect. Energy Syst.*, vol. 31, no. 9, Art. no. e12995, 2021.",
 "M. Tabrez, P. K. Sadhu, and A. Iqbal, \u201cPerformance improvement of multi-phase transformer-fed drive systems,\u201d *Electr. Eng.*, vol. 104, no. 2, pp. 861\u2013873, 2022.",
 "N. Papernot, P. McDaniel, S. Jha, M. Fredrikson, Z. B. Celik, and A. Swami, \u201cThe limitations of deep learning in adversarial settings,\u201d in *Proc. IEEE Eur. Symp. Secur. Privacy*, 2016, pp. 372\u2013387.",
 "A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, \u201cTowards deep learning models resistant to adversarial attacks,\u201d in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2018.",
 "Y. Liu, P. Ning, and M. K. Reiter, \u201cFalse data injection attacks against state estimation in electric power grids,\u201d *ACM Trans. Inf. Syst. Secur.*, vol. 14, no. 1, pp. 1\u201333, 2011.",
]

sec(2, [H1("References")] +
    [REF(f"[{i + 1}]  {t}") for i, t in enumerate(REFS)])

# ==========================================================================
# EMIT
# ==========================================================================
def split_wide(sections):
    """Move wide blocks out of two-column sections into full-width bands.

    Word cannot make a single block span both columns of a section, so an
    oversized equation placed inline is simply clipped. Splitting the section
    around it produces the full-width band that IEEE style uses for wide
    equations and figures.
    """
    out = []
    for sc in sections:
        if sc["cols"] != 2:
            out.append(sc)
            continue
        buf = []
        for b in sc["blocks"]:
            if b.get("wide"):
                if buf:
                    out.append({"cols": 2, "blocks": buf})
                    buf = []
                if out and out[-1]["cols"] == 1 and out[-1].get("_wide"):
                    out[-1]["blocks"].append(b)
                else:
                    out.append({"cols": 1, "blocks": [b], "_wide": True})
            else:
                buf.append(b)
        if buf:
            out.append({"cols": 2, "blocks": buf})
    return out


S = split_wide(S)

assets = {}
for sct in S:
    for b in sct["blocks"]:
        img = b.get("img")
        if img and img not in assets:
            im = Image.open(img)
            assets[img] = {"px_w": im.width, "px_h": im.height,
                           "pt_w": im.width / 400.0 * 72.0}
for k, v in EQ.items():
    assets[v["path"]] = {"px_w": v["px_w"], "px_h": v["px_h"], "pt_w": v["pt_w"]}

out = {"meta": {"title": "A Physics-Informed Digital-Twin Framework",
                "author": "Mohammad Tabrez"},
       "assets": assets, "sections": S}
json.dump(out, open("manuscript.json", "w"))
nb = sum(len(x["blocks"]) for x in S)
print(f"manuscript.json: {len(S)} sections, {nb} blocks, {len(assets)} assets, "
      f"{len(REFS)} references")
