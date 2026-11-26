# Physics-informed digital twin for transformer predictive maintenance

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24547690.svg)](https://doi.org/10.5281/zenodo.24547690)

Simulation code, seeds, generated datasets, trained model weights, result files
and figure scripts supporting:

> Md Tabrez, Onkar Singh, Mohammad Aknan, Pritam Kumar, Md Shadab Hussain and Md Mazharul Haque,
> "Physics-Informed Digital Twin With Distilled Edge Inference for Predictive
> Maintenance of Substation Transformers: A Model-Based Validation,"
> IEEE Access, manuscript Access-2026-53764 (under review).

**This is a simulation study.** No physical sensors were installed and no field
data were collected. Every number in the paper is produced by the code here from
the master seed `20260905`.

---

## What the study finds

| Result | Value |
|---|---|
| Hot-spot RMSE, PINN (10 % parameter mismatch) | 1.72 K |
| Hot-spot RMSE, same network without physics residual | 13.62 K |
| Hot-spot RMSE, identified gray-box IEC model | **0.22 K** |
| Hot-spot RMSE, joint-parameter extended Kalman filter | 0.35 K |
| Fault classification, random forest / teacher ensemble (2 × 5 CV) | 89.0 % / 89.0 % |
| Fault classification, distilled INT8 student | 86.6 % |
| Upstream bandwidth reduction from edge feature extraction | 95.3 % |

Conventional estimators that use the same model and inputs beat the PINN, and
the gray-box model leads under four of five structurally different ground-truth
models. Top-oil data cannot observe the hot-spot gradient (see the
identifiability analysis). Distillation gives no significant gain, physics-derived
features do not raise accuracy, and all classifiers degrade sharply under shifted
operating conditions. The reconstruction-based validator detects gross spoofing
but not replay.

---

## Layout

```
code/
  sim/core.py                 IEC 60076-7 thermal model, sensor synthesis, fault
                              mechanisms, feature extraction, fleet degradation
  sim/models.py               PINN, GRU/CNN-LSTM, distillation, VAE, RUL estimators
  sim/experiments.py          base experiments E1-E13
  sim/r1_thermal.py           ground-truth generators S1-S5; Kalman, EKF and
                              gray-box estimators; identifiability
  sim/r1_clf.py               repeated cross-validation and shift tests
  run_all.py                  staged orchestrator for the base experiments
  run_r1_*.py                 thermal benchmark, multiphase, classification,
                              edge, RUL and identifiability experiments
  analyse_r1_cv.py            cross-validation statistics and paired tests
  make_tables_r1.py           all table numbers -> results/r1_numbers.json
  make_figures.py             figures
  make_figures_r1.py          Figs. 6 and 14
  export_artifacts.py         regenerates datasets and trained weights
artifacts/      generated datasets and trained weights, with MANIFEST.json
figures/        figures, PNG and PDF
results/        result files for every number in the paper
```

## Install

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build
```

Python 3.12, PyTorch 2.15.0+cpu, NumPy 2.x, SciPy, scikit-learn, pandas,
matplotlib, networkx.

## Which script produces which result

| Script | Paper item | Output |
|---|---|---|
| `run_r1_thermal.py --workers 2` | Tables VII-VIII, Fig. 6(a)-(b), one-at-a-time mismatch | `results/r1_thermal.jsonl` |
| `run_r1_multiphase.py` | Fig. 6(c), three-phase coupling | `results/r1_multiphase.json` |
| `run_r1_identifiability.py` | Table IX | `results/ident.json` |
| `run_r1_clf.py --stage cv` | Table X, Fig. 14(a)-(b), paired tests | `results/r1_clf_cv.json` |
| `run_r1_clf.py --stage size` | Fig. 14(c) | `results/r1_clf_size.json` |
| `run_r1_clf.py --stage sev` | cross-severity test (Section VI-C) | `results/r1_clf_sev.json` |
| `run_r1_clf.py --stage shift` | Table XI, mixed faults | `results/r1_clf_shift.json` |
| `run_r1_edge.py` | Table XII, host pipeline timing | `results/r1_edge.json` |
| `run_r1_rul.py` | Tables XIII-XIV | `results/r1_rul.json` |
| `analyse_r1_cv.py` | Table X statistics, severity terciles | `results/cv_summary.json` |
| `make_tables_r1.py` | numbers of Tables VII-XIV | `results/r1_numbers.json` |
| `make_figures_r1.py` | Figs. 6 and 14 | `figures/` |
| `run_all.py` + `make_figures.py` | remaining tables and figures | `results/results.json`, `figures/` |

## Reproduce

```bash
cd code
python3 run_r1_thermal.py --workers 2      # ~1.5 h on 2 CPU cores (resumable)
python3 run_r1_multiphase.py
python3 run_r1_clf.py --stage cv            # then --stage size, sev, shift
python3 run_r1_rul.py
python3 run_r1_edge.py                      # host timings will differ on your machine
python3 run_r1_identifiability.py
python3 analyse_r1_cv.py
python3 make_tables_r1.py
python3 make_figures_r1.py ../figures
```

All result files from the authors' run are already in `results/`, so the last
four commands can be run on their own to regenerate the tables and figures.

The base experiments run in stages because a full run exceeds typical
single-process time limits (`--stage all` runs everything; `--quick` is a
reduced-budget smoke test):

```bash
python3 run_all.py --stage A1    # PINN hot-spot reconstruction
python3 run_all.py --stage A2    # physics-loss weight ablation
python3 run_all.py --stage A3    # label-efficiency sweep
python3 run_all.py --stage A4    # inter-phase coupling ablation
python3 run_all.py --stage B1    # classification, security, edge cost, bandwidth, placement
python3 run_all.py --stage B2    # physics-feature ablation
python3 run_all.py --stage C1    # sensor-noise sensitivity
python3 run_all.py --stage C2    # remaining useful life
python3 run_all.py --stage merge --out results/results.json
python3 make_figures.py
```

## Seeds

Master seed `20260905` (`sim/core.py`). PINN runs use trajectory seed 100+s,
training seed s and parameter-mismatch seed s+991 for s = 0..4.
Cross-validation uses `StratifiedKFold(random_state=20260905+rep)` for rep = 0, 1,
an inner validation split with `random_state=20260905+10*rep+fold`, and training
seed `5*rep+fold`. Shifted test sets use seeds 20260905+500+i (conditions) and
20260905+700+i (mixed faults).

## Artifacts

`artifacts/` holds the generated datasets and trained weights, with
`MANIFEST.json` giving SHA-256 digests and the library versions used. Regenerate
them with:

```bash
cd code && python3 export_artifacts.py --outdir ../artifacts
```

Loading a saved classifier:

```python
import torch
from sim.models import GRUClassifier

ck = torch.load("artifacts/weights/student_gru_kd.pt", weights_only=False)
a = ck["arch"]
m = GRUClassifier(a["d_in"], a["n_classes"], hidden=a["hidden"],
                  layers=a["layers"], bidir=a["bidir"], feat_dim=a["feat_dim"])
m.load_state_dict(ck["state_dict"])
m.eval()
```

## Notes

- The classification dataset was generated with SNR 8 dB, severity range
  [0.03, 0.32] and calibration errors of 12 % / 8 % (the `SimConfig` defaults).
  The metadata in `results/results.json` records 12 dB / [0.05, 0.45] / 8 % / 5 %
  because that stage reused a cached dataset; the defaults are the values that
  produced the data.
- `exp_pinn_multiphase` in `sim/experiments.py` uses `n_phases = 6`; the
  three-phase result in the paper comes from `run_r1_multiphase.py`.
- Embedded latency figures are analytical projections from measured FLOP counts
  and an assumed sustained throughput, not hardware measurements. Host latency
  is measured.

## How to cite

> M. Tabrez, O. Singh, M. Aknan, P. Kumar, M. S. Hussain, and M. M. Haque, "transformer-maintenance: Simulation code and data for a physics-informed digital twin of substation transformers," Zenodo, ver. v1.1.1-R1, Nov. 2026, doi: 10.5281/zenodo.24547690.

DOI of all versions (always resolves to the latest): [10.5281/zenodo.24547690](https://doi.org/10.5281/zenodo.24547690).

## Licence

MIT — see `LICENSE`.

## Contact

Md Tabrez · Motihari College of Engineering, Bihar Engineering University ·
ORCID [0000-0002-0829-1258](https://orcid.org/0000-0002-0829-1258)
