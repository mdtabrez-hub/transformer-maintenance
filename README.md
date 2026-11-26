# Physics-informed digital twin for transformer predictive maintenance

Simulation code, seeds, generated datasets, trained model weights and
figure-generation scripts supporting:

> Md Tabrez, Onkar Singh, Mohammad Aknan, Pritam Kumar, Md Shadab Hussain and Md Mazharul Haque,
> "Physics-Informed Digital Twin With Distilled Edge Inference for Predictive
> Maintenance of Substation Transformers: A Model-Based Validation,"
> IEEE Access, manuscript Access-2026-53764 (under review).

See `README_R1.md` for the experiments added in the first revision (R1);
several of them supersede numbers in the table below.

**This is a simulation study.** No physical sensors were installed and no field
data were collected. Every number in the paper is produced by the code here from
the master seed `20260905`.

---

## What the study finds

| Result | Value |
|---|---|
| Hot-spot RMSE, PINN (λ_p = 10) | **1.31 ± 0.19 K** (R² = 0.984) |
| Hot-spot RMSE, same network without physics residual | 13.72 ± 2.25 K (R² = −0.731) |
| Hot-spot labels needed by PINN to reach 1.55 K | **0** |
| Hot-spot labels needed by data-driven model to reach 1.89 K | 200 |
| Incipient-fault accuracy, teacher / distilled student | 88.46 % / 88.03 % |
| Physics-feature contribution | +6.35 pp (78.57 % → 84.92 %) |
| Compression, teacher → INT8 student | 8.4× params, 10.5× stored size, −0.43 pp accuracy |
| Upstream bandwidth reduction | 95.33 % |
| RUL, physics-informed vs. Weibull baseline | 86.7 d vs. 283.3 d MAE (R² 0.666 vs. −2.844) |

Negative results are reported alongside these: a compact random forest matches
the recurrent ensemble; the explicit inter-phase coupling term is nearly
redundant (−2.8 % imbalance RMSE); and the autoencoder integrity validator
detects gross spoofing (97.4 %) but cannot detect replay of valid data (12.0 %),
which is structural rather than a tuning deficiency.

---

## Layout

```
code/           simulation framework and manuscript build scripts
  sim/core.py           IEC 60076-7 thermal model, sensor synthesis, fault
                        mechanisms, feature extraction, fleet degradation
  sim/models.py         PINN, GRU/CNN-LSTM, distillation, VAE, RUL estimators
  sim/experiments.py    experiments E1-E13
  run_all.py            staged orchestrator
  export_artifacts.py   regenerates datasets and trained weights
  make_figures.py       all 16 figures
  make_fig06.py         Fig. 6 at 600 dpi
  make_equations.py     31 numbered equations
  make_manuscript.py    manuscript content -> manuscript.json
  build_docx.js         manuscript.json -> IEEE two-column .docx
artifacts/      generated datasets and trained weights, with MANIFEST.json
figures/        16 figures, PNG (400-600 dpi) and PDF
results/        results.json — every computed number in the paper
paper/          summary of changes from the earlier draft
```

## Install

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build
```

## Reproduce

Stages exist because a full run exceeds typical single-process time limits;
`--stage all` runs everything in one go.

```bash
cd code
python3 run_all.py --stage A1    # PINN hot-spot reconstruction, 3 seeds
python3 run_all.py --stage A2    # physics-loss weight ablation
python3 run_all.py --stage A3    # label-efficiency sweep
python3 run_all.py --stage A4    # inter-phase coupling ablation
python3 run_all.py --stage B1    # classification, security, edge cost, bandwidth, placement
python3 run_all.py --stage B2    # physics-feature ablation
python3 run_all.py --stage C1    # sensor-noise sensitivity
python3 run_all.py --stage C2    # remaining useful life
python3 run_all.py --stage merge --out results/results.json
```

Add `--quick` for a reduced-budget smoke test. Generated feature datasets are
cached as `.npz`; delete them to regenerate from scratch.

Figures and manuscript:

```bash
python3 make_equations.py
python3 make_figures.py
python3 make_manuscript.py
node build_docx.js manuscript.json manuscript.docx
```

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

## Reproducibility notes

Results are deterministic given the seed and the library versions recorded in
`results/results.json` (`meta`) and `artifacts/MANIFEST.json`. Two caveats a
reader should know:

- **PINN results average three seeds; classification results use a single seed
  on one fixed split.** Differences below roughly one percentage point between
  the closest classifier variants are within run-to-run variance, and no
  significance testing is reported.
- **Embedded latency figures in the paper are analytical projections**, derived
  from measured FLOP counts and an assumed sustained throughput. They are not
  hardware measurements. Host latency is measured.

## Licence

MIT — see `LICENSE`.

## Contact

Md Tabrez · Motihari College of Engineering, Bihar Engineering University
ORCID [0000-0002-0829-1258](https://orcid.org/0000-0002-0829-1258)
