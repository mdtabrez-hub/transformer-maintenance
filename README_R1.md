# Revision 1 (Access-2026-53764): added experiments

Every number added or changed in the R1 manuscript is produced by the scripts
below. Master seed: `20260905` (sim/core.py). Python 3.12, PyTorch 2.15.0+cpu,
NumPy 2.x, SciPy, scikit-learn, pandas, matplotlib, networkx (see requirements.txt).

| Script | Manuscript item | Output |
|---|---|---|
| `code/run_r1_thermal.py --workers 2` | Tables VII-VIII, Fig. 6(a)-(b), one-at-a-time mismatch | `results/r1_thermal.jsonl` |
| `code/sim/r1_thermal.py` | generators S1-S5, Kalman/EKF/gray-box estimators, identifiability (Table IX) | (library) |
| `code/run_r1_multiphase.py` | Fig. 6(c), three-phase coupling (N_ph = 3) | `results/r1_multiphase.json` |
| `code/run_r1_clf.py --stage cv` | Table X, Fig. 14(a)-(b), paired tests | `results/r1_clf_cv.json` |
| `code/run_r1_clf.py --stage size` | Fig. 14(c) | `results/r1_clf_size.json` |
| `code/run_r1_clf.py --stage sev` | cross-severity test (Section VI-C) | `results/r1_clf_sev.json` |
| `code/run_r1_clf.py --stage shift` | Table XI, mixed faults | `results/r1_clf_shift.json` |
| `code/run_r1_edge.py` | Table XII RF row, host pipeline timing | `results/r1_edge.json` |
| `code/run_r1_rul.py` | Tables XIII-XIV | `results/r1_rul.json` |
| `code/run_r1_identifiability.py` | Table IX | `results/ident.json` |
| `code/analyse_r1_cv.py` | Table X statistics, severity terciles | `results/cv_summary.json` |
| `code/make_tables_r1.py` | Tables VII-XIV numbers; runs the conventional estimators | `results/r1_numbers.json` |
| `code/make_figures_r1.py` | Figs. 6 and 14 (reads the two files above) | `figures/` |
| `code/make_figures.py` | Fig. 4 (corrected) and all other figures | `figures/` |

Seeds: PINN runs use trajectory seed 100+s, training seed s and parameter-mismatch
seed s+991 for s = 0..4. Cross-validation uses StratifiedKFold(random_state=20260905+rep)
for rep = 0, 1, an inner validation split with random_state 20260905+10*rep+fold,
and training seed 5*rep+fold. Shifted test sets use seeds 20260905+500+i (conditions)
and 20260905+700+i (mixed faults).

## Corrections relative to the submitted version

* The released classification dataset was generated with SNR 8 dB, severity
  range [0.03, 0.32] and calibration errors of 12 % / 8 % (the `SimConfig`
  defaults). `results/results.json` records 12 dB / [0.05, 0.45] / 8 % / 5 % in
  its metadata because the classification stage reused a cached dataset; the
  manuscript has been corrected to the values that produced the data.
* `exp_pinn_multiphase` used `n_phases = 6`; it is repeated for 3 phases in
  `run_r1_multiphase.py`.
* `LossOfLifeRUL.predict` computes the accumulated ageing but does not use it in
  the prediction; the manuscript description (Section IV-F, Algorithm 3) has been
  corrected accordingly.

## Run order

```bash
pip install -r requirements.txt
cd code
python3 run_r1_thermal.py --workers 2      # ~1.5 h on 2 CPU cores (244 PINN/NN runs, resumable)
python3 run_r1_multiphase.py
python3 run_r1_clf.py --stage cv            # then --stage size, sev, shift
python3 run_r1_rul.py
python3 run_r1_edge.py                      # host timings: will differ on your machine
python3 run_r1_identifiability.py
python3 analyse_r1_cv.py
python3 make_tables_r1.py
python3 make_figures_r1.py ../figures
```

All result files from the authors' run are already in `results/`, so the last
four commands can be run directly to regenerate the tables and figures.
`code/make_fig06.py` produces the Fig. 6 of the submitted (R0) version and is
kept for the record; `make_figures_r1.py` supersedes it.
