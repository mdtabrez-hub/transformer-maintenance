# Summary of changes

Revision of *"Edge-AI Enabled Physics-Informed Digital Twin for Predictive
Maintenance of Multi-Phase Airport Electrical Infrastructure"* into a
simulation-only study with computed results.

New title: **"A Physics-Informed Digital-Twin Framework With Distilled Edge
Inference for Predictive Maintenance of Multi-Phase Substation Transformers:
A Simulation Study"**

18 pages · ~8,900 words · 16 figures · 10 tables · 31 numbered equations ·
62 references

---

## 1. The central change

The original manuscript claimed a 12-month pilot deployment while a bracketed
author note admitted the results were simulated. **None of the reported numbers
had any computation behind them** — no code, no dataset, no model.

Carrying those numbers into a paper that claims to be a reproducible simulation
study would have been worse than the original, because the original at least
flagged the problem. So every number was recomputed by writing and running the
simulation. Master seed `20260905`; the full study reproduces exactly.

---

## 2. Deleted

| Content | Reason |
|---|---|
| §5.1 Pilot Deployment Site | No deployment occurred |
| §5.2 Field Data Collection Protocol | No data were collected |
| §6.5 Twelve-Month Validation | No such validation |
| §6.6 Cost-Benefit Analysis | Requires deployment cost and outage valuations that do not exist |
| Table 2 (sensor cost breakdown) | Hardware never procured |
| Table 10 (cost-benefit) | As above |
| All hardware references (Raspberry Pi, ESP32, LoRa, sensor part numbers) | Nothing was built |
| **All power-consumption claims (1.9 W etc.)** | Power cannot be simulated without the target hardware |
| Outage-reduction and maintenance-labour figures | Unfounded |
| The `[FLAG: CRITICAL]` note | Replaced by explicit simulation-only framing throughout |

Power is replaced by a computational-cost breakdown in MFLOP (Fig. 11a), from
which power can be estimated once a target board is chosen.

---

## 3. Numbers that changed, and why

| Quantity | Original (asserted) | Revised (computed) |
|---|---|---|
| Hot-spot RMSE, PINN | not reported | **1.31 ± 0.19 K** (R² = 0.984) |
| Hot-spot RMSE, no physics | not reported | **13.72 ± 2.25 K** (R² = −0.731) |
| Fault classification accuracy | 93.2 % | **88.46 %** teacher / **88.03 %** distilled student |
| Bandwidth reduction | 74.3 % | **95.33 %** |
| RUL mean absolute error | 12.3 days | **86.7 days** (R² = 0.666) |
| Attack detection | 94–99 % across all types | **97.4 %** temperature spoofing … **12.0 %** replay |
| Multi-phase coupling gain | +4.7 pp | **2.9 % relative** reduction in phase-imbalance RMSE |
| Inference latency | 3.8 s (claimed measured) | **0.28 ms** measured on host; **0.49 ms / 13.0 ms** *projected* for Cortex-A72 / M4F |

Two of these deserve comment.

**Bandwidth went up, not down.** Explicit byte accounting (6,144 B/window raw
versus 132 B for a quantised feature vector plus header) gives a larger
reduction than the paper claimed. The computed figure is reported rather than
one reverse-engineered to match.

**Attack detection went down sharply, and that is the interesting part.** The
original claimed 99.2 % for replay detection. A reconstruction-based detector
asks whether an observation resembles normal data; replayed data *is* normal
data. The detector cannot answer that question, and no threshold tuning fixes
it. The revision reports 12.0 % and argues that freshness requires cryptographic
sequence numbers, not anomaly detection. This is now framed as a contribution.

---

## 4. Bugs found and fixed while building the simulation

These were real defects in the modelling, not cosmetic issues. Each changed the
results.

1. **Latent-state leakage in the physics features.** The classifier was being
   handed the generator's hidden variables (`K`, `Θ_A`, `cooling`) — effectively
   the answer. Physics features now derive from observables only (measured
   top-oil temperature and current RMS) with IEC 60076-7 applied using datasheet
   parameters. This dropped accuracy from an implausible 98.7 % into a credible
   range.

2. **The temperature channel exposed the hot-spot.** The quantity the PINN
   exists to infer was directly measurable. The channel now carries top-oil
   temperature, which is what is actually instrumented.

3. **Sensor noise referenced to total signal power.** This assigned ±17 K of
   noise to a 45 °C reading. AC channels now use noise referenced to the
   fluctuating component; the temperature channel gets absolute instrument noise
   (0.3 K) plus a calibration offset.

4. **Spectral bias made the PINN nearly useless.** A plain tanh network reached
   8.5 K RMSE with physics versus 8.7 K without — the physics term did almost
   nothing because the network could not represent the diurnal forcing. Fourier
   feature encoding of the time input fixed this (1.31 K). This is now reported
   as a precondition, not an implementation detail.

5. **The physics-weight ablation was meaningless.** With the residual using
   exactly the constants that generated the data, accuracy improved monotonically
   with λ_p — no trade-off, no optimum. Adding 10 % parameter mismatch (the
   uncertainty attaching to test-report values) produced a genuine interior
   optimum at **λ_p = 10**.

6. **Three separate RUL failures.** An uncalibrated end-of-life threshold, an
   ill-conditioned log-linear Weibull fit, and catastrophic cancellation in a
   quadratic solve. Fixed respectively by calibrating the threshold on disjoint
   units, bounded nonlinear least squares, and the numerically stable root form.

7. **Random Forest size overestimated by 200×** — the `2^depth` bound assumed
   complete trees. Now computed from actual `node_count` (98 MB → 496 kB).

---

## 5. Reference corrections

- **[18] was miscited as "Anonymous", IEEE TII.** It is Alabbad & Altınkaya,
  *Electronics*, vol. 15, no. 2, Art. 416, 2026 (doi:10.3390/electronics15020416).
  Corrected and now cited as the closest prior art.
- **[67] Hong et al. removed** — page range "199901–199999" is impossible and the
  title duplicated [21].
- **Citation–content mismatch fixed:** §2.3 attributed the edge-fog-cloud digital
  twin to Luo et al.; that is Ren et al. Luo is a CNC machine-tool paper.
- Duplicates consolidated: [11]=[14], [15]=[36], [41]=[43], [42]=[44], [34]=[49].
- Non-research references removed: Matplotlib, Pandas, Géron, ResNet, Rumelhart.
- Author's prior multi-phase transformer work now cited in Section IV-B where the
  per-phase loss distribution is used.

> ⚠️ **Verify bibliographic details before submission.** Volume, page and year
> fields were reconstructed from memory, not from the publisher records. The
> works cited are real and the attributions are correct, but the numeric details
> need checking against the originals.

---

## 6. Reframed contribution

Per your instruction, novelty is now claimed for the **PINN formulation and edge
deployment**, not the architecture.

The removed claim ("first academic framework...") could not survive: Alabbad &
Altınkaya describe a five-layer DT-PdM architecture with IEC 61850 + CIM +
OPC UA Part 17 semantics and IEC 62443 security, validated on an operating
network. Section I-B now states explicitly that layered standards-aligned
architecture is no longer novel, and Section II-A adopts rather than claims it.

The five contributions are now: the PINN latent-state formulation; the
physics-versus-data trade-off under parameter mismatch; the label-efficiency
result; the compression cost-benefit analysis with measured/projected separation;
and the negative security result.

---

## 7. Negative results retained

Reported rather than suppressed:

- A compact random forest (88.03 %) matches the recurrent ensemble — recurrence
  earns little on these features.
- The multi-phase coupling term is nearly redundant (2.9 % gain), because the
  per-phase residuals already encode the shared-oil constraint.
- The data-driven model reaches a *lower training loss* than the PINN while
  reconstructing an order of magnitude worse — training loss is not a proxy for
  estimation quality across different objectives.
- Weibull RUL confidence intervals cover the truth 5.5 % of the time against a
  nominal 90 %.
- Inverting the health index to per-unit endurance was tried and abandoned;
  the paper says so and explains why.

---

## 8. Known weaknesses in the current draft

Stated in Section VII, and flagged here so you can decide whether to address them
before submitting:

1. **Single-seed classification.** PINN results average three seeds; the
   classification table does not. Sub-percentage-point differences between the
   closest variants are within run-to-run variance, and no significance testing
   is offered. **This is the weakness a reviewer is most likely to press on.**
   Multi-seed repetition with paired testing would fix it.
2. **Embedded latency is projected, not measured.** Clearly labelled as such, but
   a reviewer may still object to its presence.
3. **The physics is partly circular** — PINN residual and generator share the
   same equations, differing by 10 % in parameters. The reported gain is an upper
   bound.
4. **No public-benchmark comparison.** C-MAPSS was considered and rejected as
   incommensurable; the paper says so, but a reviewer may disagree.
5. **RUL R² = 0.666** is respectable but early-life predictions remain weak
   (171 d MAE at 35 % of life versus 48 d at 80 %).

---

## 9. Venue

Formatted in generic IEEE two-column style so it ports to whichever venue you
choose. Given the simulation-only scope, the earlier concern about IEEE TII
stands — it strongly favours experimental validation. *IEEE Access*, *Energies*,
or a simulation-focused venue are likelier fits.

Note that IEEE requires **single-column, double-spaced** format for initial
submission at most journals; the two-column layout here matches the
camera-ready convention. `make_manuscript.py` can emit single-column by setting
every section's `cols` to 1.

---

## 10. Deliverables

```
Tabrez_PINN_DigitalTwin_Manuscript.docx   revised manuscript
Tabrez_PINN_DigitalTwin_Manuscript.pdf    rendered preview
code/                                      full simulation framework + README
figures/                                   16 figures, PNG (400 dpi) + PDF
results/results.json                       every computed number
```
