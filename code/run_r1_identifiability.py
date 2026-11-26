#!/usr/bin/env python3
"""run_r1_identifiability.py -- local identifiability of the thermal constants (Table IX).

Sensitivity of the observed outputs (top oil at 180 instants, plus the hot-spot
labels if any) to log-scale changes of the seven IEC constants, and the
Cramer-Rao standard deviation of each log-parameter, for N_hs = 0, 6 and 60.
Median over seeds 0-4 of the S0 setting. Writes results/ident.json.

Author: Md Tabrez
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_r1_thermal import scenario_inputs
from sim import r1_thermal as T

out = {}
for nhs in (0, 6, 60):
    crbs, svs = [], []
    for s in range(5):
        tr, obs, tp, ic = scenario_inputs('S0', s, nhs, 'exact')
        r = T.identifiability(tr, obs, T.ONAF)
        crbs.append(r['crb_logsd']); svs.append(r['singular_values'])
        if s == 0:
            print(f'N_hs={nhs}: rank {r["rank"]} of 7; column norms', np.round(r['col_norm'], 2))
    out[nhs] = dict(crb=np.median(np.array(crbs), 0).tolist(), sv=np.median(np.array(svs), 0).tolist())
    print(f'N_hs={nhs}:', dict(zip(T.PNAMES, np.round(100 * np.median(np.array(crbs), 0), 1))), '(% log-S.D.)')
json.dump(out, open(os.path.join(HERE, '..', 'results', 'ident.json'), 'w'), indent=1)
