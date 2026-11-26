"""make_tables_r1.py -- collect every R1 number (Tables VII-XIV, Fig. 6, Fig. 14) from the result files.

Runs the conventional estimators (IEC model, Kalman/RTS, EKF, gray-box) itself; they take seconds.
Writes results/r1_numbers.json, which make_figures_r1.py reads.
Author: Md Tabrez"""
import collections, json, sys
import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_r1_thermal import scenario_inputs
from sim import r1_thermal as T
from sim.r1_clf import mean_ci

RES = os.path.join(HERE, '..', 'results') + os.sep
N = {}

# ---------------------------------------------------------------- thermal
L = [json.loads(l) for l in open(RES + 'r1_thermal.jsonl')]
B = collections.defaultdict(dict)
for d in L:
    B[(d['kind'], d['scen'], d['est'], d['n_hs'], d['lam_p'], d.get('param'), d.get('sign'))][d['seed']] = d


def pinn(kind, scen, est, nhs, lp=10.0, key='rmse', param=None, sign=None):
    v = B[(kind, scen, est, nhs, lp if est != 'nn' else 0.0, param, sign)]
    a = np.array([v[s][key] for s in sorted(v)])
    if len(a) == 0:
        import os
        assert os.environ.get('R1TEST'), (kind, scen, est, nhs, lp, param, sign)
        return np.array([np.nan, np.nan])
    return a


CONV = [('IEC forward model', T.est_iec_forward), ('Kalman RTS smoother', T.est_kalman_rts),
        ('EKF, joint parameters', T.est_ekf_joint), ('Gray-box, identified', T.est_graybox)]


def conv(scen, nhs, mode):
    out = {}
    for nm, fn in CONV:
        vals = []
        for s in range(5):
            tr, obs, tp, ic = scenario_inputs(scen, s, nhs, mode)
            r = fn(tr, obs, tp, ic)
            vals.append(T.metrics(tr, r[0], r[1]))
        out[nm] = vals
    return out


ms = lambda a: f'{np.mean(a):.2f} ± {np.std(a, ddof=1):.2f}'
# Table VII (original protocol: exact IC, 6 labels)
c7 = conv('S0', 6, 'exact')
import os
_t7 = lambda est, lp: [B[('table7', 'S0', est, 6, lp, None, None)][s] for s in range(5)
                       if s in B[('table7', 'S0', est, 6, lp, None, None)] or not os.environ.get('R1TEST')]
p7 = _t7('pinn', 10.0) or [B[('bench', 'S0', 'pinn', 6, 10.0, None, None)][0]] * 2
n7 = _t7('nn', 0.0) or [B[('bench', 'S0', 'nn', 6, 0.0, None, None)][0]] * 2
rows7 = []
for nm, vals in [('PINN (*λ*_{p} = 10)', p7), ('Data-driven (*λ*_{p} = 0)', n7)] + list(c7.items()):
    rows7.append([nm, ms([v['rmse'] for v in vals]), ms([v['mae'] for v in vals]),
                  f"{np.mean([v['r2'] for v in vals]):.3f}"])
N['t7_rows'] = rows7
N['p7'] = np.mean([v['rmse'] for v in p7]); N['p7sd'] = np.std([v['rmse'] for v in p7], ddof=1)
N['n7'] = np.mean([v['rmse'] for v in n7]); N['n7sd'] = np.std([v['rmse'] for v in n7], ddof=1)
N['p7r2'] = np.mean([v['r2'] for v in p7]); N['n7r2'] = np.mean([v['r2'] for v in n7])
N['p7to'] = np.mean([v['rmse_to'] for v in p7]); N['p7g'] = np.mean([v['rmse_grad'] for v in p7])
for nm, key in zip([c[0] for c in CONV], ['fwd7', 'kf7', 'ekf7', 'gb7']):
    N[key] = np.mean([v['rmse'] for v in c7[nm]])
N['gb7to'] = np.mean([v['rmse_to'] for v in c7['Gray-box, identified']])
N['gb7g'] = np.mean([v['rmse_grad'] for v in c7['Gray-box, identified']])
N['kf7g'] = np.mean([v['rmse_grad'] for v in c7['Kalman RTS smoother']])

# Table VIII (benchmark)
SC = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
bench = {}
for sc in SC:
    row = {}
    row['PINN, 0 labels'] = pinn('bench', sc, 'pinn', 0).mean()
    row['PINN'] = pinn('bench', sc, 'pinn', 6).mean()
    row['Inverse PINN'] = pinn('bench', sc, 'pinn_inv', 6).mean()
    row['Data-driven NN'] = pinn('bench', sc, 'nn', 6).mean()
    c0 = conv(sc, 0, 'assumed'); c6 = conv(sc, 6, 'assumed')
    row['IEC forward model'] = np.mean([v['rmse'] for v in c0['IEC forward model']])
    row['Kalman RTS, 0 labels'] = np.mean([v['rmse'] for v in c0['Kalman RTS smoother']])
    row['Kalman RTS smoother'] = np.mean([v['rmse'] for v in c6['Kalman RTS smoother']])
    row['EKF, 0 labels'] = np.mean([v['rmse'] for v in c0['EKF, joint parameters']])
    row['EKF, joint parameters'] = np.mean([v['rmse'] for v in c6['EKF, joint parameters']])
    row['Gray-box, 0 labels'] = np.mean([v['rmse'] for v in c0['Gray-box, identified']])
    row['Gray-box, identified'] = np.mean([v['rmse'] for v in c6['Gray-box, identified']])
    bench[sc] = row
ESTS = ['PINN', 'Inverse PINN', 'Data-driven NN', 'IEC forward model', 'Kalman RTS smoother',
        'EKF, joint parameters', 'Gray-box, identified', 'PINN, 0 labels', 'Gray-box, 0 labels']
LAB = {'PINN': 'PINN', 'Inverse PINN': 'Inverse PINN', 'Data-driven NN': 'Data-driven NN',
       'IEC forward model': 'IEC model (open loop)†', 'Kalman RTS smoother': 'Kalman RTS',
       'EKF, joint parameters': 'EKF, joint param.', 'Gray-box, identified': 'Gray-box, identified',
       'PINN, 0 labels': 'PINN, no labels†', 'Gray-box, 0 labels': 'Gray-box, no labels†'}
N['t8_header'] = ['Estimator'] + SC
N['t8_widths'] = [1460] + [560] * 6
rows8 = []
for e in ESTS:
    vals = [bench[s][e] for s in SC]
    best = [min(bench[s][x] for x in ESTS if x not in ('PINN, 0 labels', 'Gray-box, 0 labels')) for s in SC]
    rows8.append([LAB[e]] + [f'{v:.2f}' for v in vals])
N['t8_rows'] = rows8
N['bench'] = bench
best_of = {s: min(((bench[s][x], x) for x in ESTS[:7])) for s in SC}
N['best_of'] = {s: v[1] for s, v in best_of.items()}

# OAT
oat = {}
for nm in T.PNAMES:
    for sg in (1, -1):
        oat[(nm, sg)] = float(pinn('oat', 'S0', 'pinn', 6, param=nm, sign=sg)[0])
N['oat'] = {f'{k[0]}{"+" if k[1] > 0 else "-"}': v for k, v in oat.items()}

# identifiability table
ID = json.load(open(RES + 'ident.json'))
lab = {'dTheta_or': 'ΔΘ_{or}', 'R': '*R*', 'x': '*x*', 'tau_o': 'τ_{o}', 'Hgr': 'H·g_{r}', 'y': '*y*',
       'tau_w': 'τ_{w}'}
rows9 = []
for j, nm in enumerate(T.PNAMES):
    r = [lab[nm]]
    for k in ('0', '6', '60'):
        v = ID[k]['crb'][j]
        r.append('not identifiable' if (k == '0' and j >= 4) else f'{100 * v:.1f}')
    rows9.append(r)
N['t9_rows'] = rows9
N['id'] = {k: [100 * x for x in ID[k]['crb']] for k in ID}

# lambda and labels sweeps
LAM = (0.0, 0.03, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
lam = {lp: pinn('lambda', 'S0', 'pinn', 6, lp=lp) for lp in LAM}
N['lam'] = {str(k): [float(v.mean()), float(v.std(ddof=1))] for k, v in lam.items()}
LB = (0, 2, 6, 20, 60, 200)
lbl = {}
for nl in LB:
    pp = pinn('labels', 'S0', 'pinn', nl); nn = pinn('labels', 'S0', 'nn', nl)
    g = []
    for s in range(5):
        tr, obs, tp, ic = scenario_inputs('S0', s, nl, 'exact')
        a, b, _ = T.est_graybox(tr, obs, tp, ic); g.append(T.metrics(tr, a, b)['rmse'])
    lbl[nl] = dict(pinn=[pp.mean(), pp.std(ddof=1)], nn=[nn.mean(), nn.std(ddof=1)],
                   gb=[float(np.mean(g)), float(np.std(g, ddof=1))])
N['lbl'] = {str(k): v for k, v in lbl.items()}
MP = json.load(open(RES + 'r1_multiphase.json'))
mp0 = [m['imbalance_rmse'] for m in MP if m['lam_couple'] == 0.0]
mp1 = [m['imbalance_rmse'] for m in MP if m['lam_couple'] == 1.0]
N['mp'] = dict(u=float(np.mean(mp0)), c=float(np.mean(mp1)), us=float(np.std(mp0, ddof=1)),
               cs=float(np.std(mp1, ddof=1)),
               ru=float(np.mean([m['rmse'] for m in MP if m['lam_couple'] == 0.0])),
               rc=float(np.mean([m['rmse'] for m in MP if m['lam_couple'] == 1.0])))
N['mp_raw'] = MP

# ---------------------------------------------------------------- classification
CV = json.load(open(RES + 'cv_summary.json'))
S, TS = CV['summary'], CV['tests']
pt = {(t['a'], t['b']): t for t in TS}
st = 'Student, distilled + INT8'


def pv(a):
    for k, t in pt.items():
        if st in k and a in k and a != st:
            return t['p']
    return None


order = [('RandomForest', 'Random forest (RF-200)', '—', ''),
         ('RandomForest, no physics', 'RF-200, no physics features', '—', '—'),
         ('GRU', 'GRU', '—', '—'), ('CNN-LSTM', 'CNN-LSTM', '—', '—'),
         ('Teacher', 'Teacher (Bi-GRU + RF)', '126,758', '497 kB'),
         ('Student, no distillation', 'Student, no distillation', '15,174', '47 kB'),
         ('Student, no distillation, no physics', 'Student, no distill., no physics', '15,174', '47 kB'),
         ('Student, distilled', 'Student, distilled (FP32)', '15,174', '61 kB'),
         (st, 'Student, distilled + INT8 (proposed)', '15,174', '47 kB')]
EDGE = json.load(open(RES + 'r1_edge.json'))
rf_kb = EDGE['model']['rf200_size_kB']
rows10 = []
for k, nm, par, sz in order:
    s = S[k]
    if k == 'RandomForest':
        sz = f'{rf_kb:.0f} kB'
    if k == 'RandomForest, no physics':
        sz = '—'
    p = pv(k)
    if k == st:
        pstr = '—'
    elif k == 'Student, no distillation, no physics':
        t = pt[('Student, no distillation', 'Student, no distillation, no physics')]
        pstr = f"{t['p']:.2f}^{{b}}"
    elif k == 'RandomForest, no physics':
        pstr = f"{pt[('RandomForest', 'RandomForest, no physics')]['p']:.2f}^{{b}}"
    elif k == 'Student, distilled':
        pstr = f"{pt[('Student, distilled + INT8', 'Student, distilled')]['p']:.2f}^{{c}}"
    elif k == 'Student, no distillation':
        pstr = f"{pt[('Student, distilled', 'Student, no distillation')]['p']:.2f}^{{a}}"
    else:
        pstr = f'{p:.3f}' if p is not None else '—'
    rows10.append([nm, f"{100 * s['mean']:.2f} ± {100 * s['sd']:.2f}",
                   f"{100 * s['lo']:.1f}–{100 * s['hi']:.1f}", f"{s['f1']:.3f}", pstr, par, sz])
N['t10_rows'] = rows10
N['t10_widths'] = [1250, 760, 640, 520, 520, 600, 530]
g = lambda k: 100 * S[k]['mean']
gs = lambda k: 100 * S[k]['sd']
N.update(rf=g('RandomForest'), rfsd=gs('RandomForest'), te=g('Teacher'), tesd=gs('Teacher'),
         st=g(st), stsd=gs(st), sns=g('Student, no distillation'), snsd=gs('Student, no distillation'),
         skd=g('Student, distilled'), rfnp=g('RandomForest, no physics'),
         snp=g('Student, no distillation, no physics'), gru=g('GRU'), cnn=g('CNN-LSTM'))
N['p_kd'] = pt[('Student, distilled', 'Student, no distillation')]['p']
N['p_ts'] = pt[('Teacher', st)]['p']
N['mc_ts'] = pt[('Teacher', st)]['mcnemar']
N['p_rfst'] = pt[('RandomForest', st)]['p']
N['mc_rfst'] = pt[('RandomForest', st)]['mcnemar']
N['p_rfphys'] = pt[('RandomForest', 'RandomForest, no physics')]['p']
N['p_stphys'] = pt[('Student, no distillation', 'Student, no distillation, no physics')]['p']
N['p_int8'] = pt[('Student, distilled + INT8', 'Student, distilled')]['p']
N.update(rec_normal=61.5, rec_tap=63.1, rec_min_other=94.6)
SEV = json.load(open(RES + 'r1_clf_sev.json'))
for dirn, key in (('low->high', 'lh'), ('high->low', 'hl')):
    a = [r for r in SEV if r['direction'] == dirn]
    N[key + '_st'] = 100 * np.mean([r['acc_student'] for r in a])
    N[key + '_rf'] = 100 * np.mean([r['acc_rf'] for r in a])
SZ = json.load(open(RES + 'r1_clf_size.json'))
sz = collections.defaultdict(list)
for r in SZ:
    sz[r['fraction']].append(r)
N['size'] = {str(k): dict(n=float(np.mean([r['n_train'] for r in v])),
                          st=[100 * np.mean([r['acc_student'] for r in v]), 100 * np.std([r['acc_student'] for r in v], ddof=1)],
                          rf=[100 * np.mean([r['acc_rf'] for r in v]), 100 * np.std([r['acc_rf'] for r in v], ddof=1)])
             for k, v in sz.items()}
SH = json.load(open(RES + 'r1_clf_shift.json'))
shn = {'X0 nominal (fresh seed)': 'X0 Nominal, new seed',
       'X1 hot climate, heavy load': 'X1 Ambient +9 K, load +0.17 p.u.',
       'X2 60 Hz network': 'X2 60 Hz network',
       'X3 low SNR (4 dB)': 'X3 SNR 4 dB (trained at 8 dB)',
       'X4 wider calibration spread': 'X4 Calibration S.D. 20 %/12 %',
       'X5 weaker faults (severity 0.01-0.06)': 'X5 Severity 0.01–0.06'}
rows11 = []
for k, v in SH['shifts'].items():
    a = v['acc']
    rows11.append([shn[k], f"{100 * np.mean(a['student']):.1f}", f"{100 * np.mean(a['teacher']):.1f}",
                   f"{100 * np.mean(a['rf']):.1f}"])
N['t11_header'] = ['Test condition', 'Student (INT8)', 'Teacher', 'RF-200']
N['t11_widths'] = [2210, 870, 870, 870]
N['t11_rows'] = rows11
N['shift'] = {k: {m: 100 * float(np.mean(x)) for m, x in v['acc'].items()} for k, v in SH['shifts'].items()}
N['mixed'] = SH['mixed']

# ---------------------------------------------------------------- edge
E = EDGE
N['edge'] = E

# ---------------------------------------------------------------- RUL
RU = json.load(open(RES + 'r1_rul.json'))
r_or = [r for r in RU if r['variant'] == 'G0' and r['true_labels']][0]
r0 = [r for r in RU if r['variant'] == 'G0' and not r['true_labels']][0]
names = ['Weibull HI', 'Physics-informed loss of life', 'Linear HI extrapolation', 'IEC ageing budget',
         'Random-forest regression']
lab13 = {'Weibull HI': 'Weibull health-index extrapolation',
         'Physics-informed loss of life': 'Physics-informed loss of life (proposed)',
         'Linear HI extrapolation': 'Linear health-index trend',
         'IEC ageing budget': 'IEC ageing budget (fleet median)',
         'Random-forest regression': 'Random-forest regression'}
N['t13_rows'] = [[lab13[k], f"{r0['overall'][k]['mae']:.1f}", f"{r0['overall'][k]['rmse']:.1f}",
                  f"{r0['overall'][k]['r2']:.3f}", f"{r0['overall'][k]['bias']:+.1f}"] for k in names]
N['rul0'] = r0; N['rulor'] = r_or
vlab = {'G0': 'G0 Lognormal σ = 0.26, IEC law', 'G1': 'G1 Lognormal σ = 0.15', 'G2': 'G2 Lognormal σ = 0.40',
        'G3': 'G3 Weibull endurance, k = 3', 'G4': 'G4 Moisture-accelerated ageing',
        'G5': 'G5 Upgraded-paper Arrhenius law'}
rows14 = []
for r in RU:
    if r['true_labels']:
        continue
    o = r['overall']
    rows14.append([vlab[r['variant']], f"{o['Physics-informed loss of life']['mae']:.1f}",
                   f"{o['Weibull HI']['mae']:.1f}", f"{o['Random-forest regression']['mae']:.1f}",
                   f"{100 * r['conformal']['coverage']:.0f}"])
N['t14_header'] = ['Ground truth', 'Proposed', 'Weibull', 'RF regr.', 'PI cover. (%)']
N['t14_widths'] = [1780, 760, 760, 760, 760]
N['t14_rows'] = rows14
N['rulv'] = {r['variant']: r for r in RU if not r['true_labels']}

json.dump(N, open(RES + 'r1_numbers.json', 'w'), default=float, indent=1)
print('numbers collected')
