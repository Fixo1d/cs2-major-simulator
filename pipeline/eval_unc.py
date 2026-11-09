# Помогает ли учёт неуверенности на прошлых мажорах? Сравнение «точечной» силы и усреднения по 300 вариантам силы.
import numpy as np, backtest as bt
from model2 import Model
from stage1_v2 import CFG
bt.DATA = {k: bt.setup(k) for k in bt.CFG}
rng = np.random.default_rng(7)
for scale in (0.0, 0.5, 1.0, 1.5):
    L = []; A = []
    for k in bt.CFG:
        rows, vrs, ros, form, ms, cn, start, alias = bt.DATA[k]
        mod = Model(rows, start, vrs, ros, form, alias=alias, **CFG)
        teams = sorted({cn[c] for x in ms for c in (x['t1'], x['t2']) if c in cn and (cn[c], 0) in mod.ent})
        Smp = mod.sample(teams, 300, rng); ti = {t: i for i, t in enumerate(teams)}
        Smp = mod.R[[mod.ent[(t, 0)] for t in teams]] + scale * (Smp - mod.R[[mod.ent[(t, 0)] for t in teams]])
        for x in ms:
            a, b = cn.get(x['t1']), cn.get(x['t2'])
            if a not in ti or b not in ti: continue
            Da = mod.strength(a)[1]; Db = mod.strength(b)[1]
            ps = []
            for m, sa, sb in x['maps']:
                j = mod.MAPS.find(bt.MAPN.get(m, '?')); dm = (Da[j] - Db[j]) if j >= 0 else 0
                ps.append(Smp[:, ti[a]] - Smp[:, ti[b]] + dm)
            z = np.mean(ps, axis=0); p = 1 / (1 + np.exp(-z))
            ser = p if x['bo'] == 1 else p ** 2 * (3 - 2 * p); ser = ser.mean()
            w = x['win'] == 1; L.append(np.log(ser if w else 1 - ser)); A.append((ser > .5) == w)
    print(f'неуверенность ×{scale}: серии ll {np.mean(L):.4f} угадано {100*np.mean(A):.1f}% (n={len(L)})')
