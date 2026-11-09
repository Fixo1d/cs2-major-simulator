# Выгрузка посчитанных параметров модели для публичного репозитория (без сырых данных HLTV):
# средняя сила 16 команд, ковариация силы, поправки на карты, исходы вето для каждой пары и формата.
# Это ровно то, что использует stage1_v3.py, поэтому публичный simulate.py даёт те же числа.
import json, random, itertools, sys, numpy as np
import backtest as bt
from model2 import Model
from current import setup_current, ASOF
from veto import parse, Veto, POOL
from series import sample_veto, BO1, BO3
from stage1_v2 import CFG, SEEDS
OUT = sys.argv[1] if len(sys.argv) > 1 else '../data/params_stage1.json'; N_VETO = 400
rows, vrs, ros, form = setup_current()
alias = bt.make_alias({x[1] for x in rows} | {x[2] for x in rows}, vrs)
mod = Model(rows, ASOF, vrs, ros, form, alias=alias, **CFG)
V = Veto([m for m in parse(json.load(open('../raw/hltv/bundle.json'))['VETO']) if all(a[2] in POOL for a in m[6])], alpha=0.5, gamma=1.5)
idx = [mod.ent[(t, 0)] for t in SEEDS]
C = mod.cov_R()[np.ix_(idx, idx)]
Dmap = [[float(mod.strength(t)[1][mod.MAPS.find(m)]) for m in POOL] for t in SEEDS]
vrng = random.Random(5); VS = {}
for a, b in itertools.permutations(SEEDS, 2):          # тот же порядок и тот же генератор, что в stage1_v3.py
    for bo, order in ((1, BO1), (3, BO3)):
        VS[f'{a}|{b}|{bo}'] = ' '.join(''.join(str(POOL.index(m)) for m in sample_veto(order, a, b, {m: .5 for m in POOL}, V, vrng)[:bo if bo == 3 else 1]) for _ in range(N_VETO))
json.dump(dict(asof=ASOF, cfg=CFG, teams=SEEDS, maps=list(POOL), strength=[float(mod.R[i]) for i in idx],
               cov=C.tolist(), map_adj=Dmap, veto=VS), open(OUT, 'w'), ensure_ascii=False)
print('ok', OUT, len(VS))
