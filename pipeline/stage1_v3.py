# Stage 1, v3: как v2 + неуверенность в силе команд. 200 случайных вариантов силы (приближение Лапласа, с корреляциями) ×
# 500 прогонов швейцарки = 100 000. Для каждой команды — средний шанс и диапазон (10–90 % вариантов).
import json, random, itertools, numpy as np
from collections import Counter
import backtest as bt
from model2 import Model
from current import setup_current, ASOF
from veto import parse, Veto, POOL
from series import sample_veto, BO1, BO3
from swiss import run_stage
from stage1_v2 import CFG, SEEDS
import sys
S_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 200; PER_DRAW = int(sys.argv[2]) if len(sys.argv) > 2 else 500; N_VETO = 400
OUT = sys.argv[3] if len(sys.argv) > 3 else '../results/stage1_result_v3.json'
PICKS = json.load(open('../results/pickem_stage1.json'))['picks']; PK = Counter()   # Pick'Em модели считаем прямо по ходу
def pk_score(res): return sum(res[t] == (3, 0) for t in PICKS['30']) + sum(res[t] == (0, 3) for t in PICKS['03']) + sum(res[t][0] == 3 and res[t][1] > 0 for t in PICKS['adv'])
rows, vrs, ros, form = setup_current()
alias = bt.make_alias({x[1] for x in rows} | {x[2] for x in rows}, vrs)
mod = Model(rows, ASOF, vrs, ros, form, alias=alias, **CFG)
V = Veto([m for m in parse(json.load(open('../raw/hltv/bundle.json'))['VETO']) if all(a[2] in POOL for a in m[6])], alpha=0.5, gamma=1.5)
n = len(SEEDS); ti = {t: i for i, t in enumerate(SEEDS)}
Dmap = np.array([[mod.strength(t)[1][mod.MAPS.find(m)] for m in POOL] for t in SEEDS])     # поправки на карты (n, 7)
# вето не зависит от силы (λ=0) → разыгрываем один раз: для каждой пары (A, B) и формата — 400 исходов вето
vrng = random.Random(5); VS = {}
for a, b in itertools.permutations(SEEDS, 2):
    for bo, order in ((1, BO1), (3, BO3)):
        VS[(a, b, bo)] = np.array([[POOL.index(m) for m in sample_veto(order, a, b, {m: .5 for m in POOL}, V, vrng)[:bo if bo == 3 else 1]] for _ in range(N_VETO)])
def pA(r, a, b, bo):                                          # P(a как команда A выигрывает), r — сила (n,)
    ia, ib = ti[a], ti[b]; mm = VS[(a, b, bo)]
    p = 1 / (1 + np.exp(-(r[ia] - r[ib] + Dmap[ia, mm] - Dmap[ib, mm])))
    if bo == 1: return p[:, 0].mean()
    p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2]; return (p1 * p2 + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3).mean()
draws = mod.sample(SEEDS, S_DRAWS, np.random.default_rng(2026))
seed = {t: i for i, t in enumerate(SEEDS)}; rng = random.Random(11)
rec = {t: Counter() for t in SEEDS}; adv_by_draw = {t: [] for t in SEEDS}; zero3 = {t: [] for t in SEEDS}
for r in draws:
    P = {}
    for hi, lo in itertools.combinations(SEEDS, 2):         # SEEDS по порядку посева → hi всегда выше
        for bo in (1, 3): P[(hi, lo, bo)] = max(pA(r, hi, lo, bo), 1 - pA(r, lo, hi, bo))
    def play(a, b, bo, rg):
        hi, lo = (a, b) if seed[a] < seed[b] else (b, a)
        return hi if rg.random() < P[(hi, lo, bo)] else lo
    c = {t: Counter() for t in SEEDS}
    for _ in range(PER_DRAW):
        res, _ = run_stage(SEEDS, play, rng)
        for t, wl in res.items(): c[t][wl] += 1
        PK[pk_score(res)] += 1
    for t in SEEDS:
        rec[t].update(c[t]); adv_by_draw[t].append(sum(c[t][(3, l)] for l in (0, 1, 2)) / PER_DRAW); zero3[t].append(c[t][(0, 3)] / PER_DRAW)
N = S_DRAWS * PER_DRAW; ORDER = ['3-0', '3-1', '3-2', '2-3', '1-3', '0-3']; out = []
for t in SEEDS:
    pr = {k: rec[t][tuple(map(int, k.split('-')))] / N for k in ORDER}
    a = np.array(adv_by_draw[t])
    out.append(dict(team=t, seed=seed[t] + 1, **pr, advance=pr['3-0'] + pr['3-1'] + pr['3-2'],
                    adv_lo=float(np.percentile(a, 10)), adv_hi=float(np.percentile(a, 90)), strength=float(mod.R[mod.ent[(t, 0)]])))
out.sort(key=lambda r: -r['advance'])
print('%-18s %5s | %6s %6s %6s | %6s %6s %6s | %6s  %s' % ('Команда', 'посев', '3-0', '3-1', '3-2', '2-3', '1-3', '0-3', 'ПРОХОД', 'диапазон (10–90 %)'))
for r in out: print('%-18s %5d | %5.1f%% %5.1f%% %5.1f%% | %5.1f%% %5.1f%% %5.1f%% | %5.1f%%  %3.0f–%2.0f%%' % (r['team'], r['seed'], *[100 * r[k] for k in ORDER + ['advance']], 100 * r['adv_lo'], 100 * r['adv_hi']))
pk = dict(sorted(PK.items())); print("Pick'Em модели: верных в среднем %.2f, монета (≥5) %.1f%%" % (sum(k * v for k, v in pk.items()) / N, 100 * sum(v for k, v in pk.items() if k >= 5) / N))
json.dump(dict(teams=out, cfg=CFG, draws=S_DRAWS, per_draw=PER_DRAW, pickem_scores=pk), open(OUT, 'w'), ensure_ascii=False, indent=1)
