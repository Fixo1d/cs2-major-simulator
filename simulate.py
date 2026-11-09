# Симуляция Stage 1 PGL Major Singapore 2026 по готовым параметрам модели (data/params_stage1.json).
# Запуск:  python simulate.py                 — 200 вариантов силы × 500 турниров = 100 000 (≈10 с)
#          python simulate.py 2000 500        — 1 000 000 прогонов, как в ролике (≈1,5–2 мин)
# Нужен только numpy. Результат печатается таблицей и сохраняется в results/my_run.json.
import json, random, itertools, sys, os, numpy as np
from collections import Counter
from swiss import run_stage

HERE = os.path.dirname(os.path.abspath(__file__))
S_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 200      # сколько раз «перебросить» силу команд (неуверенность модели)
PER_DRAW = int(sys.argv[2]) if len(sys.argv) > 2 else 500     # сколько турниров сыграть на каждом варианте силы
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, 'results', 'my_run.json')

P_ = json.load(open(os.path.join(HERE, 'data', 'params_stage1.json'), encoding='utf-8'))
SEEDS = P_['teams']                                  # 16 команд в порядке посева
n = len(SEEDS); ti = {t: i for i, t in enumerate(SEEDS)}
mu = np.array(P_['strength'])                        # средняя сила команды (логиты)
C = np.array(P_['cov'])                              # неуверенность в силе и связи между командами
Dmap = np.array(P_['map_adj'])                       # поправка команды на каждую карту пула (16 × 7)
VS = {tuple(k.split('|')[:2]) + (int(k.split('|')[2]),): np.array([[int(c) for c in s] for s in v.split()])
      for k, v in P_['veto'].items()}                # 400 разыгранных вето на пару и формат (индексы карт)
picks = json.load(open(os.path.join(HERE, 'results', 'pickem_stage1.json'), encoding='utf-8'))['picks']

def pA(r, a, b, bo):
    """Шанс, что a (команда A в вето) выиграет серию у b при силе r. Bo1 — одна карта, Bo3 — до двух побед."""
    ia, ib = ti[a], ti[b]; mm = VS[(a, b, bo)]
    p = 1 / (1 + np.exp(-(r[ia] - r[ib] + Dmap[ia, mm] - Dmap[ib, mm])))
    if bo == 1: return p[:, 0].mean()
    p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2]
    return (p1 * p2 + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3).mean()

def pk_score(res):
    """Сколько угадал Pick'Em модели (из 10) в одном турнире"""
    return (sum(res[t] == (3, 0) for t in picks['30']) + sum(res[t] == (0, 3) for t in picks['03'])
            + sum(res[t][0] == 3 and res[t][1] > 0 for t in picks['adv']))

L = np.linalg.cholesky(C + 1e-9 * np.eye(n))
draws = mu + np.random.default_rng(2026).standard_normal((S_DRAWS, n)) @ L.T
seed = {t: i for i, t in enumerate(SEEDS)}; rng = random.Random(11)
rec = {t: Counter() for t in SEEDS}; adv_by_draw = {t: [] for t in SEEDS}; PK = Counter()
for di, r in enumerate(draws):
    P = {}
    for hi, lo in itertools.combinations(SEEDS, 2):   # кто выше посевом, тот выбирает роль в вето
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
        rec[t].update(c[t]); adv_by_draw[t].append(sum(c[t][(3, l)] for l in (0, 1, 2)) / PER_DRAW)
    if (di + 1) % max(1, S_DRAWS // 10) == 0: print(f'  {di + 1}/{S_DRAWS} вариантов силы…', file=sys.stderr)

N = S_DRAWS * PER_DRAW; ORDER = ['3-0', '3-1', '3-2', '2-3', '1-3', '0-3']; out = []
for t in SEEDS:
    pr = {k: rec[t][tuple(map(int, k.split('-')))] / N for k in ORDER}
    a = np.array(adv_by_draw[t])
    out.append(dict(team=t, seed=seed[t] + 1, **pr, advance=pr['3-0'] + pr['3-1'] + pr['3-2'],
                    adv_lo=float(np.percentile(a, 10)), adv_hi=float(np.percentile(a, 90)), strength=float(mu[ti[t]])))
out.sort(key=lambda r: -r['advance'])
print(f'\n{N:,} прогонов Stage 1'.replace(',', ' '))
print('%-18s %5s | %6s %6s %6s | %6s %6s %6s | %6s  %s' % ('Команда', 'посев', '3-0', '3-1', '3-2', '2-3', '1-3', '0-3', 'ПРОХОД', 'диапазон'))
for r in out:
    print('%-18s %5d | %5.1f%% %5.1f%% %5.1f%% | %5.1f%% %5.1f%% %5.1f%% | %5.1f%%  %3.0f–%2.0f%%' % (
        r['team'], r['seed'], *[100 * r[k] for k in ORDER + ['advance']], 100 * r['adv_lo'], 100 * r['adv_hi']))
pk = dict(sorted(PK.items()))
print("\nPick'Em модели: верных в среднем %.2f из 10, шанс на монету (5+) %.1f%%" % (
    sum(k * v for k, v in pk.items()) / N, 100 * sum(v for k, v in pk.items() if k >= 5) / N))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(dict(teams=out, cfg=P_['cfg'], draws=S_DRAWS, per_draw=PER_DRAW, pickem_scores=pk),
          open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
