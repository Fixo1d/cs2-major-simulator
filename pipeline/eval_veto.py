# Проверка модели вето: учим частоты банов/пиков на матчах до 01.10.2026, проверяем действия в матчах после.
import json, numpy as np, re
from veto import parse, Veto, POOL
B = json.load(open('../raw/hltv/bundle.json'))
M = parse(B['VETO'] + B['PASTMATCH'])
def dt(s):
    m = re.match(r'(\d+)\w* of (\w+) (\d{4})', s)
    mon = 'January February March April May June July August September October November December'.split().index(m.group(2)) + 1
    return f'{m.group(3)}-{mon:02d}-{int(m.group(1)):02d}'
M = [(mid, dt(d), *rest) for mid, d, *rest in M if re.match(r'\d+\w* of', d)]
cur = [m for m in M if m[1] >= '2026-07-08' and all(a[2] in POOL for a in m[6])]
train = [m for m in cur if m[1] < '2026-10-01']; test = [m for m in cur if m[1] >= '2026-10-01']
print('матчей с вето: обучение', len(train), 'проверка', len(test))
for alpha, gamma in ((1, 0), (1, 0.5), (1, 1), (0.5, 1), (2, 1), (1, 1.5)):
    V = Veto(train, alpha, gamma); L = []; hit = 0; nb = 0
    for mid, d, t1, t2, bo, lan, acts in test:
        left = list(POOL)
        for team, act, m in acts:
            if team is None or m not in left: continue
            pr = V.probs(team, act, left); L.append(np.log(pr[m])); left.remove(m)
        # Bo1: какая карта будет сыграна — 3000 симуляций вето по модели против факта
    print(f'alpha={alpha} gamma={gamma}: средний log P(действия) {np.mean(L):.3f} (случайно {np.mean([np.log(1/len(POOL[:k])) for k in [7,6,5,4,3,2]]):.3f}), действий {len(L)}')
# точность «какая карта будет сыграна в Bo1»
import random
V = Veto(train, 1, 1); rng = random.Random(1); ok = []; base = []
from series import BO1
for mid, d, t1, t2, bo, lan, acts in test:
    if bo != 1: continue
    real = [m for team, act, m in acts if team is None]
    if not real: continue
    first = acts[0][0]; A, Bt = (t1, t2) if first == t1 else (t2, t1)
    cnt = {m: 0 for m in POOL}
    for _ in range(2000):
        left = list(POOL)
        for who, act in BO1:
            pr = V.probs(A if who == 'A' else Bt, act, left); ms = list(pr); m = rng.choices(ms, [pr[x] for x in ms])[0]; left.remove(m)
        cnt[left[0]] += 1
    ok.append(cnt[real[0]] / 2000); base.append(1 / 7)
print(f'Bo1: вероятность, которую модель давала реально сыгранной карте, в среднем {np.mean(ok):.2f} (наугад 0.14), матчей {len(ok)}; угадана как самая вероятная: ', end='')
