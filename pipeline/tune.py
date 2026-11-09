# Подбор настроек с честной проверкой «оставь один мажор»: настраиваем на двух мажорах, оцениваем на третьем.
import itertools, json, numpy as np, backtest as bt
bt.DATA = {k: bt.setup(k) for k in bt.CFG}
GRID = dict(H=(60, 120, 240), sR=(0.5, 1.0, 1.5), s_era=(0.1, 0.25, 0.5), w_lan=(0.7, 1.0, 1.5), use_form=(False, True), rw=(0.3, 0.6))
KS = (0.6, 0.8, 1.0, 1.2)
keys = list(GRID); res = {}
for vals in itertools.product(*GRID.values()):
    p = dict(zip(keys, vals), sD=0.15)
    for maj in bt.CFG:
        for K in KS:
            r = bt.evaluate(p, K=K, majors=(maj,))
            res[(vals, K, maj)] = (r[3], r[4], r[5])   # серии: средний log-lik, точность, n
json.dump({repr(k): v for k, v in res.items()}, open('../results/tune_results.json', 'w'))
majors = list(bt.CFG)
def score(vals, K, ms): return sum(res[(vals, K, m)][0] * res[(vals, K, m)][2] for m in ms) / sum(res[(vals, K, m)][2] for m in ms)
print('Честная проверка (настройки выбраны без этого мажора):')
tot_ll = tot_acc = tot_n = 0
for out in majors:
    ins = [m for m in majors if m != out]
    best = max(((v, K) for v in itertools.product(*GRID.values()) for K in KS), key=lambda vk: score(*vk, ins))
    ll, acc, n = res[(best[0], best[1], out)]
    tot_ll += ll * n; tot_acc += acc * n; tot_n += n
    print(f'  {out:9s}: серии ll {ll:.4f} угадано {100*acc:.1f}%  | настройки {dict(zip(keys, best[0]))} K={best[1]}')
print(f'  ИТОГО: ll {tot_ll/tot_n:.4f} угадано {100*tot_acc/tot_n:.1f}% (n={tot_n})')
best = max(((v, K) for v in itertools.product(*GRID.values()) for K in KS), key=lambda vk: score(*vk, majors))
print('Лучшие настройки на всех трёх:', dict(zip(keys, best[0])), 'K=', best[1], 'll %.4f' % score(*best, majors))
