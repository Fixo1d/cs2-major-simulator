# Stage 1, модель v2: составы + форма игроков + LAN + вето по истории команд. 100 000 прогонов швейцарки.
import json, random, itertools, sys, numpy as np, re
from collections import Counter
import backtest as bt
from model2 import Model
from current import setup_current, ASOF
from veto import parse, Veto, POOL
from series import best_role
from swiss import run_stage
CFG = dict(H=180, sR=0.75, sD=0.15, rw=0.4, w_lan=1.25, s_era=0.25, use_form=True); K = 1.0
N_SIM = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100000
SEEDS = ['Ninjas in Pyjamas', 'PARIVISION', 'HEROIC', 'FaZe', 'Liquid', 'Eternal Fire', '100 Thieves', '9z',
         'M80', 'DENDELE', 'Imperial', 'Wildcard', 'BC.Game', 'FlyQuest', 'Chinggis Warriors', 'Lynn Vision']
if __name__ == '__main__':
    # 1) насколько эта настройка точна на прошлых мажорах
    bt.DATA = {k: bt.setup(k) for k in bt.CFG}
    r = bt.evaluate(CFG, K=K); print(f'Прошлые мажоры: серии угадано {100*r[4]:.1f}% ll {r[3]:.4f} (n={r[5]}), карты {100*r[1]:.1f}%')
    # 2) модель на сегодня
    rows, vrs, ros, form = setup_current()
    alias = bt.make_alias({x[1] for x in rows} | {x[2] for x in rows}, vrs)
    mod = Model(rows, ASOF, vrs, ros, form, alias=alias, **CFG)
    Mv = [m for m in parse(json.load(open('../raw/hltv/bundle.json'))['VETO']) if all(a[2] in POOL for a in m[6])]
    V = Veto(Mv, alpha=0.5, gamma=1.5)
    pm = lambda a, b: {m: mod.p_map(a, b, m, K) for m in POOL}
    P = {}
    for a, b in itertools.permutations(SEEDS, 2):
        for bo in (1, 3): P[(a, b, bo)] = best_role(a, b, bo, pm(a, b), V, n=500)
    seed = {t: i for i, t in enumerate(SEEDS)}
    def play(a, b, bo, rng):
        hi, lo = (a, b) if seed[a] < seed[b] else (b, a)
        return hi if rng.random() < P[(hi, lo, bo)] else lo
    rng = random.Random(2026); rec = {t: Counter() for t in SEEDS}
    for _ in range(N_SIM):
        res, _ = run_stage(SEEDS, play, rng)
        for t, wl in res.items(): rec[t][wl] += 1
    ORDER = ['3-0', '3-1', '3-2', '2-3', '1-3', '0-3']
    out = []
    for t in SEEDS:
        pr = {k: rec[t][tuple(map(int, k.split('-')))] / N_SIM for k in ORDER}
        Ra, Da = mod.strength(t)
        out.append(dict(team=t, seed=seed[t] + 1, rating=round(float(Ra), 3), form=round(form.get(alias.get(t, t)) or 0, 3),
                        best_map=POOL[int(np.argmax(Da[[mod.MAPS.find(m) for m in POOL]]))], **pr, advance=pr['3-0'] + pr['3-1'] + pr['3-2']))
    out.sort(key=lambda r: -r['advance'])
    print('%-18s %4s %6s %6s | %6s %6s %6s | %6s %6s %6s | %6s' % ('Команда', 'посев', 'сила', 'форма', '3-0', '3-1', '3-2', '2-3', '1-3', '0-3', 'ПРОХОД'))
    for r in out: print('%-18s %4d %+6.2f %6.2f | %5.1f%% %5.1f%% %5.1f%% | %5.1f%% %5.1f%% %5.1f%% | %5.1f%%' % (r['team'], r['seed'], r['rating'], r['form'], *[100 * r[k] for k in ORDER + ['advance']]))
    print('\nПервый тур (Bo1), с вето:')
    r1 = []
    for i in range(8):
        a, b = SEEDS[i], SEEDS[i + 8]; p = P[(a, b, 1)]; r1.append(dict(a=a, b=b, p=p)); print('  %-18s %3.0f%% – %3.0f%%  %s' % (a, 100 * p, 100 * (1 - p), b))
    json.dump(dict(teams=out, round1=r1, cfg=CFG), open('../results/stage1_result_v2.json', 'w'), ensure_ascii=False, indent=1)
