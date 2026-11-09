# Проверка на прошлых мажорах: модель учится только на данных ДО начала мажора и прогнозирует все его матчи.
# Остин 2025, Будапешт 2025, Кёльн 2026 — по 99 матчей (3 стадии). Метрики: log-loss и % угаданных по картам и по сериям.
import json, os, sys, itertools, numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from model2 import Model, vrs_points, vrs_rosters, player_table, team_form
from past_majors import all_majors
RAW = os.path.join(os.path.dirname(__file__), '..', 'raw')
BUNDLE = json.load(open(os.path.join(RAW, 'hltv', 'bundle.json')))
CFG = {'Austin': ('Austin 2025', '2025_06_02', '2025-06-03'), 'Budapest': ('Budapest 2025', '2025_11_03', '2025-11-24'), 'Cologne': ('Cologne 2026', '2026_06_01', '2026-06-02')}
MAPN = {'Ancient': 'A', 'Anubis': 'U', 'Cache': 'C', 'Dust II': 'D', 'Dust2': 'D', 'Inferno': 'I', 'Mirage': 'M', 'Nuke': 'N', 'Overpass': 'O', 'Train': 'T', 'Vertigo': 'V'}

def key(r):
    d, a, b, sa, sb, m = r[:6]; return (d, m, frozenset([(a, int(sa)), (b, int(sb))]))

def merge_rows(lists, lan_list):
    lan = {key(r.split('\t')) for r in lan_list}; seen = set(); out = []
    for L in lists:
        for r in L:
            f = r.split('\t')
            if len(f) < 6 or not f[3].isdigit() or not f[4].isdigit(): continue
            k = key(f)
            if k in seen: continue
            seen.add(k); out.append((f[0], f[1], f[2], int(f[3]), int(f[4]), f[5], k in lan))
    return out

def code_names(matches, pm):
    """коды Liquipedia → имена HLTV: пересечение пар по всем матчам"""
    cand = {}
    for x in matches:
        r = pm.get(x['hltv'])
        if not r: continue
        pair = {r[2], r[4]}
        for c in (x['t1'], x['t2']): cand[c] = cand.get(c, pair) & pair
    out = {c: next(iter(s)) for c, s in cand.items() if len(s) == 1}
    for _ in range(5):                          # исключение: если в паре один код известен — второй = другое имя
        for x in matches:
            r = pm.get(x['hltv'])
            if not r: continue
            for c1, c2 in ((x['t1'], x['t2']), (x['t2'], x['t1'])):
                if c1 in out and c2 not in out: out[c2] = ({r[2], r[4]} - {out[c1]}).pop()
    return out

def make_alias(names, vrs):
    low = {k.lower(): k for k in vrs}
    return {n: low[n.lower()] for n in names if n not in vrs and n.lower() in low}

def setup(k):
    name, vd, start = CFG[k]
    rows = merge_rows([BUNDLE['MAPS_' + k], BUNDLE['TEAMMAPS_' + k]], BUNDLE['LAN_' + k])
    vrs = vrs_points(os.path.join(RAW, f'vrs_global_{vd}.md')); ros = vrs_rosters(os.path.join(RAW, f'details_{vd}'))
    top, al, sh = player_table(BUNDLE['PLAYERS'][k + '_top'], BUNDLE['PLAYERS'][k + '_all'])
    form = {t: team_form(r[0], top, al, sh)[0] for t, r in ros.items()}
    pm = {r.split('\t')[0]: r.split('\t') for r in BUNDLE['PASTMATCH']}
    A = all_majors(); ms = [x for s in (1, 2, 3) for x in A[(name, s)]]
    cn = code_names(ms, pm)
    alias = make_alias({r[1] for r in rows} | {r[2] for r in rows}, vrs)
    return rows, vrs, ros, form, ms, cn, start, alias

def evaluate(params, K=1.0, majors=('Austin', 'Budapest', 'Cologne'), verbose=False):
    Lm, Am, Ls, As = [], [], [], []
    for k in majors:
        rows, vrs, ros, form, ms, cn, start, alias = DATA[k]
        mod = Model(rows, start, vrs, ros, form, alias=alias, **params)
        for x in ms:
            a, b = cn.get(x['t1']), cn.get(x['t2'])
            if not a or not b: continue
            ps = []
            for m, sa, sb in x['maps']:
                p = mod.p_map(a, b, MAPN.get(m, '?'), K); ps.append(p)
                w = sa > sb; Lm.append(np.log(p if w else 1 - p)); Am.append((p > .5) == w)
            pbar = np.mean(ps); ser = pbar if x['bo'] == 1 else pbar ** 2 * (3 - 2 * pbar)   # серия: средняя сила по картам
            w = x['win'] == 1; Ls.append(np.log(ser if w else 1 - ser)); As.append((ser > .5) == w)
    return np.mean(Lm), np.mean(Am), len(Lm), np.mean(Ls), np.mean(As), len(Ls)

if __name__ == '__main__':
    DATA = {k: setup(k) for k in CFG}
    for k, (rows, vrs, ros, form, ms, cn, start, alias) in DATA.items():
        miss = {c for x in ms for c in (x['t1'], x['t2'])} - set(cn)
        nov = sorted({n for n in cn.values() if alias.get(n, n) not in vrs})
        print(k, 'карт', len(rows), 'LAN', sum(r[6] for r in rows), '| коды без имени:', miss, '| нет в VRS:', nov, '| без формы:', sorted(n for n in cn.values() if not form.get(alias.get(n, n))))
else:
    DATA = None
