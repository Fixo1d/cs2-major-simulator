# Модель по картам HLTV: сила команды R + поправка на карту Δ(карта).
# P(A берёт карту m) = σ((R_A + Δ_A,m − R_B − Δ_B,m) / S), S = 1 (рейтинг в «логитах»).
# Обучение — взвешенное правдоподобие: свежие карты важнее (полураспад H дней), регуляризация σ_R, σ_Δ.
# Цель y — победа на карте, смешанная с долей раундов (rw): учитываем, 13-2 это или 16-14.
import numpy as np, os
from scipy.optimize import minimize
F = os.path.join(os.path.dirname(__file__), '..', 'raw', 'hltv', 'hltv_maps.csv')
POOL = 'AUCDIMN'                                 # Ancient, Anubis, Cache, Dust2, Inferno, Mirage, Nuke
MAPNAME = dict(A='Ancient', U='Anubis', C='Cache', D='Dust II', I='Inferno', M='Mirage', N='Nuke', O='Overpass')

def load():
    names, rows, sec = {}, [], None
    for line in open(F):
        line = line.strip()
        if line.startswith('#'): sec = line; continue
        if sec == '#NAMES':
            i, n = line.split(',', 1); names[int(i)] = n
        elif line:
            d, a, b, sa, sb, m = line.split(',')
            if sa.isdigit() and sb.isdigit() and int(sa) != int(sb): rows.append((int(d), int(a), int(b), int(sa), int(sb), m))
    return names, rows

def fit(rows, nteams, today, H=60, sR=1.0, sD=0.25, rw=0.0, maps=True, vrs=None):
    # vrs — массив очков VRS по командам (или None): среднее рейтинга команды = β·(VRS−1400)/400, β подбирается
    d = np.array([r[0] for r in rows]); a = np.array([r[1] for r in rows]); b = np.array([r[2] for r in rows])
    sa = np.array([r[3] for r in rows], float); sb = np.array([r[4] for r in rows], float)
    mi = np.array([POOL.find(r[5]) for r in rows])           # −1 = карта не из пула (Overpass) → без поправки
    y = (1 - rw) * (sa > sb) + rw * sa / (sa + sb)
    w = 0.5 ** ((today - d) / H)
    nm = len(POOL)
    v = np.zeros(nteams) if vrs is None else (np.asarray(vrs, float) - 1400) / 400
    sR = np.broadcast_to(np.asarray(sR, float), (nteams,))       # можно задать свою «свободу» каждой команде
    def unpack(x): return x[:nteams], x[nteams:nteams * (1 + nm)].reshape(nteams, nm)
    def f(x):
        R, D = unpack(x); beta = x[-1]; mu = beta * v
        dm = np.where(mi >= 0, D[a, np.maximum(mi, 0)] - D[b, np.maximum(mi, 0)], 0) if maps else 0
        z = R[a] - R[b] + dm; p = 1 / (1 + np.exp(-z))
        ll = -(w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).sum()
        g = w * (p - y)
        gR = np.bincount(a, g, nteams) - np.bincount(b, g, nteams)
        gD = np.zeros((nteams, nm))
        if maps:
            ok = mi >= 0
            np.add.at(gD, (a[ok], mi[ok]), g[ok]); np.add.at(gD, (b[ok], mi[ok]), -g[ok])
        ll += ((R - mu) ** 2 / (2 * sR ** 2)).sum() + (D ** 2).sum() / (2 * sD ** 2)
        gR += (R - mu) / sR ** 2; gD += D / sD ** 2
        gb = -((R - mu) * v / sR ** 2).sum()
        return ll, np.concatenate([gR, gD.ravel(), [gb]])
    x = minimize(f, np.zeros(nteams * (1 + nm) + 1), jac=True, method='L-BFGS-B').x
    R, D = unpack(x); fit.beta = x[-1]
    return R, D

def p_map(R, D, a, b, m):
    k = POOL.find(m); z = R[a] - R[b] + (D[a, k] - D[b, k] if k >= 0 else 0)
    return 1 / (1 + np.exp(-z))

ALIAS = {}                                        # HLTV-имя → имя в VRS, если отличаются
def vrs_vector(names, path):
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from data import standings
    st = standings(path); floor = sorted(p for p, _ in st.values())[len(st) // 2]   # незнакомые команды — медиана VRS
    out = np.zeros(max(names) + 1)
    for i, n in names.items(): out[i] = st.get(ALIAS.get(n, n), (floor, 0))[0]
    return out, [n for n in names.values() if ALIAS.get(n, n) not in st]
