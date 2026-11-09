# Модель v2: карты HLTV + составы (эпохи составов из VRS) + LAN + форма игроков + якорь VRS.
#   logit P(A берёт карту m) = (R_A + Δ_A,m) − (R_B + Δ_B,m)
#   R команды текущего состава ~ N(μ, σ_R),  μ = β1·(VRS−1400)/400 + β2·(рейтинг игроков − 1)·10
#   R старого состава ~ N(R текущего, σ_era · число замен)   — результаты старых пятёрок почти не переносятся
#   вес карты = 0.5^(дней назад / H) × (w_lan, если LAN)
import numpy as np, re, glob, os, json
from scipy.optimize import minimize
from datetime import date as D_

def days(s): y, m, d = map(int, s.split('-')); return D_(y, m, d).toordinal()

# ---------- VRS: очки и история составов ----------
def vrs_points(path):
    pts = {}
    for rk, p, name in re.findall(r'^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|', open(path).read(), re.M): pts.setdefault(name, int(p))
    return pts

def vrs_rosters(folder):
    """команда → [(дата, frozenset(состав))] по матчам из details"""
    out = {}
    for f in glob.glob(folder + '/*.md'):
        t = open(f).read(); name = re.search(r'Team Name: (.+?)<br', t).group(1)
        cur = frozenset(x.strip().lower() for x in re.search(r'Roster: (.+?)<br', t).group(1).split(','))
        rows = re.findall(r'^\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d{4}-\d\d-\d\d)\s*\|.*\|\s*([^|]+?)\s*\|\s*$', t, re.M)
        hist = sorted((days(d), frozenset(x.strip().lower() for x in r.split(','))) for d, r in rows)
        if name not in out or len(hist) > len(out[name][1]): out[name] = (cur, hist)
    return out

# ---------- рейтинги игроков HLTV ----------
def player_table(rows_top, rows_all):
    top = {}; al = {}
    for r in rows_top:
        n, tm, maps, rnd, rt = r.split('\t'); top.setdefault(n.lower(), (float(rt), int(maps)))
    for r in rows_all:
        n, tm, maps, rnd, rt = r.split('\t'); al.setdefault(n.lower(), (float(rt), int(maps)))
    both = [k for k in top if k in al]
    shift = np.mean([al[k][0] - top[k][0] for k in both]) if both else 0.05   # насколько рейтинг «против всех» выше, чем против топ-50
    return top, al, shift

def team_form(roster, top, al, shift):
    vals = []
    for p in roster:
        if p in top and top[p][1] >= 5: vals.append(top[p][0])   # только игры против топ-50: рейтинг против слабых завышен
    return (np.mean(vals) if len(vals) >= 3 else None), len(vals)

# ---------- обучение ----------
class Model:
    def __init__(self, rows, asof, vrs, rosters, form, H=120, sR=1.0, sD=0.15, rw=0.5, w_lan=1.0, s_era=0.25,
                 conn_shrink=True, use_form=True, alias=None):
        """rows: (дата 'YYYY-MM-DD', A, B, раундыA, раундыB, карта, lan)"""
        alias = alias or {}
        T = days(asof); self.asof = T
        rows = [r for r in rows if days(r[0]) < T]
        ent, team_of, era_k = {}, [], []          # сущности: (команда, номер эпохи)
        def roster_at(team, d):
            if team not in rosters: return None
            cur, hist = rosters[team]
            if not hist: return cur
            best = min(hist, key=lambda h: abs(h[0] - d)); return best[1]
        def eid(team, d):
            ro = roster_at(alias.get(team, team), d); cur = rosters.get(alias.get(team, team), (None,))[0]
            k = 0 if ro is None or cur is None else 5 - len(ro & cur)
            key = (team, k)
            if key not in ent: ent[key] = len(ent); team_of.append(team); era_k.append(k)
            return ent[key]
        A = []; Bt = []; Y = []; Wt = []; M = []; LAN = []
        MAPS = 'AUCDIMNOTV'
        for d, a, b, sa, sb, m, lan in rows:
            if sa == sb: continue
            dd = days(d); A.append(eid(a, dd)); Bt.append(eid(b, dd))
            Y.append((1 - rw) * (sa > sb) + rw * sa / (sa + sb)); M.append(MAPS.find(m))
            Wt.append(0.5 ** ((T - dd) / H) * (w_lan if lan else 1.0))
        for t in sorted(set(team_of)):            # у каждой команды есть «текущая» сущность (k=0)
            if (t, 0) not in ent: ent[(t, 0)] = len(ent); team_of.append(t); era_k.append(0)
        n = len(ent); self.ent = ent; self.MAPS = MAPS
        A = np.array(A); Bt = np.array(Bt); Y = np.array(Y); Wt = np.array(Wt); M = np.array(M)
        teams = sorted(set(team_of)); tix = {t: i for i, t in enumerate(teams)}
        cur_of = np.array([ent[(team_of[e], 0)] for e in range(n)]); era = np.array(era_k, float)
        v = np.array([(vrs.get(alias.get(t, t), np.median(list(vrs.values()))) - 1400) / 400 for t in team_of])
        fx = np.array([((form.get(alias.get(t, t)) or 1.0) - 1.0) * 10 if use_form else 0.0 for t in team_of])
        # связность: сколько карт против сильных (VRS ≥ 1500)
        strong = np.array([vrs.get(alias.get(t, t), 0) >= 1500 for t in team_of])
        ntop = np.bincount(A, strong[Bt], n) + np.bincount(Bt, strong[A], n)
        ntop_team = np.zeros(n)
        for e in range(n): ntop_team[cur_of[e]] += ntop[e]
        sRv = (0.3 + 0.7 * np.minimum(1, ntop_team / 20)) * sR if conn_shrink else np.full(n, sR)
        isc = era == 0; nm = len(MAPS); ok = M >= 0
        def f(x):
            R = x[:n]; Dm = x[n:n + n * nm].reshape(n, nm); b1, b2 = x[-2], x[-1]
            Dc = Dm[cur_of]                           # поправки на карты — общие для всех эпох команды
            z = R[A] - R[Bt] + np.where(ok, Dc[A, np.maximum(M, 0)] - Dc[Bt, np.maximum(M, 0)], 0)
            p = 1 / (1 + np.exp(-z)); g = Wt * (p - Y)
            ll = -(Wt * (Y * np.log(p + 1e-12) + (1 - Y) * np.log(1 - p + 1e-12))).sum()
            gR = np.bincount(A, g, n) - np.bincount(Bt, g, n)
            gDc = np.zeros((n, nm)); np.add.at(gDc, (A[ok], M[ok]), g[ok]); np.add.at(gDc, (Bt[ok], M[ok]), -g[ok])
            gD = np.zeros((n, nm)); np.add.at(gD, cur_of, gDc)
            mu = b1 * v + b2 * fx
            # текущие составы: R ~ N(mu, sRv); старые: R ~ N(R_cur, s_era*k)
            rc = np.where(isc, R - mu, 0); ll += (rc ** 2 / (2 * sRv ** 2)).sum()
            gR += np.where(isc, rc / sRv ** 2, 0)
            gb1 = -(np.where(isc, rc / sRv ** 2, 0) * v).sum(); gb2 = -(np.where(isc, rc / sRv ** 2, 0) * fx).sum()
            se = np.maximum(s_era * era, 1e-6); ro = np.where(~isc, R - R[cur_of], 0)
            ll += (ro ** 2 / (2 * se ** 2)).sum()
            gq = np.where(~isc, ro / se ** 2, 0); gR += gq; np.add.at(gR, cur_of, -gq)
            ll += (Dm ** 2).sum() / (2 * sD ** 2); gD += Dm / sD ** 2
            return ll, np.concatenate([gR, gD.ravel(), [gb1, gb2]])
        x = minimize(f, np.zeros(n + n * nm + 2), jac=True, method='L-BFGS-B', options=dict(maxiter=3000)).x
        self.R = x[:n]; self.Dm = x[n:n + n * nm].reshape(n, nm); self.beta = x[-2:]
        self._h = (A, Bt, Wt, M, ok, cur_of, isc, np.full(n, float(sR)), np.maximum(s_era * era, 1e-6), n)   # для неуверенности — без «сжатия» слабо связанных команд: их сила по-настоящему неизвестна

    def cov_R(self):
        """Неуверенность в силе команд (приближение Лапласа): ковариация R из матрицы вторых производных
        (карты + якорь к VRS + связь старых составов с текущим). Поправки на карты и β считаем известными."""
        A, Bt, Wt, M, ok, cur_of, isc, sRv, se, n = self._h
        Dc = self.Dm[cur_of]; k = np.maximum(M, 0)
        z = self.R[A] - self.R[Bt] + np.where(ok, Dc[A, k] - Dc[Bt, k], 0); p = 1 / (1 + np.exp(-z))
        h = Wt * p * (1 - p)
        H = np.zeros((n, n))
        np.add.at(H, (A, A), h); np.add.at(H, (Bt, Bt), h); np.add.at(H, (A, Bt), -h); np.add.at(H, (Bt, A), -h)
        H[np.arange(n), np.arange(n)] += np.where(isc, 1 / sRv ** 2, 0)
        old = np.where(~isc)[0]; c = cur_of[old]; q = 1 / se[old] ** 2
        np.add.at(H, (old, old), q); np.add.at(H, (c, c), q); np.add.at(H, (old, c), -q); np.add.at(H, (c, old), -q)
        return np.linalg.inv(H)

    def sample(self, teams, S, rng):
        """S случайных наборов силы для списка команд (только текущие составы), с учётом корреляций"""
        C = self.cov_R(); idx = [self.ent[(t, 0)] for t in teams]
        mu = self.R[idx]; L = np.linalg.cholesky(C[np.ix_(idx, idx)] + 1e-9 * np.eye(len(idx)))
        return mu + rng.standard_normal((S, len(idx))) @ L.T
        self.mu = lambda t: self.beta[0] * ((vrs.get(alias.get(t, t), np.median(list(vrs.values()))) - 1400) / 400) + self.beta[1] * (((form.get(alias.get(t, t)) or 1.0) - 1.0) * 10 if use_form else 0)

    def strength(self, team):
        e = self.ent.get((team, 0))
        return (self.R[e], self.Dm[e]) if e is not None else (self.mu(team), np.zeros(len(self.MAPS)))

    def p_map(self, a, b, m, K=1.0):
        Ra, Da = self.strength(a); Rb, Db = self.strength(b); k = self.MAPS.find(m)
        return 1 / (1 + np.exp(-K * (Ra - Rb + (Da[k] - Db[k] if k >= 0 else 0))))
