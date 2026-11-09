# Вероятность выиграть серию Bo1/Bo3 с учётом вето: каждое действие разыгрывается по модели вето (история команды + сила на карте),
# высший посев выбирает, быть ему командой A или B (берём лучший для него вариант).
import numpy as np, random
POOL = 'AUCDIMN'
BO1 = [('A', 'ban'), ('A', 'ban'), ('B', 'ban'), ('B', 'ban'), ('B', 'ban'), ('A', 'ban')]
BO3 = [('A', 'ban'), ('B', 'ban'), ('A', 'pick'), ('B', 'pick'), ('B', 'ban'), ('A', 'ban')]

def sample_veto(order, ta, tb, pm, veto, rng):
    left = list(POOL); picks = []
    for who, act in order:
        team = ta if who == 'A' else tb
        adv = {m: (pm[m] - .5) if who == 'A' else (.5 - pm[m]) for m in left}
        pr = veto.probs(team, act, left, adv)
        ms = list(pr); m = rng.choices(ms, [pr[x] for x in ms])[0]
        left.remove(m)
        if act == 'pick': picks.append(m)
    return picks + left                        # пики + десайдер

def p_series(ta, tb, bo, pm, veto, n=600, seed=1):
    """P(ta выигрывает), ta — команда A в вето. pm: карта → P(ta берёт карту)"""
    rng = random.Random(seed); tot = 0.0; order = BO1 if bo == 1 else BO3
    for _ in range(n):
        maps = sample_veto(order, ta, tb, pm, veto, rng)
        if bo == 1: tot += pm[maps[0]]
        else:
            p1, p2, p3 = (pm[m] for m in maps[:3]); tot += p1 * p2 + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3
    return tot / n

def best_role(hi, lo, bo, pm_hi, veto, n=600):
    """высший посев hi выбирает роль: возвращает P(hi выигрывает)"""
    as_A = p_series(hi, lo, bo, pm_hi, veto, n)
    pm_lo = {m: 1 - p for m, p in pm_hi.items()}
    as_B = 1 - p_series(lo, hi, bo, pm_lo, veto, n)
    return max(as_A, as_B)
