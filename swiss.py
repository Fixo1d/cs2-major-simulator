# Швейцарская система мажора по правилам Valve (major-supplemental-rulebook.md):
# 1-й тур 1–9, 2–10 …; 2–3 туры — внутри группы с одинаковым счётом высший посев против низшего без повторов;
# 4–5 туры — таблица приоритетов для группы из 6. Посев внутри стадии: счёт → Difficulty (Бухгольц) → начальный посев.
# Матчи на выход/вылет — Bo3, остальные — Bo1.
import random

PRIO6 = [((1,6),(2,5),(3,4)),((1,6),(2,4),(3,5)),((1,5),(2,6),(3,4)),((1,5),(2,4),(3,6)),((1,4),(2,6),(3,5)),
         ((1,4),(2,5),(3,6)),((1,6),(2,3),(4,5)),((1,5),(2,3),(4,6)),((1,3),(2,6),(4,5)),((1,3),(2,5),(4,6)),
         ((1,4),(2,3),(5,6)),((1,3),(2,4),(5,6)),((1,2),(3,6),(4,5)),((1,2),(3,5),(4,6)),((1,2),(3,4),(5,6))]

def pair_high_low(order, played):
    """Высший против самого низкого доступного без повтора; с откатом, если тупик."""
    if not order: return []
    top = order[0]
    for j in range(len(order) - 1, 0, -1):
        o = order[j]
        if o in played[top]: continue
        rest = pair_high_low(order[1:j] + order[j + 1:], played)
        if rest is not None: return [(top, o)] + rest
    return None

def pair_table(order, played):
    for row in PRIO6:
        ps = [(order[a - 1], order[b - 1]) for a, b in row]
        if all(y not in played[x] for x, y in ps): return ps
    return pair_high_low(order, played)

def run_stage(teams, play, rng=random):
    """teams — список в порядке начального посева (1..16). play(a, b, bo, rng) → победитель (a — высший посев).
    Возвращает {команда: (победы, поражения)} и итоговый посев."""
    W = {t: 0 for t in teams}; L = {t: 0 for t in teams}; seed = {t: i for i, t in enumerate(teams)}
    played = {t: set() for t in teams}
    def diff(t): return sum(W[o] - L[o] for o in played[t])
    for rnd in range(1, 6):
        active = [t for t in teams if W[t] < 3 and L[t] < 3]
        groups = {}
        for t in active: groups.setdefault((W[t], L[t]), []).append(t)
        pairs = []
        for (w, l), g in sorted(groups.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
            g.sort(key=lambda t: (-diff(t), seed[t]))
            if rnd == 1: g.sort(key=lambda t: seed[t]); pairs += [(g[i], g[i + len(g) // 2]) for i in range(len(g) // 2)]
            elif rnd in (2, 3): pairs += pair_high_low(g, played)
            else: pairs += pair_table(g, played)
        for a, b in pairs:
            bo = 3 if (W[a] == 2 or L[a] == 2) else 1
            win = play(a, b, bo, rng); lose = b if win == a else a
            W[win] += 1; L[lose] += 1; played[a].add(b); played[b].add(a)
    final = sorted(teams, key=lambda t: (-W[t], L[t], -diff(t), seed[t]))
    return {t: (W[t], L[t]) for t in teams}, final

if __name__ == '__main__':
    # самопроверка: равные команды → каждая запись встречается с ожидаемой частотой (3-0: 12,5 %, 3-1: 18,75 %, 3-2: 18,75 %)
    from collections import Counter
    T = [f't{i}' for i in range(16)]; c = Counter()
    for _ in range(20000):
        res, _ = run_stage(T, lambda a, b, bo, r: a if r.random() < .5 else b)
        c.update(res.values())
    print({k: round(v / 20000 / 16 * 100, 2) for k, v in sorted(c.items())})
