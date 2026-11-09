# Модель вето по истории HLTV (матчи 32 команд с 08.07.2026, пул с Cache).
# Каждое действие (бан/пик) — выбор из оставшихся карт: P(m) ∝ exp(γ·log(частота_команды(действие, m)+α) + λ·преимущество(m)).
# Частоты считаются отдельно для банов и пиков, со сглаживанием α. Параметры подбираются по правдоподобию действий.
import re, numpy as np
from collections import defaultdict
MAPN = {'Ancient': 'A', 'Anubis': 'U', 'Cache': 'C', 'Dust2': 'D', 'Dust II': 'D', 'Inferno': 'I', 'Mirage': 'M', 'Nuke': 'N', 'Overpass': 'O', 'Train': 'T', 'Vertigo': 'V'}
POOL = 'AUCDIMN'

def parse(rows):
    """строки matchPage → [(id, дата, команда1, команда2, bo, lan, [(команда|None, 'ban'|'pick'|'left', карта)])]"""
    out = []
    for r in rows:
        f = r.split('\t')
        if len(f) < 8: continue
        mid, date, t1, _, t2, _, ev, vb = f[:8]
        acts = []
        for team, act, m in re.findall(r'\d\. (.+?) (removed|picked) (Ancient|Anubis|Cache|Dust2|Inferno|Mirage|Nuke|Overpass|Train|Vertigo)', vb):
            acts.append((team, 'ban' if act == 'removed' else 'pick', MAPN[m]))
        left = re.search(r'(Ancient|Anubis|Cache|Dust2|Inferno|Mirage|Nuke|Overpass|Train|Vertigo) was left over', vb)
        if left: acts.append((None, 'left', MAPN[left.group(1)]))
        bo = re.search(r'Best of (\d)', vb); lan = '(LAN)' in vb
        if acts: out.append((mid, date, t1, t2, int(bo.group(1)) if bo else 0, lan, acts))
    return out

class Veto:
    def __init__(self, matches, alpha=1.0, gamma=1.0, lam=0.0):
        self.c = defaultdict(lambda: defaultdict(float)); self.alpha, self.gamma, self.lam = alpha, gamma, lam
        for mid, date, t1, t2, bo, lan, acts in matches:
            for team, act, m in acts:
                if team: self.c[(team, act)][m] += 1
    def probs(self, team, act, left, adv=None):
        cnt = self.c[(team, act)]; tot = sum(cnt[m] for m in left)
        s = np.array([self.gamma * np.log((cnt[m] + self.alpha) / (tot + self.alpha * len(left))) + (self.lam * (adv[m] if adv else 0) * (1 if act == 'pick' else -1)) for m in left])
        e = np.exp(s - s.max()); return dict(zip(left, e / e.sum()))
