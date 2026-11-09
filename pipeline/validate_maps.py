# Проверка модели по картам «вперёд по времени»: учим на картах до дня X, предсказываем карты следующих 3 недель
# между участниками мажора. Несколько срезов (сентябрь–октябрь), считаем log-loss и % угаданных.
import numpy as np, itertools
from maps import load, fit, p_map, vrs_vector
names, rows = load(); N = max(names) + 1
MAJOR = {'Ninjas in Pyjamas','PARIVISION','HEROIC','FaZe','Liquid','Eternal Fire','100 Thieves','9z','M80','DENDELE','Imperial','Wildcard',
 'BC.Game','FlyQuest','Chinggis Warriors','Lynn Vision','MOUZ','Vitality','FURIA','Aurora','Spirit','BETBOOM','Falcons','Legacy','FUT','Nemiga',
 'G2','1win','Natus Vincere','Astralis','MIBR','The MongolZ'}
M = {i for i, n in names.items() if n in MAJOR}
cuts = [120, 141, 162]                        # 01.09, 22.09, 13.10
VR = {120: '../raw/vrs_global_2026_08_03.md', 141: '../raw/vrs_global_2026_09_07.md', 162: '../raw/vrs_global_2026_10_05.md'}
USE_VRS = True
def evaluate(**kw):
    L, A, n = [], [], 0
    for c in cuts:
        tr = [r for r in rows if r[0] < c]; te = [r for r in rows if c <= r[0] < c + 21 and r[1] in M and r[2] in M]
        R, D = fit(tr, N, c, vrs=vrs_vector(names, VR[c])[0] if USE_VRS else None, **kw)
        for d, a, b, sa, sb, m in te:
            p = p_map(R, D, a, b, m); win = sa > sb
            L.append(np.log(p if win else 1 - p)); A.append((p > .5) == win)
    return np.mean(L), np.mean(A), len(L)
print('монетка %.4f' % np.log(.5))
res = []
if __name__ == '__main__':
  print('нет в VRS:', vrs_vector(names, VR[162])[1][:40])
  for H, sR, sD, rw, mp in itertools.product((60, 120, 999), (0.25, 0.5, 1.0), (0.15, 0.3), (0.0, 0.5), (False, True)):
    if not mp and sD != 0.15: continue
    ll, acc, n = evaluate(H=H, sR=sR, sD=sD, rw=rw, maps=mp)
    res.append((ll, acc, n, H, sR, sD, rw, mp))
  res.sort(reverse=True)
  for r in res[:12]: print('ll %.4f угадано %.1f%% n=%d | H=%d sR=%.1f sD=%.2f rw=%.1f карты=%s' % (r[0], 100 * r[1], *r[2:]))
  print('без карт лучшее:', [('%.4f' % r[0], r[3:]) for r in res if not r[7]][:3])
