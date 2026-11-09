# Сравнение вариантов модели на трёх прошлых мажорах (только данные до начала турнира).
import time, numpy as np, backtest as bt
bt.DATA = {k: bt.setup(k) for k in bt.CFG}
BASE = dict(H=120, sR=1.0, sD=0.15, rw=0.5)
VARIANTS = [
    ('v1: как сейчас (без составов, LAN, игроков)', dict(BASE, w_lan=1.0, s_era=10.0, use_form=False)),
    ('+ составы', dict(BASE, w_lan=1.0, s_era=0.25, use_form=False)),
    ('+ LAN x2', dict(BASE, w_lan=2.0, s_era=10.0, use_form=False)),
    ('+ форма игроков', dict(BASE, w_lan=1.0, s_era=10.0, use_form=True)),
    ('все три', dict(BASE, w_lan=2.0, s_era=0.25, use_form=True)),
]
print('монетка: log-loss 0.6931')
for name, p in VARIANTS:
    t = time.time(); r = bt.evaluate(p)
    print(f'{name:45s} карты: ll {r[0]:.4f} угадано {100*r[1]:.1f}% (n={r[2]}) | серии: ll {r[3]:.4f} угадано {100*r[4]:.1f}% (n={r[5]})  [{time.time()-t:.0f} c]')
