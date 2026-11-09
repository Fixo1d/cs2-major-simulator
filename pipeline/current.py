# Данные «на сегодня» для мажора в Сингапуре: карты HLTV (hltv_maps.csv) + LAN-пометки + VRS 02.11 + составы + форма игроков.
import json, os, sys
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(__file__))
from model2 import vrs_points, vrs_rosters, player_table, team_form
from maps import load
from backtest import key, RAW, BUNDLE
ASOF = '2026-11-03'

def rows_current():
    names, rows = load()
    lan = {key(r.split('\t')) for r in BUNDLE['LAN_CUR']}
    out = []
    d0 = date(2026, 5, 4)
    for d, a, b, sa, sb, m in rows:
        ds = (d0 + timedelta(days=d)).isoformat(); r = (ds, names[a], names[b], sa, sb, m)
        out.append((*r, key(r) in lan))
    return out

def setup_current():
    vrs = vrs_points(os.path.join(RAW, 'vrs_global_2026_11_02.md')); ros = vrs_rosters(os.path.join(RAW, 'details'))
    top, al, sh = player_table(BUNDLE['PLAYERS']['CUR_top'], BUNDLE['PLAYERS']['CUR_all'])
    form = {t: team_form(r[0], top, al, sh)[0] for t, r in ros.items()}
    return rows_current(), vrs, ros, form
