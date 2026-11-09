# Разбор страниц стадий прошлых мажоров (Liquipedia wikitext): матч → команды (коды Liquipedia), карты со счётом, id матча HLTV.
import json, re, glob, os
DIR = os.path.join(os.path.dirname(__file__), '..', 'raw', 'past')
STAGES = {'Austin 2025': 'BLAST_Major_2025_Austin_Stage_', 'Budapest 2025': 'StarLadder_2025_Major_Stage_', 'Cologne 2026': 'Intel_Extreme_Masters_2026_Cologne_Stage_'}

def score(mp, k):
    return sum(int(x) for x in re.findall(rf'\|(?:o\d)?{k}(?:t|ct)=(\d+)', mp))

def matches(fn):
    t = json.load(open(fn))['parse']['wikitext']['*']
    out = []
    for blk in re.split(r'\|\{\{Match\b', t)[1:]:
        o = re.findall(r'opponent\d=\{\{TeamOpponent\|([^}|]+)', blk)
        h = re.search(r'\|hltv=(\d+)', blk); d = re.search(r'\|date=([^|{]+)', blk)
        maps = []
        for mp in re.findall(r'\{\{Map\|map=([^|]+)\|(.*?)\}\}', blk, re.S):
            if 'finished=true' not in mp[1]: continue
            maps.append((mp[0].strip(), score(mp[1], 't1'), score(mp[1], 't2')))
        if len(o) == 2 and maps:
            w1 = sum(a > b for _, a, b in maps); w2 = sum(b > a for _, a, b in maps)
            out.append(dict(t1=o[0].strip(), t2=o[1].strip(), hltv=h.group(1) if h else '', date=d.group(1).strip() if d else '', maps=maps, bo=3 if len(maps) > 1 or max(w1, w2) == 2 else 1, win=1 if w1 > w2 else 2))
    return out

def all_majors():
    return {(m, s): matches(os.path.join(DIR, f'{p}{s}.json')) for m, p in STAGES.items() for s in (1, 2, 3)}

if __name__ == '__main__':
    A = all_majors(); ids = []
    for k, v in A.items():
        print(k, len(v), 'Bo3:', sum(x['bo'] == 3 for x in v), 'пример:', v[0]['t1'], v[0]['t2'], v[0]['maps'], v[0]['date'])
        ids += [x['hltv'] for x in v if x['hltv']]
    open(os.path.join(DIR, 'hltv_ids.txt'), 'w').write(','.join(ids)); print('hltv ids', len(ids))
