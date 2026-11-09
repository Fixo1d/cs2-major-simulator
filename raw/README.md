# Исходные данные (в репозиторий не входят)

Нужны только для `pipeline/` — обучения модели заново. Для `simulate.py` не нужны.

| Файл | Откуда | Формат |
|---|---|---|
| `vrs_global_2026_11_02.md` (+ `_2026_08_03`, `_2026_09_07`, `_2026_10_05`, `_2025_06_02`, `_2025_11_03`, `_2026_06_01`) | [Valve, GitHub](https://github.com/ValveSoftware/counter-strike_regional_standings), `live/<год>/standings_global_<дата>.md` | как у Valve: строки таблицы `\| место \| очки \| команда \| …` |
| `details/*.md` (+ `details_2025_06_02/`, `details_2025_11_03/`, `details_2026_06_01/` для прошлых мажоров) | Valve, `live/<год>/details/<дата>/` — файлы 32 участников | как у Valve: `Team Name: …<br>`, `Roster: a, b, c, d, e<br>`, таблица матчей с датой и соперником |
| `hltv/hltv_maps.csv` | HLTV, карты за 6 мес. до мажора | секция `#NAMES` — строки `номер,команда`; секция `#MAPS` — `день_от_2026-05-04,номер1,номер2,раунды1,раунды2,карта` |
| `hltv/bundle.json` | HLTV | словарь, строки через табуляцию (см. ниже) |
| `past/*.json` | Liquipedia API, `action=parse&prop=wikitext` — страницы стадий Austin 2025, Budapest 2025, Cologne 2026 | ответ API как есть |

Карты кодируются одной буквой: A Ancient, U Anubis, C Cache, D Dust II, I Inferno, M Mirage, N Nuke, O Overpass, T Train, V Vertigo.

## `hltv/bundle.json`

| Ключ | Строка (поля через табуляцию) |
|---|---|
| `LAN_CUR` | `дата  команда1  команда2  раунды1  раунды2  карта` — LAN-карты за полгода до Сингапура |
| `MAPS_<мажор>`, `TEAMMAPS_<мажор>`, `LAN_<мажор>` | то же, за полгода до мажора (`Austin`, `Budapest`, `Cologne`) |
| `PLAYERS` → `CUR_top`, `CUR_all`, `<мажор>_top`, `<мажор>_all` | `ник  команда  карт  раундов  рейтинг` — против топ-50 / против всех, за 3 мес. |
| `PASTMATCH` | `id  дата  команда1  id1  команда2  id2  турнир  текст вето` — матчи прошлых мажоров |
| `VETO` | то же для матчей с 08.07.2026: текст вето вида `1. X removed Dust2 2. Y picked Mirage … 7. Ancient was left over` |
