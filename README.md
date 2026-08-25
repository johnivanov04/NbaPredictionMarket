# NbaPredictionMarket — Historical Data Foundation

A reproducible, auditable research dataset built from NBA game results and
Kalshi NBA game-winner markets.

* **Phase 1** — ingest both sources, normalize them, and join them
  deterministically.
* **Phase 2** — extract the executable Kalshi quote and market-implied
  probability exactly *N* minutes before each game's scheduled tipoff.
* **Phase 3A0** — expand the NBA side to 20 seasons (2006-07 … 2025-26) and
  audit it: season structure, franchise identity, and chronology.
* **Phase 3A1** — lookahead-safe sequential features, forecasting baselines
  (constant, Elo, logistic), history-window selection, and a single 2025-26
  holdout evaluation against the Kalshi benchmark.
* **Phase 3A2** — improved team-strength representation: margin-of-victory Elo,
  opponent-adjusted margin, and a predetermined feature-bundle ablation.
* **Phase 3A3A0** — paid (GOAT-tier) data capability audit and historical
  ingestion of player-game and advanced stats. Data foundation only.
* **Phase 3A3** — lagged possession-adjusted efficiency and rotation features
  built from the paid feeds, ablated against the Phase 3A2 control.
* **Phase 3A3B0** — availability source audit and the prospective capture
  foundation for 2026-27. No model.

There is deliberately no model, no frontend, no database service, no trading
logic, and no execution system here. The goal is a dataset you can *trust*
before anything is built on top of it.

## What it does

1. Downloads every NBA game for a season from **BALLDONTLIE** (`GET /v1/games`).
2. Downloads every **Kalshi** `KXNBAGAME` market from *both* Kalshi stores
   (the historical archive and the live markets endpoint) and deduplicates them.
3. Preserves every raw API response verbatim under `data/raw/`.
4. Normalizes both sources into clean, typed tables.
5. Matches NBA games to Kalshi events on `(scheduled date, unordered team pair)`.
6. Writes a match report classifying every record as matched, unmatched, or
   ambiguous — with counts and examples.

## Setup

Requires Python 3.11+.

```bash
git clone <this repo> && cd NbaPredictionMarket

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"            # omit [dev] if you don't need tests/lint
```

### Configure credentials

```bash
cp .env.example .env
```

Then edit `.env` and set your key:

```
BALLDONTLIE_API_KEY=your_key_here
```

Get a free key at <https://app.balldontlie.io>. **Kalshi needs no credentials** —
the market metadata endpoints used here are public.

`.env` is gitignored. The key is also read from the plain environment, so
`BALLDONTLIE_API_KEY=... python -m ...` works too.

### Troubleshooting: `No module named 'nba_prediction_market'`

If the editable install succeeds but the import still fails, the `.pth` file
that `pip install -e .` writes has probably picked up macOS's hidden flag —
Python 3.13's `site.py` silently skips hidden `.pth` files:

```bash
ls -lO .venv/lib/python3.13/site-packages/*.pth   # look for "hidden"
chflags nohidden .venv/lib/python3.13/site-packages/*.pth
```

On some macOS setups something re-applies that flag, in which case set the path
explicitly instead — this always works and needs no install at all:

```bash
PYTHONPATH=src python -m nba_prediction_market.pipelines.build_dataset --season 2025
```

`pytest` is unaffected either way: the repo-root `conftest.py` puts `src` on the
path directly.

## Run Phase 1

One command runs the whole pipeline:

```bash
python -m nba_prediction_market.pipelines.build_dataset --season 2025
```

`--season 2025` means the **2025-26** season (BALLDONTLIE labels a season by its
starting year). The pipeline verifies this from the returned dates and *fails*
rather than proceeding if they don't look like the requested season.

An installed console script is equivalent:

```bash
nba-pm-build --season 2025
```

### Options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--season` | `2025` | BALLDONTLIE season start year |
| `--series-ticker` | `KXNBAGAME` | Kalshi series to ingest |
| `--data-dir` | `data` | Root for raw/processed/report output |
| `--no-csv` | off | Write only parquet, skip the CSV copies |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

A full run takes a few minutes, almost entirely because the BALLDONTLIE free
tier allows 5 requests/minute and the client throttles itself to stay under it.

## Output files

```
data/
  raw/
    nba/      balldontlie_games_season_2025_<UTC timestamp>.json
    kalshi/   kalshi_markets_KXNBAGAME_<UTC timestamp>.json
              kalshi_events_KXNBAGAME_<UTC timestamp>.json
              kalshi_historical_cutoff_<UTC timestamp>.json
  processed/
    nba_games_2025_26.parquet          (+ .csv)
    kalshi_nba_markets_2025_26.parquet (+ .csv)
    nba_kalshi_matches_2025_26.parquet (+ .csv)
    kalshi_nba_events_2025_26.parquet  (+ .csv)
  reports/
    match_report.json
```

Parquet is canonical (it preserves types); the CSVs are convenience copies.
Raw files are timestamped per run and never overwritten, so any processed table
can be rebuilt without re-hitting the APIs. **Everything under `data/` is
gitignored** — it is all regenerable.

### The tables

**`nba_games_*`** — one row per NBA game: source id, date, `tipoff_utc`, season,
status, period, `postseason`, both teams (id / abbreviation / full name /
canonical code), both scores, and `home_win`.

`home_win` is populated **only** when the game is final *and* both scores are
present *and* they are not level. Anything else stays null. Nothing is inferred.

**`kalshi_nba_markets_*`** — one row per Kalshi market (two per game, one per
team): ticker, event ticker, titles, `open_time_utc` / `close_time_utc` /
`expiration_time_utc` / `settlement_ts_utc`, status, result, volume, liquidity,
prices, plus derived `home_team_code` / `away_team_code` / `market_team_code`
and `market_team_is_home`.

**`kalshi_nba_events_*`** — the markets collapsed to one row per game-event,
with `home_market_ticker` and `away_market_ticker` side by side.

**`nba_kalshi_matches_*`** — the join. Every NBA game and every Kalshi event
appears at least once, classified as `matched`, `unmatched_nba`,
`unmatched_kalshi`, or `ambiguous`.

Attaching a Kalshi price to a game later is a filter plus a lookup:

```python
import pandas as pd

matches = pd.read_parquet("data/processed/nba_kalshi_matches_2025_26.parquet")
usable = matches[matches["match_status"] == "matched"]
# usable["kalshi_home_market_ticker"] -> the ticker to pull candlesticks for
```

**`match_report.json`** — counts per category, up to five worked examples of
each, per-tier match counts, cross-source quality checks, the season
verification result, and provenance for every raw file the run wrote.

## Matching

The join key is **`(scheduled game date, unordered pair of canonical team
codes)`** — e.g. `2025-10-21|HOU|OKC`.

* Both sources label a game by its **local scheduled date** (BALLDONTLIE in
  `date`, Kalshi in the settlement rules text), so no timezone shifting is
  applied to the key. All *timestamps* are UTC.
* The team pair is **unordered on purpose**. Home/away is recorded and compared
  afterwards (`orientation_agrees`), so a disagreement surfaces as a flag rather
  than silently dropping an otherwise obvious match.
* **Tier 1** matches only when exactly one game and exactly one event share a key.
* **Tier 2** allows a ±1 calendar-day difference for the same team pair, and only
  fires when the pairing is *mutually* unique among still-unmatched records.
* Any remaining many-to-one or one-to-many group is reported `ambiguous` **in
  full**, with every candidate listed. A match is never chosen arbitrarily.

Matched rows also carry `settlement_agrees_with_score`, which cross-checks the
Kalshi settlement against the final NBA score.

Team names resolve through an **exact** alias table covering all 30 franchises
(`matching/team_names.py`). There is no fuzzy matching and no "closest match".
A string either resolves to exactly one franchise or it resolves to nothing.
`"Los Angeles"` and `"LA"` resolve to `ambiguous` rather than picking a side.
An unrecognised NBA team **fails the run** instead of being dropped.

## Tests

```bash
pytest                  # unit tests only (no network)
pytest --cov            # with coverage
ruff check .            # lint
```

Unit tests never touch the network — HTTP is served by `httpx.MockTransport`,
and the fixtures in `tests/conftest.py` are trimmed copies of real captured
responses. They cover pagination, retry/rate-limit behaviour, team
normalization, matching (including ambiguity and determinism), duplicate
handling, and the pipeline end to end.

Two suites are deselected by default:

```bash
pytest -m integration   # hits the real APIs; needs BALLDONTLIE_API_KEY
pytest -m dataset       # asserts invariants of the generated data/ artefacts
```

`-m dataset` is the one to run after regenerating: it pins the regular season at
1,230 games with 82 per team, and asserts that no play-in or NBA Cup final game
entered the primary Phase 2 dataset.

They assert the *shape* of the responses, so an upstream schema change gets
caught rather than silently corrupting a dataset.

## Verified API behaviour

Checked against the live APIs on 2026-08-19. These are the non-obvious findings
the code is built around:

**BALLDONTLIE**

* Auth is a **bare API key** in the `Authorization` header — no `Bearer` prefix.
* Cursor pagination via `meta.next_cursor`; `per_page` caps at 100.
* Season `2025` returns the 2025-26 season. The pipeline verifies this from the
  returned dates rather than trusting the label.

**Kalshi**

* `GET /markets?status=all` is **rejected with HTTP 400** despite appearing in
  the docs. Omitting the filter entirely returns every status, so that is what
  the client does.
* `no_sub_title` **equals** `yes_sub_title` on every market observed. It does
  *not* name the opposing team, so it cannot be used to derive an opponent.
  An integration test guards this.
* `occurrence_datetime` is only populated for postseason markets, so it cannot
  be the primary date field. The scheduled date comes from the settlement rules
  text (which carries an explicit year), cross-checked against the event ticker.
* Market titles use **two** formats: `"A at B Winner?"` (orientation implied) and
  `"A vs B Winner?"` (no orientation). Orientation is therefore taken from the
  event's `sub_title` (`"NYK at SAS (Jun 13)"`, abbreviations, complete
  coverage), with the structured event ticker as fallback.
* The `KXNBAGAME` archive spans **multiple seasons** and is not season-filterable
  server-side, so markets are scoped client-side to the season window.

**Kalshi candlesticks (Phase 2)**

* **The two candlestick endpoints return the same data under different field
  names.** `/historical/markets/{ticker}/candlesticks` uses bare names
  (`volume`, `open_interest`, `price.close`, `yes_bid.close`); the live
  `/series/{s}/markets/{ticker}/candlesticks` suffixes every one
  (`volume_fp`, `open_interest_fp`, `price.close_dollars`,
  `yes_bid.close_dollars`). Parsing accepts both, so routing between tiers
  cannot silently produce null columns. An integration test guards each.
* **`price.close` is `null` whenever no trade occurred in that minute**, while
  `yes_bid` / `yes_ask` stay fully populated and `price.previous` still carries
  the last traded price. This happened on ~5% of selected candles. It is the
  concrete reason bid/ask must be preserved separately from trade price, and why
  `last_trade_price` cannot serve as an entry price.
* Prices arrive as **decimal-dollar strings already in `[0, 1]`** (`"0.6500"`),
  not cents. Nothing is rescaled; values outside `[0, 1]` are rejected rather
  than clamped.
* The window is **inclusive of both bounds** — a 60-minute request returns 61
  one-minute candles.
* `period_interval` must be one of `{1, 60, 1440}`; anything else is a HTTP 400.
  The client validates before sending.
* An unknown ticker returns **404**, which is what drives the archive-to-live
  fallback.
* Candlesticks need **no authentication**.

## Phase 2 — pregame quotes at T-minus-30

Answers one question per game: **what were the executable Kalshi quotes and the
market-implied probability exactly 30 minutes before scheduled tipoff?**

```bash
python -m nba_prediction_market.pipelines.build_pregame_quotes \
    --season 2025 --minutes-before-tip 30 --max-quote-age-minutes 10
```

Requires the Phase 1 outputs to exist; it fails with an actionable message if
they do not. A cold run takes ~20 minutes (2,472 rate-limited requests); re-runs
are ~3 seconds because every response is cached.

### Options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--season` | `2025` | Season start year |
| `--minutes-before-tip` | `30` | Anchor offset before tipoff |
| `--max-quote-age-minutes` | `10` | Staleness limit for a usable quote |
| `--lookback-minutes` | `60` | Candle window length |
| `--period-interval` | `1` | Candle granularity (minutes) |
| `--refresh` | off | Ignore cached responses and refetch |
| `--limit` | none | Process only the first N games (smoke runs) |
| `--no-csv` | off | Parquet only |

### Scope

The primary dataset covers **matched games whose `game_phase` is
`regular_season`** — 1,230 games for 2025-26.

Selection is on the explicit phase label, **not** on `postseason == False`. Two
kinds of game carry `postseason = False` without being regular-season games:

* the **six Play-In games** (2026-04-14 … 04-17), and
* the **NBA Cup Championship** (SAS at NYK, 2025-12-16).

Filtering on `postseason == False` admitted all of these and produced 1,236
eligible games instead of 1,230. See "Game phases" below.

Every phase is preserved in Phase 1's tables — nothing is deleted, and Play-In
games can be modelled later by selecting `game_phase == 'play_in'`.

### Game phases

`nba_games_*` and `nba_kalshi_matches_*` carry an explicit `game_phase` column
(`ingestion/game_phase.py`), one of:

| phase | 2025-26 count | how it is identified |
| --- | --- | --- |
| `regular_season` | 1,230 | everything not below (82 per team × 30 ÷ 2) |
| `play_in` | 6 | falls in the declared Play-In window |
| `playoffs` | 85 | BALLDONTLIE `postseason == True` |
| `nba_cup_championship` | 1 | BALLDONTLIE `ist_stage == "Championship"` |
| `unclassified` | 0 | undeclared season, missing date, or an unknown gap |

The playoff flag and the NBA Cup final come straight from API fields. **The
Play-In has no field at all** — BALLDONTLIE marks those games
`postseason = False` with `ist_stage = None`, indistinguishable from a regular
game except by date. That forces a season-specific calendar boundary, which is
declared once in `SEASON_PHASE_BOUNDARIES` with its provenance:

```python
2025: SeasonPhaseBoundaries(
    regular_season_end=date(2026, 4, 12),
    play_in_start=date(2026, 4, 14),
    play_in_end=date(2026, 4, 17),
    playoffs_start=date(2026, 4, 18),
)
```

Two safeguards keep that from being a magic number:

1. **An undeclared season is never guessed.** It classifies as `unclassified`,
   and Phase 2 then selects zero games rather than silently treating Play-In
   games as regular season.
2. **The declared dates are audited, not trusted.** `verify_regular_season`
   re-derives the league invariant — 30 teams, 82 games each, 1,230 total — from
   the classified data. A wrong boundary date breaks the invariant and is
   reported in `pregame_t30_report.json` under `phase_selection`. Note the
   NBA Cup *group, quarterfinal, and semifinal* games do count toward the 82;
   only the final does not.

Also note: `ist_stage` distinguishes NBA Cup games generally, so the Cup's
knockout rounds can be separated later if wanted — they are currently counted as
regular season, which is correct for standings.

### Output

```
data/processed/nba_kalshi_pregame_t30_2025_26.parquet   (+ .csv)
data/reports/pregame_t30_report.json
data/raw/kalshi/candlesticks/t30_lb60_p1/<game date>/<market ticker>.json
```

**One row per NBA game**, not per Kalshi market. 50 columns: the NBA game and
its outcome, the anchor (`prediction_ts_utc`), match provenance, then a full
`home_*` and `away_*` quote block (bid, ask, midpoint, spread, last and previous
trade price, volume, open interest, quote timestamp, age, usability, issue
code), then quality fields.

### How the quote is chosen

For each of the two team markets:

1. `prediction_ts_utc = game_datetime_utc - minutes_before_tip`. **The NBA
   tipoff is the source of truth** — Kalshi close/settlement times are never
   used as the anchor.
2. Fetch 1-minute candles for `[prediction_ts - 60min, prediction_ts]`.
3. Discard every candle with `end_period_ts > prediction_ts`. This is the single
   chokepoint that prevents lookahead, and it is applied before anything else.
4. Take the most recent remaining candle that actually carries a quote. Values
   are never forward-filled from a later candle.
5. `quote_age_seconds = prediction_ts - candle end_period_ts`, always recorded —
   even for unusable quotes, so staleness is measurable rather than invisible.
6. A quote older than `--max-quote-age-minutes` is **kept for diagnostics but
   marked unusable**. Age exactly equal to the limit is still usable.

`quote_usable` requires a fresh quote *and* both sides present, because the
midpoint is the stated probability benchmark and is only defined two-sided.
`midpoint` and `spread` are computed **only** when both bid and ask exist, and
are never synthesised from trade prices.

### Price semantics — read before modelling

* **`market_midpoint`** — `(bid + ask) / 2`. The market probability benchmark.
* **`yes_ask`** — approximate immediate price to *buy* YES.
* **`yes_bid`** — approximate immediate price to *sell* YES.
* **`last_trade_price` is NOT an executable entry price.** It is whatever last
  traded in that minute, and it is **null in ~5% of rows** because no trade
  occurred (see "Verified API behaviour"). Using it as an entry price would be
  both unfillable and biased toward liquid games.

`market_midpoint_sum` is deliberately **not** normalised to 1. The observed
deviation is data worth looking at, not noise to be scaled away.

## Phase 2 run results (2025-26)

From a live run on 2026-08-19 at T-30 minutes:

| | Value |
| --- | --- |
| Eligible games (matched, regular season) | 1,230 |
| **Both sides usable** | **1,230 (100.0%)** |
| Home-only / away-only / neither usable | 0 / 0 / 0 |
| Missing, stale, malformed, failed quotes | 0 |
| Candles selected after the anchor | **0** |

**Quote age:** 2,445 of 2,460 quotes are 0 seconds old (a candle lands exactly
on the anchor); 11 are 60s, 4 are 120s. Max 120s against a 600s limit — nothing
came close to stale.

**Spread:** 1 cent on 2,306 quotes, 2 cents on 154. No crossed or zero-width
books.

**Midpoint sum:** min 0.980, max 1.020, exactly 1.000 on 754 of 1,230 games.
Deviation from 1 exceeds 0.01 on 15 games (1.2%), exceeds 0.02 on **0**. The
spread is 1-2 cents on each side, so a 1-2 cent deviation is the expected
granularity artefact rather than a data problem.

**Calibration** (the strongest end-to-end check that the data is correct and
correctly oriented):

| home midpoint bucket | n | mean midpoint | actual home win rate |
| --- | --- | --- | --- |
| 0.0-0.1 | 24 | 0.076 | 0.000 |
| 0.1-0.2 | 71 | 0.156 | 0.127 |
| 0.2-0.3 | 97 | 0.249 | 0.258 |
| 0.3-0.4 | 149 | 0.353 | 0.356 |
| 0.4-0.5 | 170 | 0.448 | 0.494 |
| 0.5-0.6 | 160 | 0.553 | 0.519 |
| 0.6-0.7 | 187 | 0.651 | 0.679 |
| 0.7-0.8 | 170 | 0.750 | 0.771 |
| 0.8-0.9 | 159 | 0.850 | 0.805 |
| 0.9-1.0 | 43 | 0.928 | 0.977 |

Mean home midpoint **0.5519** vs actual home win rate **0.5545** — a 0.003 gap
across 1,230 games. Brier score 0.1946 against a 0.25 always-0.5 baseline. Home
favourites won 71.1% of their games; a mirrored orientation would show 28.9%.

## Phase 3A0 — 20-season historical expansion

Builds a lookahead-safe regular-season dataset for **2006-07 … 2025-26** plus the
audits needed to trust it. No model features, no Kalshi data.

```bash
python -m nba_prediction_market.pipelines.build_history --seasons 2006-2025
```

Seasons are cached one file per season under `data/raw/nba/seasons/` and never
refetched unless `--refresh` is passed, so a run can be interrupted and resumed.
A cold run takes ~50 minutes on the free tier (rate-limited); re-runs are
seconds.

### Output

```
data/processed/nba_regular_season_games_2006_26.parquet   (+ .csv)  regular season only
data/processed/nba_all_games_2006_26.parquet              (+ .csv)  every phase preserved
data/processed/nba_team_identity_2006_26.parquet          (+ .csv)  identity audit
data/reports/historical_nba_2006_26_report.json
```

### Not every season is 1,230 games

The 30×82/2 invariant is only correct for a *standard* season. Three of the
twenty are not, and forcing them to 1,230 would either drop real games or invent
missing ones. Each season declares its own structure in
`ingestion/season_metadata.py` with the reason and evidence attached:

| season | structure | expected games | per team | why |
| --- | --- | --- | --- | --- |
| 2011-12 | shortened | 990 | 66 | lockout; season opened 25 Dec 2011 |
| 2012-13 | interrupted | 1,229 | 82 (BOS/IND 81) | BOS v IND cancelled after the Boston Marathon bombing, never made up |
| 2019-20 | interrupted | *no uniform total* | *non-uniform* | COVID suspension; only 22 teams resumed in the bubble |
| 2020-21 | shortened | 1,080 | 72 | COVID-shortened schedule |
| all 16 others | standard | 1,230 | 82 | — |

For 2019-20 no uniform expectation is asserted at all, because teams genuinely
finished on different game counts. The audit checks what *can* be checked (30
teams, dates inside the declared window) and reports the rest rather than
asserting a number nobody can justify.

### Verified API behaviour (historical)

Four findings from the 20-season ingest, each of which silently corrupted the
data before it was handled:

**1. `ist_stage` is only populated for 2025-26.** The NBA Cup existed in 2023-24
and 2024-25 too, but those seasons return `ist_stage = null` for every game — so
their Cup finals carried no marker and were counted toward the regular season
(1,231 games, with the two finalists on 83). Each final is a standalone event —
**the only game played league-wide that day** — so the date is declared per
season and is self-validating. 2025-26, where both routes are available, is the
cross-check that they agree.

**2. `postseason` is unreliable for play-in games.** It is `True` for the 2019-20
and 2021-22 play-in games, `False` from 2022-23 onward, and *both values within
2020-21* (5 `True`, 1 `False`). The declared play-in window therefore takes
**precedence over the flag**; playoffs cannot fall inside it because
`SeasonInfo` enforces `play_in_end < playoffs_start`. Before this, 12 play-in
games were mislabelled as playoffs.

**3. `date` is the *scheduled* date, not always the played date.** For 49 games —
32 in 2020-21, 16 in 2021-22, 1 in 2022-23 — `date` still holds the original
schedule after a COVID postponement while `datetime` holds the actual tipoff,
diverging by up to 116 days. Ordering by `date` produces **5 physically
impossible cases** of a team playing twice in one day; ordering by
`game_datetime_utc` produces none. `game_datetime_utc` is therefore the only
valid sort key, and `tipoff_date_matches_scheduled_date` flags the 49.
The `postponed` flag is never set (0 games league-wide) and cannot be used.

**4. Four games have impossible tied "final" scores.** Games 28012 (2011-12),
32587 (2015-16), 34714 (2016-17) and 48851 (2018-19) each report equal scores
with `status = "Final"`. Game 28012's quarter scores are byte-identical between
the two teams and sum to 98 against a reported 123 — internally impossible.
These are overtime games whose stored score is an end-of-period snapshot. The
winner is **not recoverable**, so `home_win` is null and the rows are preserved
and reported rather than dropped or guessed.

### Era gating

Modern rules are never projected backwards:

* **Play-In** did not exist before 2019-20. Seasons before that declare no
  play-in window, so no game can be classified `play_in` — a mid-April 2007 game
  is simply a regular-season game.
* **NBA Cup** began in 2023-24. Before that `ist_stage` is always null, so
  `nba_cup_championship` is unreachable.

### Game phases

`game_phase` extends the Phase 1/2 vocabulary with `other_special`:

| phase | how it is identified |
| --- | --- |
| `other_special` | a team id outside the 30 franchises (exhibition opponent) |
| `playoffs` | `postseason == True` |
| `nba_cup_championship` | `ist_stage == "Championship"` |
| `play_in` | inside the season's declared play-in window (2019-20 onward) |
| `regular_season` | inside the season's declared regular-season window |
| `unclassified` | undeclared season, missing date, or an unknown gap |

The modelling dataset is `game_phase == 'regular_season'` only. Every other phase
is preserved in `nba_all_games_*`.

### Franchise identity — an empirical finding

**BALLDONTLIE returns present-day franchise identity for every era.** Verified
against the live API:

| historical reality | what `/v1/games` returns |
| --- | --- |
| Seattle SuperSonics (through 2007-08) | `id=21 OKC "Oklahoma City Thunder"` |
| Charlotte Bobcats (2004-2014) | `id=4 CHA "Charlotte Hornets"` |
| New Orleans Hornets (through 2012-13) | `id=19 NOP "New Orleans Pelicans"` |
| New Jersey Nets (through 2011-12) | `id=3 BKN "Brooklyn Nets"` |

`/v1/teams` contains no SuperSonics, Bobcats, or New Orleans Hornets entry,
confirming this is normalization rather than per-era records.

Two consequences:

1. **No relocation mapping is needed.** The source team id is already a stable
   canonical franchise id across all 20 seasons, so Elo and other sequential
   features carry across relocations automatically. `matching/franchises.py`
   documents the 30 ids and their historical identities rather than building
   redundant machinery.
2. **Historical display names are not recoverable from this source.** A 2007-08
   Sonics game is labelled "Oklahoma City Thunder". That is correct for franchise
   continuity and wrong for historical presentation. Documented rather than
   patched — inventing era-accurate names would be fabricating data.

Ids outside 1-30 (defunct 1940s clubs, international exhibition opponents) are
deliberately *not* franchises, which is what lets exhibition games be identified.

### Audits in the report

* **Per season** — raw games returned, counts by phase, teams, games-per-team
  distribution, first/last regular-season date, duplicate ids, missing scores and
  datetimes, games outside the declared window, validation status.
* **Chronology** — timezone awareness, missing timestamps, games sharing a
  timestamp, and the impossible case of one team appearing twice at the same
  instant, computed for **both** `date` and `game_datetime_utc` so the report
  shows which field can be trusted for ordering. Run before building sequential
  features, not after.
* **Identity** — every distinct (source id, abbreviation, full name) combination
  observed across history, so relocations stay auditable.

### Phase 3A0 run results

From a live run on 2026-08-20 covering all 20 seasons:

| | Count |
| --- | --- |
| All games ingested | 25,749 |
| **Regular season (modelling dataset)** | **24,038** |
| Playoffs | 1,671 |
| Play-in | 37 |
| NBA Cup finals | 3 |
| Other special / **unclassified** | 0 / **0** |
| Seasons passing their own validation | **20 / 20** |

Every game id is unique, every regular-season game has both scores, all 30
franchises appear in every season, and no id carries more than one label.

Upstream defects, all resolved in Phase 3A0.1 (see below):

* **4 games** with impossible tied finals (quadruple overtime) → corrected.
* **13 games** with no tipoff timestamp → recovered as exact UTC instants.
* **49 games** whose `date` predates the actual tipoff (COVID postponements) →
  flagged; `game_datetime_utc` is the authoritative chronology.

**24,038 of 24,038 regular-season rows are modelling eligible.**

### Phase 3A0.1 — correction layer

Seventeen records had known defects. All seventeen are now resolved against
**ESPN**, which is independent of BALLDONTLIE. Raw files under `data/raw/` are
never modified — corrections are declared in
`ingestion/source_corrections.py` and applied only when building the trusted
representation.

**Two systematic defects, not random corruption:**

**1. Quadruple-overtime games report an impossible tie (4 games).**
BALLDONTLIE's schema exposes `ot1`/`ot2`/`ot3` and **no `ot4` field**. A fourth
overtime happens only when the score is level after the third, so the stored
total is the score through OT3 — necessarily a tie — and the deciding period is
unrepresentable. Game 48851 shows the mechanism exactly: OT1 16-16, OT2 7-7,
OT3 8-8, total 155-155, actual final 168-161. All four 4OT games in the range are
affected; the **457 games with one to three overtimes are unaffected**.

| game | date | matchup | source | verified final |
| --- | --- | --- | --- | --- |
| 28012 | 2012-03-25 | UTA at ATL | 123-123 | **ATL 139 - UTA 133** |
| 32587 | 2015-12-18 | DET at CHI | 127-127 | **DET 147 - CHI 144** |
| 34714 | 2017-01-29 | NYK at ATL | 130-130 | **ATL 142 - NYK 139** |
| 48851 | 2019-03-01 | CHI at ATL | 155-155 | **CHI 168 - ATL 161** |

**2. Missing tipoff timestamps (13 games).** Two on 2009-01-22 and an entire
eleven-game slate on 2022-12-02. All recovered as exact UTC instants. For every
one, ESPN's final score was compared against BALLDONTLIE's and matched before the
timestamp was accepted, so a timestamp cannot be attached to the wrong game.

**Guards.** Each correction declares `expects` — the date, both teams, and the
value being replaced — checked before it applies. A mismatch raises
`CorrectionMismatchError` rather than writing a verified value onto the wrong
record. This is what makes the layer safe to keep as upstream data changes.

**Provenance is preserved in the data**, so "what BALLDONTLIE returned" and
"what we verified" are always distinguishable:

| column | meaning |
| --- | --- |
| `source_game_datetime_utc` / `source_home_score` / `source_away_score` | raw source values, verbatim |
| `game_datetime_utc` / `home_score` / `away_score` | trusted values after corrections |
| `datetime_corrected` / `score_corrected` | whether a correction applied |
| `chronology_precision` | `exact_datetime`, `date_only_verified`, or `missing` |
| `modeling_eligible` / `exclusion_reason` | explicit eligibility, never an implicit `dropna` |

### Chronology and rest-day policy

`ingestion/chronology.py` states the rules once, before any feature is written
against them:

* **`game_datetime_utc` is the only valid sort key.** `date` is the *scheduled*
  date and was never updated for postponed games.
* **Rest days derive from actual played tipoffs**, so a game postponed from
  January to May gives a long rest before the May game rather than appearing as a
  January back-to-back.
* **A game without an orderable timestamp cannot be sequenced** — it raises
  rather than being skipped, because a silent gap changes every rest value after
  it.
* Naive datetimes are refused; a silent zone assumption would shift
  back-to-backs.

Validated across all **600 team-seasons**: zero sequencing failures, zero
negative or zero-length rests, minimum gap 0.854 days (a legitimate
back-to-back), maximum 145.9 days (the 2020 COVID suspension).

## Phase 3A1 — forecasting baselines

```bash
python -m nba_prediction_market.pipelines.build_baselines
```

Outputs `nba_model_features_2006_26.parquet` (24,038 rows),
`nba_predictions_2025_26.parquet` (1,230 rows), and
`data/reports/model_baselines_2025_26.json`.

### Research split

2025-26 is the **holdout**, evaluated exactly once after every choice is frozen.
Development validation uses 2021-22 … 2024-25; for each validation season the
model trains only on strictly earlier seasons.

Two things are carefully distinguished:

* **Sequential state may use earlier games of the same season.** A February 2026
  prediction can use January 2026 results, because they existed at prediction
  time.
* **Model parameters may not.** Coefficients and hyperparameters are frozen
  before the holdout and never refit during it.

`stage_select` and `stage_holdout` are separate functions, and
`assert_no_holdout` raises if any development stage is handed season 2025 — so
tuning against the holdout requires removing a guard, not forgetting one.

### Feature engine

Every feature is captured **before** the current game's result touches any state,
and games are sequenced by trusted `game_datetime_utc` — never the scheduled
`date`. Current-season record and rolling form reset each season; a team with no
prior games this season gets explicit nulls rather than fabricated statistics,
and a season opener has **null rest**, never an offseason length. Missing values
are imputed inside the sklearn `Pipeline`, fitted per training split.

Model inputs are an **allowlist** (`features/feature_spec.py`), not a denylist —
a new leaky column upstream is excluded by default.

### Selection results

**Elo history barely matters.** With offseason regression of 0.5, information
decays by half each year, so histories of 3, 5, 8, 10, 15 and all-available score
within **1.2e-8** mean Brier of each other. Twelve configurations tie within
tolerance; all share K=20 and HCA=40. `all_available` is chosen among the ties
because it leaves `elo_diff` defined for every logistic training example.

**Logistic history does matter, and recent history wins.** A 5-season window
(0.21634) beats all-available (0.21709) and 15 seasons (0.21699). Recency-weighted
all-history with a 2-season half-life is essentially tied (0.21637).

| Elo | Logistic |
| --- | --- |
| K = 20, HCA = 40, regression = 0.50, history = all_available | 5-season window, C = 10 |

### 2025-26 holdout

| model | Brier | log loss | acc | AUC | ECE |
| --- | --- | --- | --- | --- | --- |
| constant (0.5794) | 0.24765 | 0.68847 | 0.5545 | 0.500 | 0.025 |
| Elo | 0.20829 | 0.60453 | 0.6837 | 0.731 | 0.038 |
| logistic | 0.20575 | 0.59834 | 0.6797 | 0.736 | 0.020 |
| Kalshi raw midpoint | 0.19465 | 0.57014 | 0.6911 | 0.765 | 0.030 |
| **Kalshi normalized** | **0.19465** | **0.57013** | **0.6911** | **0.765** | 0.034 |

**The market wins.** Paired bootstrap (10,000 resamples, fixed seed; negative
favours the model):

| comparison | ΔBrier | 95% CI | verdict |
| --- | --- | --- | --- |
| logistic − Kalshi | +0.01109 | [+0.0058, +0.0164] | market better |
| Elo − Kalshi | +0.01364 | [+0.0081, +0.0191] | market better |
| logistic − Elo | −0.00254 | [−0.0055, +0.0004] | inconclusive |

Kalshi is a **benchmark only** and never enters a model matrix.

## Phase 3A2 — team strength

```bash
python -m nba_prediction_market.pipelines.build_team_strength
```

Extends Phase 3A1 without touching it: separate feature table, prediction table
(`nba_predictions_3a2_2025_26.parquet`) and report
(`model_team_strength_2025_26.json`). The Phase 3A1 model is refit as an exact
control so both are judged on identical games.

### What was tested, and what actually helped

| experiment | outcome |
| --- | --- |
| Home-court grid extended to 0 | **HCA = 40 confirmed a true interior optimum** (0 → 0.2223, 20 → 0.2201, 40 → 0.2193, 60 → 0.2199). Phase 3A1's boundary result was a false alarm. |
| Margin-of-victory Elo | **Helped.** `sqrt` multiplier, dev Brier 0.21765 vs 0.21926 binary. |
| Opponent-adjusted margin + SOS | **Did not help** (bundle C worse than B). |
| Points scored/allowed, league-relative | **Did not help** (bundle D worst of all). |
| Home/away venue splits | **Did not help.** |
| Schedule fatigue | **Did not help.** |
| Blending | **Did not help** — weight 0 optimal against all three partners. |

Bundle ablation (fixed logistic, mean development Brier):

| bundle | features | Brier | AUC |
| --- | --- | --- | --- |
| A (3A1 control) | 11 | 0.21635 | 0.7019 |
| **B (+ MOV Elo)** | **12** | **0.21609** | **0.7027** |
| C (+ adjusted margin, SOS) | 16 | 0.21621 | 0.7021 |
| D (+ scoring) | 27 | 0.21634 | 0.7016 |
| E (+ venue splits) | 29 | 0.21621 | 0.7020 |
| F (+ fatigue) | 36 | 0.21623 | 0.7018 |

**Only one of five new feature families earned its place.** Bundle B — one extra
feature — is frozen.

### Frozen Phase 3A2 configuration

| | |
| --- | --- |
| Elo (feature) | K=20, HCA=40, regression=0.50, all_available |
| MOV Elo | K=20, HCA=40, regression=0.50, `sqrt`, all_available |
| Bundle | B (12 features) |
| Logistic | 5-season window, C=1.0 |
| Blend | none |

### 2025-26 (secondary benchmark)

| model | Brier | log loss | acc | AUC | ECE |
| --- | --- | --- | --- | --- | --- |
| Phase 3A1 logistic | 0.20575 | 0.59834 | 0.6797 | 0.7362 | 0.020 |
| **Phase 3A2 logistic** | **0.20451** | **0.59552** | **0.6927** | **0.7396** | 0.034 |
| MOV Elo alone | 0.20440 | 0.59564 | 0.6927 | 0.7400 | 0.040 |
| Kalshi normalized | 0.19465 | 0.57013 | 0.6911 | 0.7650 | 0.034 |

Paired bootstrap (10,000 resamples, fixed seed; negative favours the model):

| comparison | ΔBrier | 95% CI | verdict |
| --- | --- | --- | --- |
| 3A2 − 3A1 | −0.00124 | [−0.0022, −0.0003] | **3A2 better** |
| 3A2 − Kalshi | +0.00986 | [+0.0048, +0.0149] | market better |
| 3A1 − Kalshi | +0.01109 | [+0.0058, +0.0164] | market better |

The improvement over Phase 3A1 is real but small; the gap to the market narrowed
from 0.0111 to 0.0099 and remains clearly significant.

## Phase 3A3A0 — paid data audit

```bash
python -m nba_prediction_market.pipelines.build_paid_data
```

Audits what the GOAT tier actually provides, ingests the feeds that are safe,
and writes `data/reports/paid_data_audit.json`. **Nothing is merged into the
Phase 3A1/3A2 feature datasets** — this phase is foundation only.

### Endpoint capability matrix

Every row verified empirically against the live API on 2026-08-20, not taken
from documentation.

| endpoint | tier | verified seasons | granularity | safety |
| --- | --- | --- | --- | --- |
| `/v1/stats` | ALL-STAR | 2006–2025 | player × game | **A** safe when lagged |
| `/v1/stats/advanced` | GOAT | 2006–2025 | player × game | **A** safe when lagged |
| `/v2/stats/advanced` | GOAT | **2012**–2025 | player × game × period | **A** safe when lagged |
| `/v1/box_scores` | GOAT | 2006–2025 | game | **A** (no new pregame info) |
| `/v1/season_averages` | GOAT | 1996–2025 | player × season | **C** prohibited |
| `/v1/player_injuries` | ALL-STAR | current state only | player | **B** prohibited |
| `/v1/lineups` | GOAT | **2025 only** | player × game | **C** prohibited |

Safety classes: **A** safe as a historical input *when lagged to prior games*;
**B** usable only prospectively; **C** post-tip/post-game, unsafe for a
same-game T-30 prediction.

### Three prohibitions

**Lineups must never be a historical T-30 feature.** Documented as available only
once a game begins, and verified: zero rows for a 2006 game, zero for a 2024
game, twenty rows for a 2025-26 game. The `starter` flag is exactly the
information we would want and exactly what is not available before tip. That a
historical lineup can be retrieved *today* says nothing about what was known at
our prediction timestamp.

**Injuries are prospective-only.** Records carry `player`, `status`,
`return_date` and `description` — **no as-of timestamp, no history, no date
filter**. Verified: descriptions discuss the 2025-26 season in the past tense, so
the feed is today's state. A current injury record must never be attached to an
old game. It can be snapshotted forward from now on, but the past is
unrecoverable from this source.

**Season averages must not be attached retrospectively.** They are completed
full-season aggregates; using one inside its own season imports results that had
not happened yet. The player-game feed already provides the same information in a
lookahead-safe, game-by-game form.

### Documentation discrepancies found

* V2 advanced stats are documented as "2015 season onward"; the feed actually
  returns data from **2012**.
* Coverage is not uniform within V2: tracking fields (speed, distance, touches,
  passes, possessions) begin in **2012**, hustle fields (deflections, contested
  shots, box-outs) only in **2016**. Two distinct discontinuities.

### Other verified facts

* GOAT rate limit is **600 req/min** (`x-ratelimit-limit` header); `per_page`
  caps at **100** (500 returns HTTP 400).
* **Advanced stats are player-level only.** A game's twelve rows for a team carry
  twelve *different* pace and rating values, because each is that player's
  on-court estimate. Team-level pace or ratings must be aggregated deliberately
  (minutes-weighted), never read off one row.
* `min` is `null` on ~15% of player rows, and those rows have every other stat
  null too. They are absent observations, not zero-minute appearances, so
  minutes parse to `None` rather than `0`.
* The minutes format is **mixed within a single season** — both `MM:SS` and bare
  integers appear in 2006-07.

## Phase 3A3 — lagged advanced team and rotation features

```bash
python -m nba_prediction_market.pipelines.build_paid_features
```

Derives a team-game box score from the paid player feed, estimates possessions
and four factors, lags them into pregame features, and adds a rotation family
built purely from prior-game minutes. Phase 3A1/3A2 artefacts are read but never
written.

### Team-game derivation

Player rows aggregate to two observations per game. **Possessions are an
estimate, not an official NBA statistic** — Oliver's formula as used by
Basketball-Reference:

```
0.5 * (team + opponent), each side =
    FGA + 0.44*FTA - 1.07*(OREB/(OREB+OppDREB))*(FGA-FGM) + TOV
```

Validated across 48,060 team-games: median 96.5, p1 82.8, p99 113.5, only 2
outside [60,140], none non-positive. The four corrected 4OT games correctly
estimate ~132 possessions.

**Nothing is fabricated.** The 16 team-games whose player points do not
reconcile with the trusted score are flagged `box_score_complete = False`, their
source totals preserved, and every derived efficiency value set to null. They are
then excluded from rolling efficiency rather than imputed at source; the model
pipeline imputes on training data only.

### Rotation features — definitions

All computed from a team's **prior** games only:

* **minute share** — a player's minutes in a window over all team minutes in it.
* **HHI** — `sum(share²)`; near `1/N` for an even N-man rotation.
* **overlap(A,B)** — `sum over players of min(share_A(p), share_B(p))`. It is 1.0
  when two windows distribute minutes identically, 0.0 when they share nobody.
  Minutes-weighted by construction, so losing a starter costs far more than
  losing a fringe player.
* **rotation disruption** — compares a baseline window (games t-10…t-4) against a
  recent one (t-3…t-1), summing each player's minute shortfall.

**This is not an injury feature.** A player who stopped appearing three games ago
is visible; a player returning tonight is not, and must not be.

Player quality is a per-36 plus-minus shrunk toward zero by
`games/(games+20)`, so a 0.3-minute cameo cannot produce a large rating.

### Ablation result — one family worked, one clearly did not

| bundle | features | mean Brier | AUC |
| --- | --- | --- | --- |
| A (Phase 3A2 control) | 12 | 0.21609 | 0.7027 |
| B (+ efficiency / four factors) | 24 | **0.21635** ✗ | 0.7017 |
| C (+ roster continuity) | 18 | 0.21568 ✓ | 0.7035 |
| **D (+ rotation disruption)** | **15** | **0.21553** ✓ | **0.7044** |
| E (+ player quality) | 13 | **0.21659** ✗ | 0.7012 |
| F (C + D combined) | 21 | 0.21568 | 0.7032 |

**Possession-adjusted efficiency made the model worse** — it is still a
score-derived measure, and Phase 3A2 already showed that vein is exhausted.
**Rotation disruption helped**, and combining it with roster continuity added
nothing over disruption alone, so the simpler bundle D was frozen.

### 2025-26 (secondary benchmark)

| model | Brier | log loss | acc | AUC | ECE |
| --- | --- | --- | --- | --- | --- |
| Phase 3A2 logistic | 0.20451 | 0.59552 | 0.6927 | 0.7396 | 0.034 |
| **Phase 3A3 logistic** | **0.20369** | **0.59366** | 0.6894 | 0.7419 | **0.027** |
| MOV Elo | 0.20440 | 0.59564 | 0.6927 | 0.7400 | 0.040 |
| Kalshi normalized | **0.19465** | **0.57013** | 0.6911 | **0.7650** | 0.034 |

| comparison | ΔBrier | 95% CI | verdict |
| --- | --- | --- | --- |
| 3A3 − 3A2 | −0.00082 | [−0.0022, +0.0005] | **inconclusive** |
| 3A3 − MOV Elo | −0.00072 | [−0.0034, +0.0020] | **inconclusive** |
| 3A3 − Kalshi | +0.00903 | [+0.0040, +0.0141] | market better |

The paid data produced the **best-calibrated** model so far (ECE 0.027) and a
small AUC gain, but the improvement over Phase 3A2 is **not statistically
distinguishable from zero**.

## Phase 3A3B0 — availability sources

```bash
python -m nba_prediction_market.pipelines.build_availability_audit
```

Writes `data/reports/availability_source_audit.json`. **No model, and nothing
merged into Phase 3A3 features** — availability may not become a feature until a
source's as-of properties are proven.

### The governing rule

An observation is usable only if `observed_at <= prediction_ts`, where
`prediction_ts = scheduled tipoff - 30 minutes`. That makes **timestamp
precision**, not history depth, the deciding property of a source. Final
participation is never a substitute for pregame status.

### Source matrix

| source | historical coverage | intraday precision | status | proj. lineup | conf. lineup | historical as-of | cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NBA official injury report | **rolling ~8 months** | **exact timestamp** | ✓ | ✗ | ✗ | **unsafe** | free |
| Sportradar Daily Injuries | URL accepts 2013-2026; unverified | **date only** | ✓ | ✗ | ✗ | unknown | trial key needed |
| SportsDataIO | advertised, depth unknown | unknown | ✓ | ✓ | ✓ | unknown | paid |
| BALLDONTLIE injuries | none (current state) | **none** | ✓ | ✗ | ✗ | unsafe | owned |
| BALLDONTLIE lineups | 2025-26 only, post-tip | none | ✗ | ✗ | ✓ | unsafe | owned |

**No audited source can reconstruct a past T-30 state.** Historical availability
for 2006-2025 appears unrecoverable.

### NBA official report — verified behaviour

`https://ak-static.cms.nba.com/referee/injury/Injury-Report_{YYYY-MM-DD}_{hh}_{mm}{AM|PM}.pdf`

* Published **every 30 minutes, around the clock**.
* `Last-Modified` matches the filename slot exactly (04:00PM → 21:00:05 GMT in
  January, 20:00:05 GMT in March — confirming Eastern time with DST).
* The timestamp also appears in the PDF header.
* Missing files return **403**, verified by requesting an invalid minute on a
  valid date.
* Retention boundary observed between **2025-12-15 (403)** and **2025-12-28
  (200)**; every date tested in 2016-2025 returned 403.
* PDFs extract as text with `pypdf` — **no OCR required**.

### Infrastructure built

* **Append-only snapshots** (`data/raw/availability/<source>/<date>/`) — a newer
  state is a new file beside the old one, never an overwrite. Identical
  re-captures are skipped; replay with `until=` shows only what existed then.
* **Normalized events** — `available / probable / questionable / doubtful / out /
  unknown`, with the raw status always preserved. **Absence from a report is
  `unknown`, never `available`.**
* **As-of engine** — `observed_at <= anchor`, with an exactly-at-anchor
  observation accepted and one a second later rejected. A **date-only**
  observation is refused for a T-30 anchor rather than silently accepted.
* **Identity** — resolves by NBA reference id, then registered alias, then exact
  name *within a team*. Name-only resolution is refused because names collide.
  Name normalization reconciles the report's `"Porter Jr., Michael"` with
  BALLDONTLIE's `"Michael Porter Jr."`.
* **Schedule-aware capture planner** — works backwards from each tipoff and
  guarantees one capture immediately before the anchor; refuses to emit any
  capture at or after it.

### Do not read a training window into this

The point of 20 seasons is to make the history *available*, not to assert it is
all useful. Whether 3, 5, 8, 10, 15, or all seasons help is an open question to
be settled by chronological validation **before** the 2025-26 holdout — never by
2025-26 performance.

## Phase 3A3B1 — salvaging the 2025-26 injury reports

```bash
python -m nba_prediction_market.pipelines.build_availability_salvage   # recover the archive
python -m nba_prediction_market.pipelines.run_availability_capture --dry-run   # plan captures
```

Phase 3A3B0 established that **no audited source can reconstruct a past T-30
availability state**, and that the NBA's own injury reports survive on the CDN
for roughly eight months before disappearing. That made this phase
time-sensitive: whatever was not copied would be gone. This phase copies it, and
builds the machinery so nothing is lost again.

**Still no model.** Availability is not merged into the Phase 3A3 feature set,
and will not be until its as-of properties are proven end to end.

### The archive

7,899 reports, **2025-12-22 to 2026-06-13**, ~660 MB, stored append-only at
`data/raw/availability/nba_official/YYYY/MM/DD/` beside a JSON sidecar recording
the URL, HTTP status, headers and content hash of every fetch. An identical
re-fetch is skipped; a *different* payload for a slot already held is written
alongside as `.conflict-<hash>.pdf` and flagged, never overwritten.

| | |
| --- | --- |
| slots checked | 9,264 |
| reports archived | 7,899 |
| slots returning 403 | 1,365 |
| hash conflicts | 0 |
| errors | 0 |
| span coverage | 94.6% of all 30-minute slots across 174 days |
| complete days | 162 / 174 |

The gaps are explained, not merely counted: **2026-02-13 to 02-17** is the
All-Star break, and the four remaining empty days are Finals off-days.

### 403 means two different things, so the archiver carries a canary

The CDN answers *both* "this report was never published" and "you are being rate
limited" with **403**. That ambiguity is dangerous in one specific direction: a
throttled run does not fail loudly, it quietly records real reports as missing.

This was not theoretical. An early archiver run used 8 concurrent workers, was
rate-limited within seconds, and returned 403 for **every** URL including ones
already known to exist — 0 reports archived. The block cleared after ~7 minutes.

The fix is in the code, not in the operator's memory:

* requests are **sequential**, with a 0.35 s floor between them;
* whenever a run sees a 403, it re-checks a **canary** URL known to exist. If
  the canary also 403s, the run was blocked and the report says so rather than
  letting the run masquerade as a quiet news day.

The canary held at 200 for the entire salvage run, so the 1,365 403s are genuine
non-publication. `archive_inventory` additionally flags near-empty days sitting
between two complete days as `suspect_blocked` — a deliberately conservative
heuristic that fires on two Finals off-days here, and which the canary evidence
overrides.

### The parser reads coordinates, not columns

The reports have a real text layer, so **no OCR is used**. But the obvious
approach — layout-mode text extraction, slice by character offset — is wrong
here, and wrong silently:

* **the header row prints on page 1 only**; pages 2..N continue the table with
  no header of their own;
* **layout mode rescales the character grid per page**, so page 2's offsets bear
  no relation to page 1's.

An offset-based implementation therefore parsed page 1 and dropped every later
page: **15 entries out of 117 on a real 8-page report, with no warning raised**.
Absolute glyph coordinates are identical across pages, so the parser now takes
column anchors from page 1's header and applies them everywhere by x-position.

Three further quirks, all observed and all handled:

* the page content matrix is a **vertical flip**, so increasing text-space y is
  the visual reading order;
* **group columns print once** — game date, time, matchup and team appear on the
  first row of a block and are carried forward, including across page breaks,
  since a team's block can span pages;
* **reasons wrap** onto a continuation row that belongs to the player above.

A team that has not filed prints `NOT YET SUBMITTED` with no player. That is a
statement about the team, not a wrapped reason for whoever came before, so it is
recorded separately as an outstanding filing — tied to the specific game it was
outstanding for, because one report covers several dates.

Result across the full archive: **7,899 / 7,899 reports parsed, 0 failures**, and
**503,494 player-status rows**. A stratified sample of 140 reports spanning
December through the Finals produced 0 warnings and a single layout variant.

Two measurements are worth recording together. Under **layout-mode character
offsets**, header positions shift by a few characters between reports, because
the PDF uses a proportional font and the extractor snaps glyphs to a character
grid — fingerprinting on those offsets reported 26 "variants" of what is one
layout. Under **glyph coordinates**, the same anchors are
`23.1 / 119.6 / 200.0 / 264.2 / 425.0 / 585.7 / 666.1` in every one of the 7,899
reports, minimum equal to maximum. The apparent drift was an artifact of the
extraction mode, not a property of the documents; the layout is now named by its
column structure and the coordinate range is reported alongside it.

### Coverage, stated as a boundary rather than a percentage

| coverage class | games |
| --- | --- |
| valid pre-anchor state | **808** |
| no surviving report | 422 |
| **total 2025-26 regular season** | **1,230** |

65.7% is the headline, but the useful statement is sharper: the 422 uncovered
games run **2025-10-21 to 2025-12-21 without exception**, and coverage from
**2025-12-22 onward is complete**. The cut is the CDN's retention boundary, not
a sampling artifact — so the usable window is a clean date range, not a
scattering of holes.

Report age at the anchor is **0 minutes for all 808 covered games**. That is not
a bug: every 2025-26 tipoff falls on :00 or :30, so `tipoff − 30 min` lands
exactly on the report publication grid.

### The leakage guarantee

Every selected report satisfies `report_timestamp <= prediction_ts`. A game
whose earliest surviving report postdates its anchor is marked unavailable and
**never backfilled** from a later report. `no_surviving_report` is likewise never
filled in from an adjacent game or a neighbouring day.

`both_teams_submitted` is nullable on purpose: for the 422 games with no report,
it is null rather than `False`, because "no report survived" is not the same
claim as "a team failed to file".

### Player identity, and what is deliberately left unresolved

Resolution is by NBA reference id, then registered alias, then exact name
*within a team*. Name-only matching is refused because names collide. In
practice the reports carry no player identifiers, so **every** resolution here
came from the team-and-name tier — which is precisely why the name repairs below
matter so much.

Two artifacts of glyph extraction are repaired first, both of them repairs to
*our own join* rather than fuzzy matches: `"Collins,Zach"` → `"Collins, Zach"`,
and hyphenated surnames drawn as separate chunks — `"Gilgeous- Alexander"` →
`"Gilgeous-Alexander"`. The latter alone accounted for 5,857 unresolved rows.

After that repair **495,679 / 503,494 rows (98.45%)** resolve, all of them by
exact name within a team. The remaining 7,815 rows are just **17 (player, team)
pairs**, listed by name in the report rather than matched. They fall into three
kinds, and none of them is a parser defect:

* **Nickname against legal name (6).** The player is on that team's roster under
  a different given name — `Sarr, Alex` / *Alexandre Sarr*, `Claxton, Nic` /
  *Nicolas Claxton*, `Bailey, Ace` / *Airious Bailey*, `Hyland, Bones` /
  *Nah'Shon Hyland*, `Carrington, Bub` / *Carlton Carrington*, `Williams, Nate` /
  *Jeenathan Williams*.
* **Listed under a team they had not yet played for (6).** `Gordon, Eric` is on
  the report for MEM while the registry has him at PHI; likewise Conley, Ball,
  Boucher, Terry and Landale. Mid-season moves put a player on an injury report
  before he appears in a box score for the new team.
* **No 2025-26 box-score appearance at all (3).** `Jones Garcia, David`,
  `Djurisic, Nikola` and `Hayes-Davis, Nigel` never played an NBA game that
  season. The registry is derived from player-game stats, so a player with zero
  appearances has no entry to match — absent by construction, not by error.

All three are left unresolved on purpose. Resolving the first needs an
explicitly verified alias per player on the Phase 3A0.1 pattern; the second needs
a date-aware roster rather than a season-level one; the third cannot be resolved
from box scores at all. A silent name match is exactly the guess this project
refuses to make.

### Game matching, and four discrepancies worth naming

808 matched, **0 ambiguous**, 93 unmatched. The unmatched count is broken down
rather than reported bare, because most of it is expected:

* **89** fall after the regular season — play-in and playoff games, outside the
  1,230-game frame by design;
* **4** fall inside the regular-season window and are genuine source
  discrepancies between the league's reports and the schedule:
  `2026-01-08 MIA@CHI`, `2026-01-24 GSW@MIN`, `2026-01-25 DAL@MIL`,
  `2026-01-25 DEN@MEM`.

The GSW@MIN case was checked in detail. Six of the seven games in that report's
date block match the schedule on date *and* Eastern tipoff time exactly; only
GSW@MIN differs, and the schedule lists that matchup on two consecutive days
(2026-01-25 and 2026-01-26), which is itself anomalous. The parser reproduces
the PDF faithfully. **These four are recorded as discrepancies and not
reconciled** — resolving them needs independent verification, which is the
Phase 3A0.1 correction-layer process, not a silent edit here.

### Prospective capture

`run_availability_capture` plans captures backwards from each tipoff and refuses
to emit any capture at or after an anchor. Verified against the full 2025-26
slate: **1,230 / 1,230 games get a pre-anchor capture, and 0 of 11,070 planned
tasks land at or after their anchor.** The runner is restart-safe — an
already-archived slot is skipped rather than refetched — and every capture is
written to the append-only snapshot store as well as the PDF archive.

Two secondary feeds are wired for **forward-only** capture. Neither carries an
as-of timestamp or a history endpoint, so the observation time is the moment we
fetched it and nothing else; the adapters therefore never write a
`source_report_timestamp`, which is what stops a consumer back-dating today's
state onto an earlier game.

* **BALLDONTLIE `/v1/player_injuries`** — snapshotted forward. Its *historical*
  use remains prohibited.
* **SportsDataIO** — a deliberately inert placeholder. There is no subscription
  and no response shape verified against live data, and enabling a feed on the
  strength of vendor documentation alone would put unverified data into the
  availability store.

### Kalshi multi-anchor benchmark

Seven research anchors are reconstructible from one cached candle stream —
`T-24h, T-6h, T-3h, T-1h, T-30m, T-15m, T-5m` — each resolved by the Phase 2
selector, so the same lookahead guarantee applies at every anchor. Kalshi remains
a **benchmark only, never a model feature**.

One honest limitation: the Phase 2 cache was fetched under the slug
`t30_lb60_p1` — a 60-minute window *ending* at T-30. It can serve T-1h and T-30m
and nothing else; T-24h/6h/3h fall before the window starts and T-15m/5m after it
ends. Backfilling the full anchor set needs a refetch at the wider window, one
request per market. Because the cache slug is part of the cache path, that
refetch lands in its own directory and cannot overwrite the Phase 2 quotes.
**Implemented and tested; not backfilled.**

### Outputs

| file | contents |
| --- | --- |
| `data/raw/availability/nba_official/` | 7,899 PDFs + per-fetch JSON sidecars |
| `nba_official_availability_events_2025_26.parquet` | 503,494 player-status rows |
| `nba_game_availability_t30_partial_2025_26.parquet` | 1,230 games, T-30 state where recoverable |
| `data/reports/nba_official_archive_salvage.json` | coverage, identity, matching, drift |
| `data/reports/availability_capture_run.json` | prospective capture run log |

## Phase 3A3B2 — historical availability recovery and cross-source validation

```bash
python -m nba_prediction_market.pipelines.build_availability_backfill --start 2024-10-01 --end 2025-06-30
python -m nba_prediction_market.pipelines.build_availability_salvage --season 2024
python -m nba_prediction_market.pipelines.build_availability_coverage
```

This phase set out to find whether third-party archives could extend availability
history backwards. They can, a little. It turned out not to matter, because the
official source itself was never actually gone.

### The Phase 3A3B0 retention finding was wrong

Phase 3A3B0 probed the CDN with the 2025-26 filename pattern, got 403 on older
dates, and concluded the league retained roughly eight months of reports. Phase
3A3B1 then read the 2025-12-22 boundary in our own archive as that retention
edge, and reported 808 of 1,230 games as the best obtainable.

**Both readings were the same mistake.** The league publishes under two filename
conventions and 403 is what the wrong one returns:

```
legacy   Injury-Report_2025-01-15_05PM.pdf        hourly, stamped at :30
modern   Injury-Report_2026-01-15_05_30PM.pdf     every 30 minutes
```

The switch is sharp and was measured, not assumed: on **2025-12-22** the legacy
name still serves the 08:30 ET report and the modern name first serves the 09:00
ET one. Nothing was being deleted. Reports remain fetchable back to
**2018-12-17**, found by bisection.

Two further properties of the legacy convention matter:

* **the filename hour is not the report time.** `05PM` is the **5:30 PM**
  report. Reading it as 5:00 shifts the observation half an hour early — the
  direction that lets a report published *after* an anchor be accepted for it.
  Our parser takes the timestamp from inside the PDF, so the filename is never
  load-bearing.
* **cadence changed twice.** Three slots a day (01PM/05PM/08PM) through 2020-21,
  hourly from 2021-22, half-hourly from the cutover.

| era | convention | slots/day |
| --- | --- | --- |
| 2018-19 … 2020-21 | legacy | 3 |
| 2021-22 … 2025-26 (to 12-22) | legacy | 24 |
| 2025-26 (from 12-22) | modern | 48 |

### 2025-26 is now complete

The 422-game gap Phase 3A3B1 attributed to retention was simply the pre-cutover
period, and it recovered cleanly.

| | Phase 3A3B1 | now |
| --- | --- | --- |
| reports archived | 7,899 | **9,420** |
| player-status rows | 503,494 | **619,182** |
| games with valid T-30 state | 808 / 1,230 (65.7%) | **1,230 / 1,230 (100%)** |
| report age at anchor, p95 | 0 min | 30 min |
| parse failures | 0 | **0** |
| unexplained report-vs-schedule mismatches | 4 | **0** |

Every unmatched report key is now classified rather than merely counted: 89
play-in and playoff games outside the 1,230-game frame, 4 postponed occurrences,
1 NBA Cup final, and **zero unexplained**.

### The parser reads every era the league has published

Older reports are not merely older; they are laid out differently, and both
differences fail silently rather than loudly.

* **Page geometry.** Reports through 2023 are a *portrait* media box with
  `/Rotate 90`, drawn sideways under a `(0,1,-1,0)` content matrix. Read without
  the rotation the table arrives transposed — each apparent row is an entire
  column, so a whole report's statuses concatenate into one string. From 2024 the
  page is landscape with a vertical flip. The parser now composes the content
  matrix with the page rotation and works in displayed coordinates, so one code
  path reads both.
* **Column layout.** 2018-19 uses a **nine**-column table — `Category` and
  `Previous Status` columns that were later dropped, with Reason before Current
  Status — against the seven-column layout used since. Layouts are matched by
  their header labels, nine-column first, because its labels are a superset and
  testing the shorter sequence first would read `Category` as `Current Status`.
  The `Category` and `Reason` halves are rejoined with `" - "`, reproducing the
  modern convention exactly, and both raw halves are kept.

Team names also wrap onto their own row in the narrower portrait layouts
("Minnesota" then "Timberwolves"); a row carrying nothing but a team fragment now
continues the name above it. Across nine era samples spanning 2019-2025, every
team name resolves to a known franchise.

### Cross-validation found a real defect in our own parser

Professor-Pete's repository retains no PDFs and covers 2024-25, which does not
overlap our 2025-26 salvage, so the two outputs could not be compared directly.
Recovering 2024-25 ourselves removed that obstacle: both parsers can be pointed
at the identical file.

On 20 shared reports:

| dimension | agreement |
| --- | --- |
| games | 100% |
| teams | 100% |
| player rows present | 1,908 / 1,908, none either-side-only |
| **status** | **1,908 / 1,908 (100.000%)** |
| reason, before the fix | 690 / 1,908 (36.2%) |
| **reason, after the fix** | **1,896 / 1,908 (99.37%)** |

The reason defect was ours, and their code predicted it exactly: *"Reason text is
vertically centred inside its cell and straddles adjacent player rows, so
clustering words by y-position mis-assigns it."* Our parser grouped glyphs into
rows by y and then read a Reason column — but a wrapped reason's first line is
drawn *above* the player it describes, so it was handed to the previous player.
The visible symptom was two players' reasons welded together:

```
ours (before) : 'Injury/Illness - Back; Lumbar; Strain Injury/Illness - Right Pelvic;'
theirs        : 'Injury/Illness - Back; Lumbar; Strain'
```

Reason cells are now rebuilt from individual glyph positions, each assigned to
the nearest player baseline on its own page. Only the Reason column is treated
this way; the grouping columns print on their block's first row rather than
centred, which is why they carried forward correctly all along — confirmed by the
100% team agreement.

This was a defect fix, not an exercise in agreeing with a third party: welded
reason strings are self-evidently wrong. The residual 0.63% is not chased, and
part of it is our output being arguably the better reading.

### Professor-Pete: schema, and what it does and does not preserve

`data/season_rows.csv`, 107,337 rows:

```
snapshot_date, snapshot_time, game_date, game_time, matchup, team, player, status, reason
```

**It does preserve snapshot identity** — `snapshot_date` plus the filename slot.
An exact report timestamp is recoverable, so the data is genuinely T-30
reconstructable, and no timestamp has to be inferred from row order.

With that timestamp applied, all **1,230 of 1,230** 2024-25 regular-season games
have at least one report at or before T-30. But the archive covers only **5 of
the 24 slots** the league published that season (01PM/03PM/05PM/08PM/11PM), so it
is T-30 *safe* while being materially stale:

| report age at the T-30 anchor | median | p75 | p95 | max |
| --- | --- | --- | --- | --- |
| Professor-Pete, 5 slots/day | 90 min | 120 min | 150 min | 780 min |
| official archive, recovered directly | ~0-30 min | | 30 min | 60 min |

Two concerns in their source code, both inherited by anyone reusing it:

* **`snap_key` maps a slot to the wrong instant.** It computes `05PM` → 17:00,
  but the report is stamped 17:30. The CSV itself stores the raw slot so the
  error is recoverable, but the helper encodes a half-hour lookahead.
* **The report's own timestamp is discarded.** Their parser never reads the
  "Injury Report: MM/DD/YY HH:MM" line, so the filename is the only surviving
  provenance. Ours reads and keeps both.

Their fetcher runs **four concurrent workers** against a CDN that answers 403
under throttling — the same hazard we hit. They do document the ambiguity and
retry, but with only ~3.6 s of total backoff and **no canary**; our own block
lasted about seven minutes. Their 233 retained days each hold all five slots, so
this run appears not to have lost anything, but the design cannot detect it if it
had. On the multi-page risk they were ahead of us: their `_y_grid` cutoff
comment shows they hit and fixed the continuation-page failure directly.

### StatSurge: date-only, and the 2 PM claim is now exact

Schema, 35,522 rows over 3,901 games, 2021-10-19 to 2024-06-17:

```
PLAYER, STATUS, REASON, TEAM, GAME, DATE
```

**There is no time field of any kind**, and exactly one row per player and game.
The publisher describes an approximately 2 PM report; that cannot be checked from
the file, so the data is classed by what it carries — `DATE_ONLY` — not by the
claim.

The claim *can* be checked against the source, and it holds. Comparing StatSurge
against every hourly official report on six dates spanning all three seasons:

| official slot | report time | StatSurge rows present | status agreement |
| --- | --- | --- | --- |
| 11AM | 11:30 ET | 85.6% | 96.3% |
| 01PM | 13:30 ET | 96.7% | 99.6% |
| **02PM** | **14:30 ET** | **100.0%** | **100.0%** |
| 03PM | 15:30 ET | 100.0% | 99.8% |
| 05PM | 17:30 ET | 100.0% | 98.4% |
| 08PM | 20:30 ET | 100.0% | 82.3% |

StatSurge is exactly the **14:30 ET** report, at 576/576 on both measures.

### Why a 2 PM snapshot may not stand in for T-30

Because it is wrong roughly a quarter of the time. Measured across the complete
2025-26 season, comparing each home-team designation's status in the 14:30 ET
report against its true state at T-30:

| | designations | share |
| --- | --- | --- |
| unchanged 14:30 → T-30 | 4,935 | 77.1% |
| **changed** | **1,465** | **22.9%** |

The falling agreement down the StatSurge table above says the same thing
independently: by 20:30 ET only 82.3% of the 14:30 statuses still hold.

StatSurge is therefore classified **`EARLY_DAY_ONLY` / `DATE_ONLY`**, never
`T30_SAFE`, and a 2 PM status is never forward-filled to tip. Forward-filling
would manufacture exactly the certainty the anchor rule exists to deny.

### The seventeen identity pairs

All investigated; **16 resolved, 1 deliberately not**, correcting 6,913 rows and
lifting 2025-26 resolution to **99.64%** — the only rows left unresolved are the
one player below. Recovering 2024-25 surfaced 13 more pairs of the same three
kinds, all resolved, taking that season to **100.00%**. Every resolution is an
individually
verified fact with its evidence recorded in
`availability/player_aliases.py`. **There is no similarity-based fallback**: an
unlisted mismatch stays unresolved and is reported.

| category | pairs | evidence used |
| --- | --- | --- |
| preferred name vs legal name | 6 | legal-name player is the *only* holder of that surname on that exact team |
| naming convention / legal change | 3 | canonical BALLDONTLIE record including its team |
| reported before first box score | 7 | full name unique league-wide, so the id is unambiguous though the team differs |
| no canonical record | 1 | — |

Resolved: Sarr→Alexandre Sarr, Claxton→Nicolas, Bailey→Airious, Hyland→Nah'Shon,
Williams→Jeenathan, Carrington→Carlton; Jones Garcia→David Jones (BDL id
1028245237, *on San Antonio*), Hayes-Davis→Nigel Hayes (id 2221, *on Phoenix*, both
his PHX and MIL rows); Gordon, Landale, Conley (CHI and CHA), Ball, Terry,
Boucher. All twelve ids were verified to share our registry's id space exactly.

**Left unresolved: `Djurisic, Nikola` (ATL, 2,246 rows).** No BALLDONTLIE record
under any spelling searched and no 2025-26 box-score appearance; his reports read
"G League - On Assignment". Listed rather than guessed.

The pre-first-appearance pattern is recurring rather than exceptional. The
durable fix is a date-aware roster instead of a season-level one; until then each
instance is listed with its evidence.

### The four report-vs-schedule discrepancies

All four are the same thing, and **neither source was wrong**. Verified
independently against the ESPN scoreboard API, which reports the original date
`STATUS_POSTPONED` and the replay date `STATUS_FINAL`:

| original date | matchup | replayed | our schedule holds |
| --- | --- | --- | --- |
| 2026-01-08 | MIA@CHI | 2026-01-29 | 2026-01-29 ✓ |
| 2026-01-24 | GSW@MIN | 2026-01-25 | 2026-01-25 ✓ |
| 2026-01-25 | DEN@MEM | 2026-03-18 | 2026-03-18 ✓ |
| 2026-01-25 | DAL@MIL | 2026-03-31 | 2026-03-31 ✓ |

The injury report describes the date a game was *originally scheduled* for; the
trusted schedule holds the date it was *played*. GSW@MIN corroborates itself
inside the reports: the 01/24 listing stops appearing between 19:30Z and 20:00Z on
2026-01-24 and is replaced by an 01/25 listing at the same 5:30 PM tip.

Rows filed against a postponed occurrence describe a game that did not happen —
two of these were replayed weeks later — so they are **not** attached to the
replayed game. They stay unmatched, now classified rather than unexplained.

A fifth mismatch surfaced from the newly recovered period and resolved the same
way: **2025-12-16 SAS@NYK** is the **NBA Cup Championship**, played at T-Mobile
Arena in Las Vegas. It is the one Cup game that does not count toward
regular-season records, so it is correctly absent from the 1,230-game frame. The
season metadata the project already carries records that date, so it is
classified from existing knowledge rather than hard-coded.

### Two more defects the recovery surfaced

Extending the parser across five seasons exposed two faults that a single
season had hidden. Both were silent.

**Phantom players from split page markers.** In the portrait-rotated era the
trailing page count is drawn on its own baseline, so the marker row reads
`Page 1 of` and a bare count floats separately. Matching only the complete
`Page 1 of 8` let the fragment fall into the Player column and become a player:
**18,220 rows in 2021-22 and 16,010 in 2022-23**, which is also most of what
made those seasons' identity resolution look poor. Reports from 2023-24 onward
draw the marker in one piece and were never affected.

**Postponed games keyed by the wrong date.** The schedule keeps a postponed
game's *original* `date` while moving `game_datetime_utc` to the replay. The
anchor comes from the tipoff, so keying the report join on `date` attached
reports filed for the original occurrence to an anchor months later — one
2021-22 game acquired a "T-30 state" from a report **102 days stale**. The join
now keys on the Eastern date of the tipoff, which is both consistent with the
anchor and immune to the disagreement. Fifty-eight games across 2020-21 to
2022-23 carry it; the existing chronology code already ordered by
`game_datetime_utc`, so rest-day features were never affected.

Fixing the key also *recovered* games: 2021-22 went from 1,221 to 1,230 covered
and 2022-23 from 1,228 to 1,230, because reports filed on the replay date now
match the game they describe.

### Historical availability coverage matrix

| season | source | games | T-30 safe | class | report age p95 |
| --- | --- | --- | --- | --- | --- |
| 2021-22 | official | 1,230 | 1,230 | `T30_SAFE` | 30 min |
| 2022-23 | official | 1,230 | 1,230 | `T30_SAFE` | 30 min |
| 2023-24 | official | 1,230 | 1,229 | `T30_SAFE` | 30 min |
| 2024-25 | official | 1,230 | 1,230 | `T30_SAFE` | 30 min |
| 2025-26 | official | 1,230 | 1,230 | `T30_SAFE` | 30 min |
| **total** | | **6,150** | **6,149 (99.98%)** | | |

Across all five seasons: **31,647 reports parsed, zero parse failures,
2,007,322 player-status rows**, one layout variant reported, and **zero
unmatched report keys left unexplained**. Identity resolution runs 96.70% to
100.00% by season.
| 2024-25 | Professor-Pete | — | reconstructable | `T30_SAFE`, stale | 150 min |
| 2021-24 | StatSurge | — | not reconstructable | `DATE_ONLY` | n/a |

The single uncovered game is **2023-10-25 BOS@NYK**, and it is a gap in the
*source*: ESPN lists it `STATUS_FINAL` as one of twelve games that day and the
schedule agrees, but the league's reports carry eleven of the twelve and never
mention it — not in any of the 55 reports archived across 2023-10-23..26, in
entries or not-yet-submitted markers. It is reported as uncovered and never
filled from a neighbouring game.

Every unmatched report key across all five seasons is now classified:
pre-season exhibitions, play-in and playoff games, postponed occurrences, the
NBA Cup final, and All-Star weekend events, which the reports file under a
`Non-NBA Team` placeholder.

### Is there enough history to train an availability model?

Yes, and the answer is the strongest of the four the brief anticipated —
**outcome B: every development season contains exact T-30 states**, obtained
from the authoritative source rather than a third party.

That means the Phase 3A1 validation architecture does **not** need weakening:

* **Development** — 2021-22, 2022-23, 2023-24, 2024-25, with chronological
  expanding-window validation exactly as in Phase 3A1. Availability features
  are built under the same `observed_at <= prediction_ts` rule already
  enforced, and verified: **0 of 6,149 selected reports postdate their anchor.**
* **Holdout** — 2025-26, untouched.

One caveat worth carrying forward rather than discovering later. The
publication cadence differs between training and holdout: hourly through
2024-25, half-hourly from 2025-12-22. That is a mild distribution shift in
*feature freshness*, not in the feature itself, and it runs in the harmless
direction for deployment, since the holdout matches what production will see.
It is small in practice — the p95 report age is 30 minutes in every season, and
99%+ of games in both eras have a report within 30 minutes of the anchor — but
report age is now carried on every row, so it can be controlled for directly.

**No model was built in this phase**, and availability is still not merged into
the Phase 3A3 feature set.

### When availability news actually arrives

Measured across the complete 2025-26 season, for every player-game designation:
how long before tip was its final pre-anchor status set?

| the status held from | share of designations |
| --- | --- |
| T-30m or later | 7.8% |
| T-1h or later | 16.0% |
| T-3h or later | 23.1% |
| T-6h or later | 35.0% |
| T-24h or later | 50.2% |

Half of all designations are settled more than a day out; the other half move
inside 24 hours, and a third inside six. That is what makes an early-day
snapshot a weak proxy and a T-30 state a strong one.

### Kalshi multi-anchor: the smallest useful backfill

Kalshi's `KXNBAGAME` history covers **2025-26 only** — 1,230 games, all with a
usable quote on both sides at T-30. Availability now covers the same 1,230 games
completely, so overlap at any anchor is the full season.

Two anchors in the original list should be dropped outright. **T-15m and T-5m
fall *after* the prediction anchor.** Our whole framework forecasts at T-30, so a
quote from T-15m could not be an input without breaking the rule the project is
built on. They are only useful for studying market behaviour after the anchor,
which is not the modelling target.

Of the rest, the informative window is where availability news actually lands —
inside about six hours. So:

**Recommendation: one refetch per market at a six-hour window ending at T-30.**

That single request per market yields **T-6h, T-3h, T-1h and T-30m** from one
candle stream, covering 35% of the news flow. T-24h adds a stable baseline for
the half of designations settled more than a day out, and can be included by
widening the same window to 24 hours if the extra volume is acceptable.

Cost: 1,230 games x 2 markets = 2,460 requests. Because the cache slug encodes
the window, this lands in its own directory and cannot overwrite the Phase 2
quotes.

### Licensing and provenance

Nothing third-party is committed. Raw downloads live under the gitignored `data/`
tree; only source code and provenance metadata are committed.

| source | terms | redistributable |
| --- | --- | --- |
| Professor-Pete | **no licence declared** — all rights reserved by default | **no** |
| StatSurge | no terms stated on the distribution page | **no** |
| official NBA reports | league's own published artefacts, retrieved unmodified | archived locally, not redistributed |

Neither third-party dataset is a data dependency. Both are cross-checks against a
source we can now obtain directly, at finer cadence and with full provenance.

## Phase 3A3C — T-30 player availability

```bash
python -m nba_prediction_market.pipelines.build_availability_features --cadence native
python -m nba_prediction_market.pipelines.build_availability_features --cadence harmonized
python -m nba_prediction_market.pipelines.build_availability_model
```

The question is not "does availability matter" — of course it does — but whether
*genuine point-in-time* availability adds anything to a basketball-strength model
that already carries rotation disruption. Every development fold therefore trains
two models on **identical examples**: a control with only the frozen Phase 3A3
features, and an enhanced model adding exactly one availability family.

**It does help, consistently, and it helps where it should.**

### Availability coverage, all seven recovered seasons

| season | games | T-30 | T-1h | T-3h | report age p95 | cadence |
| --- | --- | --- | --- | --- | --- | --- |
| 2019-20 | 1,059 | 944 (89.1%) | 944 | 939 | 150 min | 3/day |
| 2020-21 | 1,080 | 1,079 (99.9%) | 1,078 | 1,068 | 150 min | 3/day |
| 2021-22 | 1,230 | 1,230 | 1,230 | 1,230 | 30 min | hourly |
| 2022-23 | 1,230 | 1,230 | 1,230 | 1,230 | 30 min | hourly |
| 2023-24 | 1,230 | 1,229 | 1,229 | 1,228 | 30 min | hourly |
| 2024-25 | 1,230 | 1,230 | 1,230 | 1,230 | 30 min | hourly |
| 2025-26 | 1,230 | 1,230 | 1,230 | 1,230 | 30 min | half-hourly |

117 games have no T-30 state and every one is preserved as
`availability_coverage = False` with null availability fields, never imputed from
a neighbour. 115 fall in 2019-20, where the league published only three reports a
day through a season the pandemic cut in half. One is
**2023-10-25 BOS@NYK**, the game the source genuinely omitted, and it carries
`availability_source_omission = True` so it stays distinguishable from an archive
gap. The 2025-26 holdout has **zero** uncovered games.

The 3-slot seasons are usable but visibly weaker: p95 report age of 150 minutes
against 30 for every later season. They earn their place as training history for
the first fold, not as evidence about how good availability features can be.

### What a designation is actually worth

Learned per fold from training seasons only. Fold 2024-25 (training 2019-20 …
2023-24), 32,598 designations:

| status | n | P(plays) | ± | mean actual min | baseline min | minutes ratio |
| --- | --- | --- | --- | --- | --- | --- |
| probable | 1,164 | 0.912 | 0.008 | 23.74 | 17.87 | 1.33 |
| available | 5,865 | 0.867 | 0.004 | 21.50 | 15.24 | 1.41 |
| questionable | 2,273 | 0.551 | 0.010 | 14.69 | 17.29 | 0.85 |
| doubtful | 348 | 0.011 | 0.006 | 0.28 | 15.03 | 0.02 |
| out | 22,948 | 0.001 | 0.000 | 0.01 | 10.14 | 0.001 |

**Questionable really is a coin flip** (55.1%), doubtful is effectively out
(1.1%), and out means out (0.1%). The *learned* ordering matches the documented
one in every fold, which is a check rather than an assumption.

The ratio above 1.0 for available and probable is not an error: the baseline is a
player's mean minutes over his last ten games *including* games he missed as
zeros, so a player explicitly listed available is by construction healthier than
his own trailing average.

Participation is the **target** of this estimation and never a feature of the
game it came from. A fold's mapping is estimated from that fold's training
seasons alone, so a validation game cannot inform the mapping applied to itself,
and 2025-26 informs nothing at all.

### Player role, reusing Phase 3A3

A designation matters in proportion to the player. Role weight is the Phase 3A3
rotation machinery reused rather than a new player model: shrunk mean minutes
over the last ten prior games within the season, shrunk by appearances so one
big night does not make a call-up look like a starter.

A player with no prior history has an **unknown** role, and unknown contributes
nothing to the minute features rather than being recorded as zero — zero would
silently assert he does not matter. The count features still record that he was
designated, so the information is not lost, only kept in the right place.

### Development ablation

Four folds (2021-22 … 2024-25), training on all complete prior
availability-enabled seasons capped at five. The policy is fixed in advance and
never searched.

| bundle | adds | mean Brier | vs control | mean log loss | mean AUC |
| --- | --- | --- | --- | --- | --- |
| **C** | **role-weighted status minutes** | **0.21297** | **−0.00279** | 0.61393 | 0.7120 |
| D | training-calibrated expected minutes lost | 0.21347 | −0.00230 | 0.61496 | 0.7102 |
| E | D + quality-weighted loss | 0.21349 | −0.00227 | 0.61513 | 0.7103 |
| G | C + late news | 0.21368 | −0.00208 | 0.61551 | 0.7096 |
| F | D + late news | 0.21382 | −0.00194 | 0.61574 | 0.7092 |
| B | raw status counts | 0.21446 | −0.00131 | 0.61768 | 0.7086 |
| A | control (frozen Phase 3A3) | 0.21576 | — | 0.62041 | 0.7033 |

Read down that column and the ordering is the interesting part:

* **Role weighting is what matters.** Raw counts (B) are the weakest family by
  a wide margin, exactly as a deliberately weak baseline should be. Knowing four
  players are out is worth far less than knowing whose minutes they were.
* **The learned calibration did not beat simply splitting by status.** Bundle D
  compresses five statuses into one number using the fold's own mapping; bundle
  C keeps them separate and lets the regression weight them. C wins. The
  calibration is still reported — it is good basketball information — but the
  model does better with the statuses left apart.
* **Player quality adds nothing, again.** E ≈ D to four decimal places. Phase
  3A3 found generic player quality unhelpful; conditioning it on a player
  actually being missing does not rescue it. Discarded.
* **Late news adds nothing on top.** F and G are both *worse* than the simple
  families they extend. Movement between T-3h and T-30 is real — 626 of 1,230
  holdout games carry a late downgrade — but by T-30 the level already encodes
  it, so the change term is redundant.

Improvement is consistent across every fold, not driven by one unusual year:

| fold | control Brier | 3A3C Brier | delta | control AUC | 3A3C AUC |
| --- | --- | --- | --- | --- | --- |
| 2021-22 | 0.22214 | 0.21828 | −0.00386 | 0.6866 | 0.6981 |
| 2022-23 | 0.22531 | 0.22216 | −0.00315 | 0.6620 | 0.6751 |
| 2023-24 | 0.20942 | 0.20743 | −0.00199 | 0.7281 | 0.7320 |
| 2024-25 | 0.20623 | 0.20400 | −0.00222 | 0.7369 | 0.7427 |

### Frozen configuration

Written before the holdout was scored.

| | |
| --- | --- |
| bundle | **C** — Phase 3A3 control + role-weighted status minutes |
| availability features | `avail_{out,doubtful,questionable,probable}_expected_minutes_diff` |
| C | 0.1 |
| training history | most recent 5 complete availability-enabled seasons |
| player role | shrunk mean minutes, 10-game window, 3-game shrinkage, within season |
| status calibration | per fold, training seasons only, shrunk toward the pooled ratio |
| late news | anchors T-3h and T-1h; missing earlier report is unavailable, never "no change" |
| preprocessing | SimpleImputer → StandardScaler → LogisticRegression, fitted on training only |

Selection rule: lowest mean development Brier; inside a 1e-4 band prefer the
fewest added features, then the lower Brier, then the smaller C.

### 2025-26 holdout, 1,230 games

| model | Brier | log loss | accuracy | AUC | ECE |
| --- | --- | --- | --- | --- | --- |
| Kalshi T-30 normalized | **0.19465** | **0.57013** | 0.6911 | **0.7650** | 0.0335 |
| **Phase 3A3C native** | **0.20039** | 0.58539 | **0.6984** | 0.7505 | 0.0275 |
| Phase 3A3C harmonized | 0.20054 | 0.58574 | 0.6967 | 0.7500 | 0.0285 |
| Phase 3A3 (original) | 0.20369 | 0.59366 | 0.6894 | 0.7419 | 0.0269 |
| fold-matched control | 0.20380 | 0.59396 | 0.6894 | 0.7416 | 0.0248 |
| MOV Elo | 0.20440 | 0.59564 | 0.6927 | 0.7400 | 0.0395 |
| Phase 3A2 | 0.20451 | 0.59552 | 0.6927 | 0.7396 | 0.0337 |

Paired bootstrap, 10,000 resamples, fixed seed. Negative favours 3A3C.

| comparison | Brier difference | 95% CI | verdict |
| --- | --- | --- | --- |
| 3A3C native − Phase 3A3 | **−0.00329** | [−0.00595, −0.00068] | **3A3C better** |
| 3A3C native − matched control | −0.00341 | [−0.00605, −0.00077] | 3A3C better |
| 3A3C native − Kalshi | +0.00574 | [+0.00131, +0.01010] | Kalshi better |
| 3A3C harmonized − Phase 3A3 | −0.00314 | [−0.00579, −0.00049] | 3A3C better |
| 3A3C harmonized − Kalshi | +0.00589 | [+0.00145, +0.01026] | Kalshi better |

The improvement over Phase 3A3 is real and its interval excludes zero, on log
loss as well as Brier. AUC rises 0.7419 → 0.7505 and accuracy 0.6894 → 0.6984,
which puts 3A3C **ahead of Kalshi on accuracy** (0.6984 vs 0.6911) while still
clearly behind on Brier and AUC. Calibration is essentially unchanged
(ECE 0.0269 → 0.0275).

**36.5% of the Phase 3A3 → Kalshi Brier gap is closed** (0.00330 of 0.00904).

### The cadence shift does not explain the result

Development is hourly-era; the holdout is half-hourly from 2025-12-22, so part of
it carries fresher information than anything the model trained on. Harmonizing
the holdout down to the legacy grid moves mean report age from 3.5 to 9.8
minutes — squarely onto development's 9.6 — and moves Brier by **0.00015**.

| | mean age | p95 | share age 0 | Brier |
| --- | --- | --- | --- | --- |
| development | 9.6 min | 30 | 68.1% | — |
| holdout native | 3.5 min | 30 | 88.6% | 0.20039 |
| holdout harmonized | 9.8 min | 30 | 67.6% | 0.20054 |

The gain survives harmonization almost intact (−0.00314 vs −0.00329), so it comes
from knowing *who is unavailable*, not from the holdout being handed fresher
reports. Report age is **not** used as a model feature; development gave no
independent reason to add it.

### Where the improvement lives

| segment | games | control Brier | 3A3C Brier | delta |
| --- | --- | --- | --- | --- |
| top quartile expected minutes lost | 308 | 0.19202 | 0.18321 | **−0.00881** |
| late downgrade games | 626 | 0.19781 | 0.19152 | −0.00629 |
| high-minute OUT player present | 769 | 0.20371 | 0.19849 | −0.00522 |
| stable-status games | 253 | 0.19513 | 0.19135 | −0.00378 |
| late upgrade games | 790 | 0.21088 | 0.20850 | −0.00238 |
| questionable high-minute player | 30 | 0.18741 | 0.18936 | +0.00195 |
| **no meaningful availability burden** | **78** | **0.20539** | **0.20766** | **+0.00227** |

This is the sanity check that mattered most, and it passes. The gain is
concentrated where availability should matter — nearly three times the average in
the top expected-loss quartile — and **disappears where there is nothing to know**,
turning slightly negative in games with no meaningful burden. A uniform gain would
have suggested the features were proxying for something else.

## Phase 3A4 — nonlinear model class, calibration, ensembles

```bash
python -m nba_prediction_market.pipelines.build_nonlinear_model
```

Phase 3A3C closed about a third of the Brier gap to Kalshi by adding genuine
T-30 availability. The obvious next question is whether the rest is *structure*
the linear model cannot express — interactions between availability burden and
team strength, say — or simply information the model does not have.

**It is information.** Nonlinear modelling did not help, and the result is
unambiguous enough to be worth stating plainly rather than hedged.

### The answer in one table

**Zero of 120 nonlinear candidates beat the frozen Phase 3A3C logistic** on mean
development Brier. Not the best one, not narrowly — none of them.

| candidate | mean Brier | vs control | mean log loss | mean AUC | mean ECE |
| --- | --- | --- | --- | --- | --- |
| **Phase 3A3C logistic (control)** | **0.21297** | — | 0.61393 | 0.7120 | 0.0298 |
| hgb01 CORE raw | 0.21500 | +0.00203 | 0.61897 | 0.7072 | 0.0324 |
| xgb04 CORE raw | 0.21515 | +0.00218 | 0.61927 | 0.7068 | 0.0371 |
| xgb03 CORE raw | 0.21515 | +0.00218 | 0.61934 | 0.7066 | 0.0366 |
| xgb01 CORE raw | 0.21523 | +0.00226 | 0.61931 | 0.7062 | 0.0366 |
| hgb03 CORE raw | 0.21537 | +0.00240 | 0.61977 | 0.7062 | 0.0333 |
| best EXTENDED (hgb03) | 0.21643 | +0.00346 | — | — | — |

And it loses in **every fold**, not on average:

| fold | control Brier | best nonlinear | delta | control AUC | nonlinear AUC |
| --- | --- | --- | --- | --- | --- |
| 2021-22 | 0.21828 | 0.22085 | +0.00257 | 0.6981 | 0.6908 |
| 2022-23 | 0.22216 | 0.22318 | +0.00102 | 0.6751 | 0.6739 |
| 2023-24 | 0.20743 | 0.21113 | +0.00369 | 0.7320 | 0.7226 |
| 2024-25 | 0.20400 | 0.20485 | +0.00085 | 0.7427 | 0.7415 |

Four folds, four losses. There is no fold to cherry-pick.

### What was actually searched

The candidate ranges the brief specified cross to **128 configurations**, which
is a large enough search to find something by luck on ~6,000 training games.
Instead a **compact hand-picked grid of 16 XGBoost configurations** spans those
ranges, varying regularisation one axis at a time and moving depth, learning
rate and tree count together so total capacity stays comparable. Depth is capped
at 3. Every fit is seeded and deterministic.

Four `HistGradientBoostingClassifier` configurations were included for one
purpose: to separate "gradient boosting finds no interactions here" from "this
XGBoost setup was unlucky". The two families land in the same place
(0.21500 vs 0.21515), so the finding is about gradient boosting, not about
XGBoost.

20 configurations × 2 feature sets × 3 calibrations × 4 folds = 480 fits.

### Did previously rejected features become useful nonlinearly?

**No — they made things worse.** EXTENDED adds five families that earlier phases
built, audited, and then dropped under logistic regression: possession-adjusted
efficiency and Four Factors, roster continuity, generic player quality,
availability-weighted quality loss, and late-news status transitions. Nothing
new was engineered for this phase.

| feature set | features | best mean Brier | vs control |
| --- | --- | --- | --- |
| CORE (frozen 3A3C allowlist) | 19 | 0.21500 | +0.00203 |
| EXTENDED (+26 previously rejected) | 45 | 0.21643 | +0.00346 |

CORE beats EXTENDED by a clear margin. Linearity was not what made those
families look useless; they carry little signal conditional on what the model
already has, and 26 extra columns on 6,000 rows mostly add variance.

### Calibration made it worse, not better

Boosted trees often rank well and calibrate badly, so this was the most likely
place for a fix. Calibrators were fitted on **chronological out-of-fold
predictions generated entirely inside each fold's training history** — walk the
training seasons forward, predict each with a model fitted only on its
predecessors, fit the calibrator on those honest out-of-sample probabilities,
freeze it, then apply it to the outer validation season. The earliest training
season supplies training rows but is never scored, since nothing precedes it.

| calibration | best mean Brier | best mean ECE |
| --- | --- | --- |
| none (raw) | 0.21500 | 0.0324 |
| sigmoid (Platt) | 0.21579 | 0.0413 |
| isotonic | 0.21590 | 0.0360 |

Raw wins on both. The models are not badly calibrated in a way a monotone
transform can fix; they are simply slightly worse at ranking, and a calibrator
cannot add discrimination it was never given.

### Blending did not help either

| logistic weight | mean Brier | mean log loss | mean AUC |
| --- | --- | --- | --- |
| **1.00 (pure logistic)** | **0.21297** | 0.61393 | 0.7120 |
| 0.75 | 0.21298 | 0.61403 | 0.7122 |
| 0.50 | 0.21333 | 0.61490 | 0.7115 |
| 0.25 | 0.21400 | 0.61654 | 0.7098 |
| 0.00 (pure nonlinear) | 0.21500 | 0.61897 | 0.7072 |

Brier rises monotonically as weight shifts to the nonlinear model. The 0.75
blend is a hair better on AUC and a hair worse on Brier — well inside noise, and
not a reason to add a second model to the stack.

### Frozen configuration

The negative-result rule was applied as written: no expanded search, no deeper
trees, no new features.

| | |
| --- | --- |
| model | **`logistic_control`** — the frozen Phase 3A3C logistic, unchanged |
| reason | no nonlinear candidate beat it by the 5e-4 materiality threshold |
| feature set | CORE (19 features) |
| calibration | none |
| logistic blend weight | 1.00 |
| C | 0.1 |
| training history | most recent 5 complete availability-enabled seasons |
| random seed | 20260824 |

**Control reproduction: max |difference| = 0.0 across all 1,230 holdout games.**
Phase 3A3C is reproduced exactly, not approximately.

### 2025-26 holdout

| model | Brier | log loss | accuracy | AUC | ECE |
| --- | --- | --- | --- | --- | --- |
| Kalshi T-30 normalized | **0.19465** | **0.57013** | 0.6911 | **0.7650** | 0.0335 |
| **Phase 3A4 (= 3A3C logistic)** | **0.20039** | 0.58539 | **0.6984** | 0.7505 | 0.0275 |
| Phase 3A4 cadence-harmonized | 0.20054 | 0.58574 | 0.6967 | 0.7500 | 0.0285 |
| standalone nonlinear (hgb01 CORE) | 0.20129 | 0.58771 | 0.6927 | 0.7499 | 0.0376 |
| Phase 3A3 | 0.20369 | 0.59366 | 0.6894 | 0.7419 | 0.0269 |
| MOV Elo | 0.20440 | 0.59564 | 0.6927 | 0.7400 | 0.0395 |

The standalone nonlinear model confirms the development finding out of sample:
worse Brier, worse log loss, and notably worse calibration (ECE 0.0376 vs
0.0275) — the miscalibration the out-of-fold calibrators failed to repair.

Paired bootstrap, 10,000 resamples, fixed seed:

| comparison | Brier difference | 95% CI |
| --- | --- | --- |
| Phase 3A4 − Phase 3A3C | +0.00000 | [+0.00000, +0.00000] |
| Phase 3A4 − Kalshi | +0.00574 | [+0.00131, +0.01010] |
| Phase 3A4 harmonized − Phase 3A3C | +0.00015 | [−0.00036, +0.00063] |
| Phase 3A4 harmonized − Kalshi | +0.00589 | [+0.00145, +0.01026] |

The first row is identically zero because the frozen model *is* Phase 3A3C.
Cadence harmonization again moves nothing (CI spans zero), reproducing the
3A3C sensitivity result without retuning.

**0% of the remaining gap was closed.** The Phase 3A3 → Kalshi gap was 0.00904;
Phase 3A3C closed 0.00330 (36.5%) and Phase 3A4 closed none of the remaining
0.00574.

### Where the nonlinear model differed

Diagnostics only — the configuration was frozen before the holdout was scored,
so nothing here could have altered it. Because the frozen model is the control,
the informative comparison is the standalone nonlinear model against it:

| segment | games | 3A3C Brier | nonlinear Brier | delta |
| --- | --- | --- | --- | --- |
| both sides heavily designated | 473 | 0.19711 | 0.19918 | +0.00208 |
| low availability burden | 308 | 0.20700 | 0.20896 | +0.00196 |
| underdogs | 307 | 0.18364 | 0.18520 | +0.00156 |
| late season | 615 | 0.17978 | 0.18136 | +0.00158 |
| high availability burden | 308 | 0.18321 | 0.18439 | +0.00117 |
| high-minute OUT present | 769 | 0.19849 | 0.19934 | +0.00085 |
| close games | 390 | 0.24189 | 0.24180 | −0.00009 |

This is the part that makes the negative result convincing. The **both sides
heavily designated** segment is precisely where the brief expected interactions
to live — several important players carrying simultaneous statuses — and it is
the segment where boosting does *worst*. Only close games are a tie, and that is
where every model is near 0.24 anyway.

### Feature importance

Permutation importance on development data (fold 2024-25, negative-Brier
scoring), reported descriptively:

| feature | permutation importance |
| --- | --- |
| `mov_elo_diff` | +0.03831 |
| `avail_out_expected_minutes_diff` | +0.00732 |
| `elo_diff` | +0.00181 |
| `last5_point_diff_difference` | +0.00145 |
| `last10_point_diff_difference` | +0.00115 |

Margin-of-victory Elo dominates by an order of magnitude, and **role-weighted
OUT burden is the second most important feature in the model** — ahead of every
other strength measure. That is consistent with Phase 3A3C: availability
carries real signal, and it is concentrated almost entirely in who is OUT.
`avail_doubtful_expected_minutes_diff` contributes nothing measurable, which
matches its calibration (doubtful players play 1.1% of the time, so the feature
is nearly collinear with OUT).

Importance is not causal here: these features correlate with each other, and a
family absorbing another's credit is expected.

### What this means

The gap to Kalshi is **information-limited, not model-limited**. Two independent
gradient-boosting families, twenty configurations, two feature sets, three
calibrations and five blend weights all failed to find structure that logistic
regression on the same information was missing. The remaining 0.00574 Brier is
unlikely to be recovered by a better fit to what we already know.

Kalshi still wins on Brier, log loss and AUC — but Phase 3A4 is **ahead on
accuracy** (0.6984 vs 0.6911) and **better calibrated** (ECE 0.0275 vs 0.0335).
The market's edge is in discrimination, which is what a market with access to
information we do not have should look like.

## Phase 4A0 — execution-aware Kalshi market edge audit

```bash
python -m nba_prediction_market.pipelines.build_market_edge_audit
```

**Research only, and exploratory.** Nothing here places an order. 2025-26 was
inspected repeatedly throughout model development, so no result in this section
is validation of anything.

**Conclusion: classification A — no evidence our model's disagreement with
Kalshi is useful.** The evidence is not merely absent; it points the other way.

### There is no earlier season to develop on

Established empirically rather than from documentation. `KXNBAGAME`'s earliest
event is **2025-04-15**, and **zero events fall before 2025-04-13**, the end of
the 2024-25 regular season. The 86 events in April-June 2025 are that season's
play-in and playoffs.

| season | KXNBAGAME regular-season markets |
| --- | --- |
| 2022-23 | none |
| 2023-24 | none |
| 2024-25 | none (play-in and playoffs only) |
| 2025-26 | full — 1,230 games with T-30 quotes |

Eight plausible legacy tickers (`NBAGAME`, `NBA`, `NBAWIN`, `NBAWINNER`,
`PRONBAGAME`, `NBAG`, `NBAMONEYLINE`, `NBAML`) return no events, and
`KXNBAGAME` is the only game-winner series in Kalshi's catalogue — `KXNBA` is
championship futures. So market rules cannot be developed on one season and
evaluated on another. That is why this phase is exploratory by construction.

### Fees, dated and sourced

Not hardcoded from memory. Two dated copies of Kalshi's published schedule were
retrieved from the Internet Archive, covering the season:

| effective | taker | maker | source |
| --- | --- | --- | --- |
| Oct 1, 2025 | `round up(0.07 x C x P x (1-P))` | `round up(0.0175 x C x P x (1-P))` | [archived 2025-10-08](https://web.archive.org/web/20251008232930id_/https://kalshi.com/docs/kalshi-fee-schedule.pdf) |
| Feb 5, 2026 | `round up(0.07 x C x P x (1-P))` | `round up(0.0175 x C x P x (1-P))` | [archived 2026-02-14](https://web.archive.org/web/20260214014036id_/https://kalshi.com/docs/kalshi-fee-schedule.pdf) |

The schedule *did* change mid-season, which is why the engine is date-effective —
but the general formulas are identical across both versions. A trade date
outside every documented window raises rather than falling back to the newest
schedule.

**KXNBAGAME has no product-specific fee.** Verified on the documents themselves:
zero mentions of NBA, Basketball, Sports or KXNBAGAME in either version. The only
product-specific schedules are S&P 500 and Nasdaq-100 index products.

**Rounding is per order, not per contract**, so the per-contract cost falls with
size. At a 50c price: $0.0200 at 1 contract, $0.0180 at 10, **$0.0175 at 100** —
where it equals the exact formula. That rounding-stability is why 100 contracts
is the headline view. It is not a bankroll recommendation.

A floating-point subtlety worth naming: `0.07 x 100 x 0.5 x 0.5` is exactly
$1.75, but in binary it lands a hair above and ceilings to $1.76. The engine
uses decimal arithmetic, so fees are not systematically overstated.

### Market quality at T-30

All 1,230 games have usable quotes on both sides. Spreads are 1c on 1,144 games
and 2c on 86.

| | min | median | max | mean |
| --- | --- | --- | --- | --- |
| bid sum | 0.970 | 0.990 | 1.010 | 0.9917 |
| ask sum | 0.990 | 1.010 | 1.030 | 1.0130 |
| midpoint sum | 0.980 | 1.000 | 1.020 | 1.0024 |

An ask sum above 1.0 is just the round-trip cost of the spread. **Twelve games
show a bid sum above 1.0**, which would be a genuine crossed pair; these are
almost certainly artifacts of aggregating a one-minute candle rather than
standing arbitrage, and they are reported rather than repaired.

### Who is more right when we disagree?

This is the question that decides the phase, and it is answered before any P&L.
Bins were declared in advance.

| disagreement (model − market) | games | model p | market p | **actual** | model Brier | market Brier | model − market |
| --- | --- | --- | --- | --- | --- | --- | --- |
| < −10% | 133 | 0.534 | 0.675 | **0.654** | 0.2074 | 0.1971 | +0.0103 |
| −10 to −5% | 208 | 0.591 | 0.663 | **0.683** | 0.1984 | 0.1904 | +0.0080 |
| −5 to −3% | 118 | 0.615 | 0.654 | 0.619 | 0.1906 | 0.1931 | −0.0025 |
| −3 to −1% | 128 | 0.615 | 0.636 | 0.656 | 0.1822 | 0.1809 | +0.0013 |
| −1 to +1% | 130 | 0.547 | 0.548 | 0.569 | 0.1728 | 0.1730 | −0.0003 |
| +1 to +3% | 108 | 0.512 | 0.492 | 0.509 | 0.2086 | 0.2080 | +0.0005 |
| +3 to +5% | 105 | 0.512 | 0.472 | 0.514 | 0.2092 | 0.2112 | −0.0020 |
| +5 to +10% | 155 | 0.495 | 0.421 | **0.381** | 0.1978 | 0.1847 | +0.0131 |
| > +10% | 145 | 0.506 | 0.357 | **0.372** | 0.2360 | 0.2200 | +0.0160 |

Read the two tails. Where the model is *much lower* on the home side than the
market, the home side won 65.4% — the market said 67.5%, the model said 53.4%.
Where the model is *much higher*, the home side won 37.2% — the market said
35.7%, the model said 50.6%. **In both tails the outcome tracks the market.**

And the pattern strengthens with disagreement:

| \|disagreement\| quartile | games | mean \|d\| | model Brier | market Brier | difference | 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | 308 | 0.012 | 0.18841 | 0.18907 | −0.00067 | [−0.0021, +0.0007] |
| Q2 | 307 | 0.038 | 0.19099 | 0.18986 | +0.00112 | [−0.0028, +0.0050] |
| Q3 | 307 | 0.072 | 0.20141 | 0.19334 | **+0.00806** | [+0.0010, +0.0151] |
| Q4 | 308 | 0.141 | 0.22075 | 0.20631 | **+0.01444** | [−0.0008, +0.0298] |

The larger the disagreement, the more the market outperforms. That is the exact
opposite of a tradable signal: when our model departs from the market, the model
is the one that is wrong. Correlation between disagreement and the market's
residual is **+0.019** — indistinguishable from none.

### Every predeclared threshold, taker execution, 100 contracts

Threshold = model probability − executable ask − applicable per-contract fee.

| threshold | trades | coverage | hit rate | spent | fees | net P&L | ROI | 95% CI on P&L/trade |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| > 0% | 986 | 80.2% | 36.8% | $37,080 | $1,352 | −$2,132 | −5.75% | [−4.75, +0.41] |
| ≥ 1% | 860 | 69.9% | 36.5% | $31,780 | $1,182 | −$1,562 | −4.91% | [−4.46, +0.78] |
| ≥ 2% | 751 | 61.1% | 36.4% | $27,263 | $1,038 | −$1,001 | −3.67% | [−4.13, +1.60] |
| ≥ 3% | 646 | 52.5% | 34.4% | $23,044 | $892 | −$1,736 | −7.53% | [−5.85, +0.51] |
| ≥ 5% | 470 | 38.2% | 33.4% | $16,510 | $657 | −$1,467 | −8.88% | [−6.58, +0.32] |
| ≥ 7.5% | 302 | 24.6% | 34.4% | $10,145 | $424 | −$169 | −1.67% | [−5.13, +3.87] |
| ≥ 10% | 184 | 15.0% | 37.5% | $6,264 | $262 | **+$374** | **+5.97%** | [−4.06, +7.93] |

Six of seven thresholds lose money. **Every interval spans zero**, including the
one positive result. The sequence is also non-monotonic (−5.75, −4.91, −3.67,
−7.53, −8.88, −1.67, +5.97), which is what noise looks like rather than a signal
strengthening with selectivity.

### The one positive result contradicts itself

The ≥10% threshold deserves a direct look, because it is exactly the number a
careless reading would seize on.

**All 184 of those trades come from games where |disagreement| > 10%** — the two
tail bins in the calibration table above. On those same 184 games the model's
Brier is 0.22980 against the market's 0.21881: **the model is measurably worse
there than anywhere else**. The trades won 37.5% against a breakeven of 35.5%,
a two-point margin on 184 bets.

So the only profitable threshold selects precisely the games where our model is
least accurate, and its profit is a two-point hit-rate margin with an interval
four times its own width. It is not an edge.

### Diagnostics, all negative

| segment | trades | hit | ROI |
| --- | --- | --- | --- |
| combined spread 2c | 885 | 0.358 | −7.10% |
| combined spread 3c | 86 | 0.453 | +1.10% |
| combined spread 4c | 15 | 0.467 | +29.93% |
| volume Q4 (most liquid) | 247 | 0.352 | −12.29% |
| open interest Q1 | 247 | 0.409 | +3.81% |
| low availability burden | 326 | 0.334 | −10.84% |
| high availability burden | 335 | 0.415 | +2.09% |
| high-minute questionable | 24 | 0.458 | +13.80% |
| model likes the market favourite more | 147 | 0.626 | −7.40% |
| model thinks the favourite is overpriced | 426 | 0.268 | −4.58% |

The apparent positives are the small samples — 15 games at a 4c spread, 24 with
a high-minute questionable player. The "wider spread pays better" pattern runs
backwards from how execution costs work and is a straightforward small-sample
artifact. The most liquid quartile is the worst performing.

The availability interaction is the one worth a second look, since availability
was the largest genuine model improvement: high-burden games return +2.09%
against −10.84% for low-burden ones. That is directionally consistent with the
model knowing something extra where availability matters, but on 335 games with
a hit rate of 0.415 it is far short of evidence, and it is not a result to tune
on.

### Maker execution does not rescue it

Hypothetical only — a resting order is not a fill, and the candlestick data
carries no fill evidence. Even so, the arithmetic at the current best bid gives a
mean net edge of **−0.0010 (home)** and **+0.0023 (away)**. Essentially
breakeven before any assumption about whether the order would ever trade. **No
conclusion here depends on maker assumptions.**

### Recommendation

**No candidate strategy is proposed for 2026-27.** The pre-registration slot is
deliberately left empty: proposing a rule built on the ≥10% threshold would mean
pre-registering the one region where the model is demonstrably worse than the
market.

**Multi-anchor Kalshi backfill is not justified yet.** The T-30 result is not
"our edge decays by tip" — it is "our disagreement is anti-predictive at T-30".
Earlier anchors would refine the timing of a signal that has not been shown to
exist. The 2,460 calls should wait for a reason.

The honest summary: the frozen model is a good *forecaster* — better calibrated
than Kalshi (ECE 0.0275 vs 0.0335) and more accurate (0.6984 vs 0.6911) — and
still a worse *probability* (Brier 0.20039 vs 0.19465). Where the two disagree,
the market wins, and it wins by more the louder we disagree.

## Phase 4A1 — multi-anchor convergence and availability news reaction

```bash
python -m nba_prediction_market.pipelines.build_multi_anchor_market
python -m nba_prediction_market.pipelines.build_anchor_models
python -m nba_prediction_market.pipelines.build_availability_event_study
```

**Research only, and exploratory.** Nothing here trades.

Phase 4A0 showed no tradable disagreement at T-30. That was a statement about
one instant. This phase asks whether the market is less efficient *earlier*,
when the injury picture is incomplete — and whether it prices official NBA
availability news slowly enough to matter.

**Classification: C.** There is real, measurable evidence that Kalshi does *not*
absorb official injury news instantaneously. It is also economically useless:
the entire adjustment is smaller than the bid-ask spread.

### The backfill

2,460 markets, one request each, covering a six-hour window ending at T-30:
**784,156 candles, zero failures.** The cache slug is `t30_lb360_p1`, so Phase
2's `t30_lb60_p1` is untouched.

| anchor | markets | usable | games with both sides | quote age p95 |
| --- | --- | --- | --- | --- |
| T-6h | 2,460 | 2,313 | **1,103** | 0s |
| T-3h | 2,460 | 2,460 | 1,230 | 0s |
| T-1h | 2,460 | 2,460 | 1,230 | 0s |
| T-30m | 2,460 | 2,460 | 1,230 | 0s |

T-6h is the one incomplete anchor — 127 games had no two-sided market six hours
before tip, because the market had not opened yet. That is itself worth knowing:
the earliest anchor is partly unavailable, not merely less informative.

T-15m and T-5m are **structurally absent**: the fetched window ends at T-30, so
they were never retrieved. Their exclusion is a property of the data, not of
discipline.

### The market leads at every anchor, by a near-constant margin

Each anchor model is the frozen Phase 3A3C bundle-C specification refitted on
information available at that anchor. No new feature families, same history
policy, same small C grid chosen on development folds only. All four were frozen
before the market was consulted. Every anchor selected C = 0.1.

The T-30 anchor model reproduces the frozen Phase 3A3C predictions **exactly**
(max |difference| = 0.0 over 1,230 games), which is the check that the anchor
machinery is doing what it claims.

| anchor | games | model Brier | market Brier | difference | 95% CI | model AUC | market AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-6h | 1,103 | 0.20678 | 0.20001 | +0.00677 | [+0.0020, +0.0114] | 0.7338 | 0.7528 |
| T-3h | 1,230 | 0.20107 | 0.19547 | +0.00560 | [+0.0012, +0.0100] | 0.7485 | 0.7627 |
| T-1h | 1,230 | 0.20099 | 0.19488 | +0.00611 | [+0.0018, +0.0104] | 0.7488 | 0.7643 |
| T-30m | 1,230 | 0.20039 | 0.19465 | +0.00574 | [+0.0014, +0.0101] | 0.7505 | 0.7650 |

**The market wins at every anchor and every interval excludes zero.** The margin
does not shrink earlier in the day — if anything T-6h is the worst anchor for us.
Both sides sharpen toward tip in lockstep; the market's lead is stable.

The familiar pattern holds throughout: our model is **better calibrated** at
every anchor (ECE 0.019-0.028 against 0.025-0.034) and more accurate at three of
four, while still being the worse probability.

### The gap does not converge

| anchor | games | mean \|gap\| | median \|gap\| |
| --- | --- | --- | --- |
| T-6h | 1,103 | 0.0679 | 0.0556 |
| T-3h | 1,230 | 0.0669 | 0.0549 |
| T-1h | 1,230 | 0.0659 | 0.0535 |
| T-30m | 1,230 | 0.0659 | 0.0533 |

Six hours of price discovery and new injury reports move the average
disagreement by **0.002**. Between consecutive anchors the gap shrinks about
half the time and expands about half the time — a coin flip.

| transition | games | gap shrank | expanded | sign flip | mean model move | mean market move |
| --- | --- | --- | --- | --- | --- | --- |
| T-6h → T-3h | 1,103 | 48.4% | 569 | 79 | 0.0128 | 0.0096 |
| T-3h → T-1h | 1,230 | 54.1% | 565 | 90 | 0.0139 | 0.0092 |
| T-1h → T-30m | 1,230 | 49.0% | 627 | 52 | 0.0081 | 0.0052 |

Our probability moves **more** than the market's at every step — new availability
information genuinely arrives — and yet the gap does not close. The model's
movement is not toward the market. That is the signature of two forecasts that
differ in content rather than in timeliness.

### Executable edge, every anchor, every predeclared threshold

Same thresholds as Phase 4A0, not re-chosen per anchor. Taker at the ask,
date-effective fees, 100 contracts.

**26 of 28 anchor-threshold combinations lose money.**

| threshold | T-6h | T-3h | T-1h | T-30m |
| --- | --- | --- | --- | --- |
| > 0% | −4.28% | −4.73% | −4.85% | −5.75% |
| ≥ 1% | −2.64% | −5.39% | −4.75% | −4.91% |
| ≥ 2% | −4.17% | −5.37% | −4.53% | −3.67% |
| ≥ 3% | −2.66% | −4.66% | −9.10% | −7.53% |
| ≥ 5% | −4.01% | −6.81% | −9.83% | −8.88% |
| ≥ 7.5% | −9.54% | −2.88% | −2.17% | −1.67% |
| ≥ 10% | **−7.67%** | +8.06% | +7.12% | +5.97% |

The two survivors are the ≥10% threshold at T-3h and T-1h, and both intervals
span zero ([−3.46, +8.86] and [−4.05, +9.27] per trade). More telling: **at T-6h
— the earliest anchor, where an unpriced-news story would be strongest — that
same threshold loses 7.67%.** The one threshold that looks profitable is the one
that fails precisely where the hypothesis predicts it should work best.

**No earlier anchor is more promising than T-30.**

### Does Kalshi price official injury news immediately? No — but it does not matter

This is what the report archive makes uniquely answerable. An event is a change
between consecutive official reports for the same player and game, timestamped
by the league. Participation never defines an event.

**2,705 events in the six-hour window; 2,543 with usable market observations.**

| | |
| --- | --- |
| downgrades / upgrades | 1,393 / 1,150 |
| role bands (high / medium / low / unknown) | 711 / 898 / 656 / 278 |
| commonest transitions | questionable→available (1,055), questionable→out (885), doubtful→out (455) |
| median pre-quote latency | 60s |
| median post-quote latency | **0s** |

Moves are signed toward the direction the news implies. Events that move the
other way are kept, not discarded.

**All 2,543 events:**

| horizon | n | mean move | share > 0 | 95% CI |
| --- | --- | --- | --- | --- |
| immediate | 2,543 | +0.00015 | 4.1% | [−0.00001, +0.00039] |
| +5 min | 2,543 | +0.00127 | 13.3% | [+0.00082, +0.00178] |
| +15 min | 1,817 | +0.00202 | 23.5% | [+0.00133, +0.00281] |
| +30 min | 1,810 | +0.00221 | 27.5% | [+0.00146, +0.00304] |
| +60 min | 1,009 | +0.00265 | 34.1% | [+0.00132, +0.00405] |

**High-role events only (711 events, ≥20 expected minutes):**

| horizon | n | games | mean move | 95% CI |
| --- | --- | --- | --- | --- |
| immediate | 711 | 491 | +0.00027 | [−0.00005, +0.00070] |
| +5 min | 711 | 491 | +0.00290 | [+0.00160, +0.00446] |
| +15 min | 489 | 358 | +0.00469 | [+0.00267, +0.00725] |
| +30 min | 487 | 356 | **+0.00562** | [+0.00373, +0.00786] |
| +60 min | 265 | 205 | +0.00660 | [+0.00306, +0.01095] |

The immediate interval **includes zero at both scopes**. The market does not jump
on the report stamp. Reaction speed, on events with a move large enough for a
ratio to be meaningful:

| fraction of the 30-minute move present by | median | mean |
| --- | --- | --- |
| immediately | 0.00 | 0.06 |
| +5 min | 0.00 | 0.40 |
| +15 min | 1.00 | 0.65 |

**So adjustment is genuinely gradual — roughly 5 to 15 minutes.** That is a real
market-microstructure finding and it contradicts a naive efficient-market story.

And it is economically worthless. The entire 30-minute reaction to a **high-role**
availability change is **0.0056** — against a median spread of **0.01**. Capturing
the whole adjustment, perfectly, with no fee and no latency, would recover about
half the round-trip cost of the spread. The spread does not widen defensively
either: unchanged in 2,428 of 2,543 events, wider in only 62.

A midpoint that drifts half a cent over a quarter of an hour is not a trade.

### One genuinely interesting result

The availability-burden diagnostic, run at every anchor, on model-minus-market
Brier (negative = our model better):

| anchor | low burden (Q1) | mid | high burden (Q4) |
| --- | --- | --- | --- |
| T-6h | +0.00712 | +0.00566 | +0.00851 |
| T-3h | +0.00599 | +0.00719 | +0.00202 |
| T-1h | +0.00104 | +0.01119 | +0.00104 |
| T-30m | +0.00408 | +0.00949 | **−0.00007** |

In high-burden games the gap closes monotonically as tip approaches — +0.0085,
+0.0020, +0.0010, −0.0001 — until at T-30 the model **exactly ties the market**.
That is the one cell in the entire phase where we are not behind, and it is
where availability information is most complete and most decisive.

It explains Phase 4A0's weak +2.09% hint without vindicating it. Our availability
work buys back precisely enough to match the market where availability dominates,
and not a basis point more. A tie is not an edge, and this one arrives at T-30 —
not earlier, which is what the phase set out to test.

### No strategy is proposed

Phase 4A0 declined to pre-register a rule. Phase 4A1 declines again, for a
sharper reason: the slow-adjustment finding is real but sub-spread, and the only
profitable threshold fails at the anchor where its own hypothesis is strongest.

**Prospective 2026-27 discovery/validation protocol.** Since no historical
strategy exists, the split must be declared before any 2026-27 outcome is seen:

* **Discovery** — games through **2026-12-31**. Capture only. Any hypothesis must
  be written down, with its threshold and anchor, before the period closes.
* **Freeze** — 2027-01-01. The hypothesis file is committed and not edited.
* **Validation** — games from **2027-01-01** to the end of the regular season.
  These games score the frozen hypothesis and are never used to form one.

The two periods share no games. If discovery produces nothing worth freezing,
validation scores nothing — which is a legitimate outcome, not a failure to be
worked around.

The prospective collector already preserves what this needs: every official
report snapshot with its exact timestamp, the normalized availability state, and
market quotes at each anchor. The one addition worth making is finer-grained
market capture immediately around report timestamps, which is what made this
phase's event study possible at all.

## Phase 1 run results (2025-26)

From a live run on 2026-08-19 (`--season 2025`):

| | Count |
| --- | --- |
| NBA games | 1,322 (85 postseason, all final) |
| Kalshi `KXNBAGAME` markets | 2,726 (1,363 events) |
| **matched** | **1,317** (99.6% of NBA games, 96.6% of Kalshi events) |
| unmatched NBA | 5 |
| unmatched Kalshi | 46 |
| ambiguous | 0 |

1,316 matches were exact; 1 came from the ±1 day tier (GSW at MIN, labelled
Jan 24 by Kalshi and Jan 25 by BALLDONTLIE).

**Independent validation:** on all 1,317 matched rows the Kalshi settlement
agrees with the NBA final score, and home/away orientation agrees. Zero
disagreements. That is a strong signal the join is correct, since the two
sources settle those fields independently.

Every unmatched record has an identified cause:

* **43 Kalshi events, Oct 10-17 2025** — preseason. The regular season opened
  Oct 21, and BALLDONTLIE's `seasons[]=2025` does not return preseason games.
* **3 Kalshi events with real volume and no NBA game anywhere near them** —
  MIA at CHI (Jan 8), DAL at MIL (Jan 25), DEN at MEM (Jan 25). Each has
  1.0-1.7M volume, and BALLDONTLIE has no game for that pair within five days.
  Most likely postponed fixtures that Kalshi listed and the schedule later moved
  (CHI/MIA subsequently played three times in eight days). **Worth review.**
* **1 NBA game** — SAS at NYK, 2025-12-16, `ist_stage = "Championship"`: the NBA
  Cup final. It carries `postseason = False` and has no `KXNBAGAME` event.
* **4 NBA games** — all four are the **Game 7** of their series (BOS/PHI May 2,
  CLE/TOR May 3, DET/ORL May 3, CLE/DET May 17). In each case Kalshi's
  `KXNBAGAME` series stops at Game 6. **Worth review** — if Game 7s live under a
  different series ticker, that series needs ingesting too.
* **2 markets flagged `is_nba_matchup = False`** — a Guangzhou (`GUA`)
  exhibition against Minnesota. Correctly excluded from matching rather than
  forced onto a canonical team.

## Known limitations & assumptions to review

* **Availability history reaches back to 2018-12-17, not eight months.** The
  earlier "retention boundary" was a filename-convention change misread as
  deletion (Phase 3A3B2). Reports before the 2025-12-22 cutover use the legacy
  hourly name, and the filename hour is *half an hour earlier* than the report it
  serves. Anything reading these URLs must handle both conventions, and must not
  treat the legacy filename hour as the observation time.
* **One player remains unresolved by choice.** `Djurisic, Nikola` (ATL, 2,246
  rows) has no canonical identifier in any source searched. Sixteen of the
  original seventeen pairs are resolved through a verified alias table with
  per-entry evidence; there is deliberately no similarity-based fallback, so new
  mismatches will surface as unresolved rather than be silently absorbed.
* **The pre-first-appearance identity pattern is recurring, and the current
  remedy is manual.** A player traded mid-season appears on his new team's injury
  report before any box score for it, so a season-level registry has him
  elsewhere. Each instance is listed with evidence; a date-aware roster is the
  durable fix and is not yet built.
* **The 2019-20 and 2020-21 availability features are materially staler than
  every later season.** Those seasons published three reports a day, giving a p95
  report age of 150 minutes against 30 from 2021-22 onward. They are usable as
  training history for the first development fold, but they are not evidence
  about how good availability features can be, and 2019-20 additionally has 115
  games with no T-30 state at all.
* **Kalshi absorbs official injury news gradually, not instantly — and it does
  not matter.** The immediate reaction is statistically zero; adjustment takes
  5-15 minutes. But the entire 30-minute move for a high-role status change is
  0.0056 against a 1-cent spread, so capturing all of it perfectly would recover
  about half the round-trip cost. A real microstructure finding with no
  economic content.
* **No earlier anchor is better than T-30.** The market leads at T-6h, T-3h,
  T-1h and T-30m alike, every interval excluding zero, and the disagreement gap
  barely converges (0.0679 to 0.0659 over six hours). At T-6h even the one
  otherwise-profitable threshold loses 7.67%.
* **No tradable edge has been demonstrated, and 2025-26 cannot demonstrate one.**
  Phase 4A0 found the market more accurate wherever the two disagree, and more
  so as disagreement grows. 2025-26 was inspected throughout model development
  and Kalshi published no earlier NBA regular-season markets, so there is no
  clean season on which to develop a market rule and no season on which to
  validate one. Any future trading claim needs prospective paper trading.
* **The only profitable edge threshold selects the games the model is worst at.**
  All 184 trades clearing a 10% net edge come from the >10% disagreement tails,
  where the model's Brier is 0.2298 against the market's 0.2188. Its interval
  spans zero and is four times the point estimate.
* **Slippage and depth are not modelled.** The simulation fills at the quoted
  ask from a one-minute candle aggregate. There is no order-book data, so real
  fills at size could be worse; no fictional slippage was invented to cover the
  gap.
* **Nonlinear modelling was tested and rejected, and the search was
  deliberately not widened.** Zero of 120 gradient-boosted candidates beat the
  frozen logistic on development folds, so the negative-result rule applied: no
  deeper trees, no expanded grid, no new features. A later phase that revisits
  model class should bring *new information*, not a bigger search over the same
  information.
* **Late-news features are built and tested but not used.** Movement between
  T-3h and T-30 is real, yet adding it made development performance worse: by
  T-30 the level already encodes the movement. The machinery stays because
  T-15m/T-5m research may want it, but nothing in the frozen model consumes it.
* **Availability rows for postponed occurrences are withheld, not remapped.**
  They describe a game that did not happen — two of the four were replayed weeks
  later — so they are never attached to the replayed game.
* **`suspect_blocked_days` is a conservative heuristic, not a verdict.** It
  flags near-empty days bracketed by complete days, and on the current archive it
  fires on two Finals off-days. The canary evidence — a known-good URL returning
  200 throughout the run — is the authority on whether a run was throttled.
* **Season window.** A season is assumed to fall entirely within 1 Jul (year N)
  through 30 Jun (year N+1). Kalshi's archive is one undated multi-season
  stream, so *some* explicit window is unavoidable. It is defined in
  `config.season_window` and tested, not buried in the matcher.
* **±1 day matching tier.** Justified by the two sources occasionally labelling
  a late tip-off on adjacent calendar days. It is deliberately narrow and
  requires mutual uniqueness, but it is a judgement call — the report breaks out
  how many matches came from it (`match_tiers`), and that number is worth
  eyeballing after every run.
* **Event ticker team codes are assumed to be away-then-home** (`...NYKSAS` =
  NYK at SAS). Verified against every market whose title uses the `"A at B"`
  form, with zero counter-examples, and cross-checked against the event
  `sub_title` per row.
* **Team codes in tickers are assumed to be exactly three characters.** A
  flexible width would split `NYKSAS` incorrectly. Non-conforming tickers parse
  to "unknown" rather than to a guess.
* **Non-NBA opponents exist in the series.** The archive contains at least one
  exhibition against a non-NBA club, whose code is not in the canonical map.
  Such markets are flagged `is_nba_matchup = False` and get no matchup key, so
  they can never match a game.
* **Kalshi price fields are metadata snapshots**, not a pregame price. They are
  whatever the market last showed. Deriving an actual pregame price needs
  candlesticks — the next task.
* **No candlestick / time-series data yet.** Phase 1 is metadata only.
* **Rate limiting is a fixed minimum interval**, tuned to the BALLDONTLIE free
  tier (12.5s). A paid tier can go much faster via
  `BALLDONTLIE_MIN_INTERVAL_SECONDS`.

**Phase 2**

* **`quote_usable` bundles two conditions** — fresh enough *and* two-sided. The
  granular reason is always in `quote_issue`, so the two can be separated if you
  later want a one-sided quote to count as usable.
* **Derived prices are rounded to 6dp.** Kalshi quotes whole cents, so midpoints
  land on half-cents; left unrounded, `abs(0.99 - 1.0)` evaluates to
  `0.010000000000000009` and compares greater than 0.01, which inflated the
  deviation-threshold counts. 6dp is lossless for every real value.
* **Candle fetching is sequential**, not concurrent. A cold run is ~20 minutes;
  the cache makes every re-run ~3 seconds. Determinism and a trivially resumable
  run were worth more than the wall-clock saving.
* **The cache is keyed by request geometry.** Changing `--minutes-before-tip`,
  `--lookback-minutes`, or `--period-interval` writes to a different directory,
  so windows can never be mixed; changing them does mean refetching.
* **One game had a two-sided ask sum below $1.00** (0.99), a theoretical
  1-cent lock before fees. One occurrence in 1,236 games is a market artefact,
  not a data error, but it is worth knowing the data contains such rows.
* **`--minutes-before-tip` uses the *scheduled* tipoff.** If a game's actual
  start slipped, the anchor still refers to the schedule. BALLDONTLIE's
  `datetime` is the only start time available, and it is not marked as
  scheduled-vs-actual.

## Layout

```
src/nba_prediction_market/
  config.py                     settings, paths, season conventions
  clients/base.py               timeouts, retries, rate limiting, pagination
  clients/balldontlie.py        GET /v1/games
  clients/kalshi.py             both market stores + events + cutoff
  ingestion/raw_store.py        verbatim raw-payload persistence
  ingestion/nba_games.py        game normalization + season verification
  ingestion/kalshi_markets.py   market normalization + field derivation
  ingestion/candlesticks.py     candle parsing + lookahead-safe quote selection
  ingestion/candle_cache.py     resumable per-market raw response cache
  matching/team_names.py        30 franchises, exact aliases, no fuzzy matching
  matching/game_market_matcher.py   deterministic join + classification
  availability/sources.py            audited source matrix + as-of classes
  availability/snapshot_store.py     append-only immutable capture store
  availability/as_of.py              observed_at <= prediction_ts enforcement
  availability/identity.py           player resolution, no silent fuzzy matching
  availability/capture_schedule.py   schedule-aware prospective capture planner
  availability/nba_official.py       report URL slots + immutable PDF archive
  availability/nba_report_parser.py  coordinate-based PDF table parser
  availability/archive_inventory.py  archive coverage, gaps, blocked-run suspicion
  availability/runner.py             prospective capture runner + anchor health
  availability/prospective_adapters.py  forward-only secondary feeds
  availability/kalshi_anchors.py     multi-anchor benchmark reconstruction
  availability/player_aliases.py     verified identity corrections, no fuzzy fallback
  availability/postponements.py      postponed games, evidenced against ESPN
  availability/external_sources.py   third-party archives on one normalized schema
  availability/reason_categories.py  broad deterministic reason buckets (diagnostic)
  features/availability_features.py  role-weighted T-30 availability aggregates
  models/status_calibration.py       what each designation predicts, per fold
  models/availability_bundles.py     Phase 3A3C ablation bundles
  models/nonlinear.py                gradient-boosted configs, compact fixed grid
  models/nonlinear_bundles.py        Phase 3A4 CORE and EXTENDED allowlists
  models/probability_calibration.py  leakage-safe chronological calibration
  models/kalshi_fees.py              date-effective Kalshi fee engine
  pipelines/build_dataset.py         Phase 1 CLI entry point
  pipelines/build_pregame_quotes.py  Phase 2 CLI entry point
  pipelines/build_availability_audit.py     Phase 3A3B0 CLI entry point
  pipelines/build_availability_salvage.py   Phase 3A3B1 CLI entry point
  pipelines/run_availability_capture.py     prospective capture CLI
  pipelines/build_availability_backfill.py  Phase 3A3B2 historical backfill CLI
  pipelines/build_availability_coverage.py  season/source coverage matrix
  pipelines/build_availability_features.py  T-30 availability feature builder
  pipelines/build_availability_model.py     Phase 3A3C CLI entry point
  pipelines/build_nonlinear_model.py        Phase 3A4 CLI entry point
  pipelines/build_market_edge_audit.py      Phase 4A0 CLI entry point
  pipelines/build_multi_anchor_market.py    Phase 4A1 multi-anchor backfill
  pipelines/build_anchor_models.py          Phase 4A1 anchor models + convergence
  pipelines/build_availability_event_study.py  Phase 4A1 news-reaction event study
```
