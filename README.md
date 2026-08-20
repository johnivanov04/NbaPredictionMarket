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

### Do not read a training window into this

The point of 20 seasons is to make the history *available*, not to assert it is
all useful. Whether 3, 5, 8, 10, 15, or all seasons help is an open question to
be settled by chronological validation **before** the 2025-26 holdout — never by
2025-26 performance.

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
  pipelines/build_dataset.py         Phase 1 CLI entry point
  pipelines/build_pregame_quotes.py  Phase 2 CLI entry point
```
