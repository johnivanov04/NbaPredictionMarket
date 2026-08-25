# 2026-27 prospective capture — operations

Research capture only. **Nothing here places an order**, and the package is
tested to contain no order-placement code path.

## What this collects, and why

Phase 4A1 established that Kalshi absorbs official NBA injury news over roughly
5-15 minutes rather than instantly, but that the adjustment (≈0.0056 for a
high-role status change) is smaller than the 1-cent spread. That is the only
imperfection this project has found, and one-minute candles are the coarsest
lens that can see it.

So the collector exists to make 2026-27 observable at finer resolution than
2025-26 was — not to trade it.

## Commands

```bash
# Ingest / refresh the forward schedule. Idempotent; run weekly and after any
# NBA Cup announcement, which is when the last 30 games get assigned.
python -m nba_prediction_market.pipelines.build_forward_schedule

# What is coming, and can we capture it?
python -m nba_prediction_market.pipelines.show_upcoming_games
python -m nba_prediction_market.pipelines.show_upcoming_games --hours 168 --json

# Is the system ready to capture tonight's slate?  PASS / WARN / FAIL
python -m nba_prediction_market.pipelines.check_capture_readiness
python -m nba_prediction_market.pipelines.check_capture_readiness --json

# Prove the pipeline behaves before trusting it live (uses the 2025-26 archive)
python -m nba_prediction_market.pipelines.replay_capture --limit 300
python -m nba_prediction_market.pipelines.replay_capture --limit 0   # whole season

# THE COLLECTOR. Long-running; the same command runs in preseason and in the
# regular season, with no special preseason mode.
python -m nba_prediction_market.pipelines.run_capture_collector

# Health of the collector that is already running (0 PASS, 1 WARN, 2 FAIL)
python -m nba_prediction_market.pipelines.show_capture_health

# After the slate
python -m nba_prediction_market.pipelines.summarise_capture_day --date 2026-10-03
```

`build_availability_backfill` is **not** the collector. It walks a range of
past dates and archives official reports; it never polls Kalshi, holds no lock,
writes no heartbeat, and exits immediately. It is the right tool for filling a
hole after the fact and the wrong one for a live slate.

### First preseason slate — 3 October 2026

```bash
python -m nba_prediction_market.pipelines.build_forward_schedule
python -m nba_prediction_market.pipelines.show_upcoming_games --hours 36
python -m nba_prediction_market.pipelines.check_capture_readiness   # must not FAIL
python -m nba_prediction_market.pipelines.run_capture_collector     # leave running
```

In a second terminal, while it runs:

```bash
python -m nba_prediction_market.pipelines.show_capture_health
```

### Stop and restart

Stop with `Ctrl-C` or `kill <pid>`. Both are handled: the collector finishes
the tick it is in, writes a final health snapshot, and releases its lock. To
restart, re-run the same command. It resumes rather than restarting work:

* archived report slots are never refetched, so `first_observed_at` keeps the
  moment a report was *first* seen rather than the moment of the last restart;
* status-change triggers are deduplicated on the report's own identity, so
  re-reading the same report cannot re-fire a sampling window;
* elevation is derived from observed changes, so a restarted process resumes at
  the 60-second baseline and re-elevates on the next real change.

The lock refuses to start a second collector alongside a live one, but takes
over a lock whose pid is dead — a crash should not need a human before capture
can resume. A lock we may not signal (`EPERM`) counts as live, not stale.

## Preseason and the official injury report

The NBA's Injury Report is a regular-season and playoff product. The league's
reporting policy does not cover preseason exhibitions, so **the normal case on
a preseason night is that no report is published at all.**

The collector still requests every slot. Expectation gates the *alarm*, never
the capture, so a preseason report that does get published is archived exactly
as a regular-season one would be. What changes is only how absence is read:

| situation | preseason | regular season |
| --- | --- | --- |
| no report observed | INFO `report_publication_not_expected` | INFO `no_report_observed_yet` |
| newest report >6h old | not reported | WARNING `availability_report_unusually_old` |
| canary URL failing | **CRITICAL** `report_source_canary_blocked` | **CRITICAL** (same) |
| no market observed | INFO `markets_not_yet_listed` | **CRITICAL** `no_market_observations` |
| market feed stale >10m | not reported | **CRITICAL** `market_feed_stale` |
| storage unwritable | **CRITICAL** | **CRITICAL** (same) |

Two properties matter here and are covered by tests:

* **Expected absence never masks a real outage.** The canary points at a URL
  known to exist from a *past* season, so it is unaffected by whether tonight's
  report exists. A blocked or throttled source stays CRITICAL in preseason.
* **Regular-season expectations are untouched.** The market checks key off the
  count of games that count toward research, which in the regular season equals
  the full slate — so the behaviour is identical to before this distinction
  existed. A test asserts that the unset default reproduces the old behaviour.

Market, schedule, and storage capture run identically in both phases. Kalshi is
unlikely to list preseason games; that is recorded as INFO and the collector
continues, because the point of preseason is to exercise everything else.

## Preseason soak test

Preseason exists to break the collector where it costs nothing. Run one full
slate unattended, then verify:

Most of this is one command — `summarise_capture_day --date 2026-10-03` reports
scheduled versus observed games, book rows at each sampling rate, fetch
failures, and any game never observed. The rest is checked by hand:

| check | how |
| --- | --- |
| every scheduled game observed | `summarise_capture_day` → `games_never_observed` must be empty |
| report snapshots archived *if published* | PDFs under `data/raw/availability/nba_official/2026/10/03/`. **Zero is the expected result in preseason** — the league does not publish then. Absence is only a fault if `official_reports_expected` is true. |
| raw market observations archived | `book_rows` > 0, and one mapped game observed per listed market |
| no unexpected restart | one heartbeat sequence in the lock, no gaps beyond the poll interval |
| no stale feeds | `check_capture_readiness` stayed PASS/WARN throughout |
| no duplicate triggers | replay the day's reports; trigger count must equal the distinct status changes |
| no post-anchor leakage | every stored quote timestamp <= its anchor |
| storage growth plausible | roughly 4 MB of PDFs per game day, plus market rows |
| health acceptable | no CRITICAL issues recorded |

**Preseason results feed nothing.** They are not part of discovery, not part of
validation, and never enter the model.

`check_capture_readiness` exits 0 on PASS or WARN and 1 on FAIL, so it can gate
a startup script.

## Daily operation

1. **Before the slate** — run the readiness check. Investigate any FAIL; a WARN
   is usually "markets not listed yet", which resolves itself.
2. **During the slate** — the collector polls official reports on the published
   grid and samples Kalshi at a 60-second baseline, dropping to 5 seconds for
   any game where a rotation player's status just changed.
3. **After the slate** — nothing. Raw artefacts are permanent; derived tables
   are rebuilt from them on demand.

## Storage layout

```
data/raw/availability/nba_official/YYYY/MM/DD/   official report PDFs + sidecars
data/raw/capture/markets/YYYY-MM-DD/             raw market responses
data/raw/capture/collector.lock                  pid + heartbeat
data/processed/                                  derived tables (rebuildable)
data/reports/                                    audits and health snapshots
```

Raw observations are **never overwritten**. A season costs roughly **1.2 GiB**
(≈0.6 GiB of report PDFs, ≈0.6 GiB of market observations).

## Restarting after a failure

The collector is restart-safe by construction:

* an already-archived report slot is never refetched;
* status-change triggers are deduplicated on the report's own identity, so a
  restart cannot re-fire a window that already ran;
* a restarted process re-establishes its baseline from the first report it
  sees, which yields no spurious "changes".

If a killed process leaves a stale lock, readiness reports it by pid as a
WARNING and starting the collector clears it automatically — a crash should not
need a human before capture resumes. The warning still fires, because a crash is
worth noticing even when it is handled.

## Failure severities

| severity | meaning | examples |
| --- | --- | --- |
| **CRITICAL** | coverage is being lost right now and cannot be recovered later | collector stopped, raw storage unwritable, market feed stale, report canary blocked, game near its anchor with no market identity |
| **WARNING** | something is degraded but capture continues | a report failed to parse, one player unresolved, an individual fetch failed, unusually old report |
| **INFO** | normal, worth noting | no upcoming games, markets not yet listed |

Failures are written as rows, never skipped. A dataset with unrecorded holes is
worse than no dataset.

## Known limitations

* **No Kalshi credentials.** The WebSocket (`orderbook_delta`, `market_ticker`)
  requires API-key authentication even for public market data, so the collector
  polls the public REST orderbook instead. Supplying `KALSHI_API_KEY_ID` and
  `KALSHI_PRIVATE_KEY` would allow a push-based upgrade.
* **The last 30 regular-season games are unassigned.** The league publishes 80
  of each team's 82; the rest depend on Emirates NBA Cup results. The schedule
  reports this as `incomplete_by_design`, and a refresh after the Cup will add
  them without disturbing captured data.
* **Replay uses source timestamps** as a stand-in for observation time, because
  the 2025-26 archive predates the first-observed field. Live capture records
  both clocks.

## Research protocol (fixed)

| period | dates | what happens |
| --- | --- | --- |
| **Discovery** | season start → 2026-12-31 | capture; hypotheses may be researched |
| **Freeze** | 2027-01-01 | any hypothesis is written to an immutable spec: anchor, threshold, inputs, execution assumptions, fees, no-trade rules |
| **Validation** | 2027-01-01 → season end | score frozen hypotheses only |

The periods share no games. **If discovery yields nothing credible, freeze "no
strategy"** — the validation period is not an obligation to use.
