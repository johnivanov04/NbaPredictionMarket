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
# Is the system ready to capture tonight's slate?  PASS / WARN / FAIL
python -m nba_prediction_market.pipelines.check_capture_readiness
python -m nba_prediction_market.pipelines.check_capture_readiness --json

# Prove the pipeline behaves before trusting it live (uses the 2025-26 archive)
python -m nba_prediction_market.pipelines.replay_capture --limit 300
python -m nba_prediction_market.pipelines.replay_capture --limit 0   # whole season

# Historical backfill of official reports (also the restart path)
python -m nba_prediction_market.pipelines.build_availability_backfill \
    --start 2026-10-01 --end 2026-10-31
```

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

If a stale lock remains from a killed process, the readiness check reports it by
pid and names the file to remove.

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
* **The 2026-27 schedule is not yet ingested.** The readiness check reports this
  as a WARN; it must be resolved before capture begins.
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
