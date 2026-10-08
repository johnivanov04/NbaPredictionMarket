"""The long-running prospective capture collector.

This is the process that actually runs on a game night. Everything it needs
was built and unit-tested in Phase 4A2 -- the observation timing model, the
orderbook parser, the event trigger, the health assessment -- but those are
components, and until now nothing drove them. ``run_availability_capture`` is
a *one-shot* archiver of official reports that never touches Kalshi, and
``build_availability_backfill`` walks historical dates. Neither is a collector.

What this adds is only the process: a loop, a lock, a heartbeat, and the
wiring between parts that already exist. No capture logic is reimplemented
here -- report fetching goes through ``AvailabilityRunner``, book parsing
through ``parse_book``, transitions through ``diff_states``, sampling rate
through ``CaptureScheduler``, and status through ``health.assess``.

**This process cannot place an order.** It uses only public read endpoints
(``/events``, ``/markets``, ``/orderbook``), holds no credentials, and never
imports an order path. ``tests/unit/test_capture_collector.py`` asserts that
mechanically rather than trusting this paragraph.

Restart behaviour is deliberate rather than incidental:

* An archived report slot is never refetched, so ``first_observed_at`` keeps
  the moment we *first* saw a report rather than the moment we last restarted.
* The scheduler deduplicates triggers on report identity, so replaying the
  same report after a restart cannot re-fire a sampling window.
* Elevation state is derived from observed changes, so a restarted process
  resumes at the baseline rate and re-elevates on the next real change.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import Any

import httpx

from nba_prediction_market.availability.capture_schedule import plan_captures
from nba_prediction_market.availability.nba_official import ReportArchive
from nba_prediction_market.availability.nba_report_parser import parse_report_pdf
from nba_prediction_market.availability.runner import AvailabilityRunner
from nba_prediction_market.availability.snapshot_store import SnapshotStore
from nba_prediction_market.capture.event_trigger import (
    CaptureScheduler,
    diff_states,
)
from nba_prediction_market.capture.health import (
    CollectorState,
    summarise,
)
from nba_prediction_market.capture.kalshi_live import PUBLIC_REST_BASE, parse_book
from nba_prediction_market.capture.market_identity import MATCHED, map_games
from nba_prediction_market.capture.observation import (
    ObservationTiming,
    RawObservation,
    content_hash,
)
from nba_prediction_market.capture.schedule import reports_expected, upcoming
from nba_prediction_market.config import (
    ConfigError,
    Settings,
    load_settings,
)
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.pipelines.build_forward_schedule import load_stored
from nba_prediction_market.pipelines.run_availability_capture import (
    CAPTURE_SOURCES,
    PacedFetcher,
    canary_is_reachable,
)

logger = logging.getLogger(__name__)

#: Asserted by the test suite. This collector is research-only by construction.
PLACES_ORDERS: bool = False

#: How often the loop wakes. Must divide the event interval, or a 5-second
#: elevated window would be sampled at whatever coarser rate the loop happens
#: to run at and the elevation would be real but ineffective.
TICK_SECONDS: float = 1.0

#: Heartbeat cadence. Comfortably inside the 300s the health module treats as
#: "collector stopped", so a brief network stall is not read as a dead process.
HEARTBEAT_SECONDS: float = 30.0

#: How often market identity is re-resolved. Kalshi lists games progressively,
#: so a game with no ticker at start of shift can acquire one mid-shift.
IDENTITY_REFRESH_SECONDS: float = 600.0

#: How often the referee-assignment page is re-read. The league posts around
#: 09:00 ET and occasionally reassigns during the day, so a half-hourly read
#: catches changes at a cost of ~48 requests a day. Strictly additive: a
#: failure here never touches report or market capture.
REFEREE_REFRESH_SECONDS: float = 1800.0

REQUEST_TIMEOUT_SECONDS: float = 30.0


class LockHeld(RuntimeError):
    """Another live collector already owns the lock."""


@dataclass
class CollectorLock:
    """A pid-stamped lock with a heartbeat, released on any ordinary exit.

    A lock whose pid is dead is *stale*, not held: the machine rebooted or the
    process was killed, and refusing to start then would turn a crash into an
    outage that needs a human. A lock whose pid is alive is honoured, because
    two collectors writing the same archive is how duplicate and interleaved
    observations get created.
    """

    path: Path
    mode: str = "live"
    _acquired: bool = field(default=False, init=False)

    def acquire(self, now: datetime | None = None) -> CollectorLock:
        now = now or utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            holder = self._holder()
            if holder is not None:
                raise LockHeld(
                    f"collector already running as pid {holder}; "
                    f"stop it before starting another"
                )
            logger.warning("clearing stale lock from a dead process")
        self._write(now, now)
        self._acquired = True
        return self

    def _holder(self) -> int | None:
        """The live pid holding this lock, or None if it is stale."""
        try:
            pid = int(json.loads(self.path.read_text()).get("pid", -1))
        except Exception:
            return None
        if pid <= 0 or pid == os.getpid():
            return None
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except PermissionError:
            # The process exists, we simply may not signal it. Reading that as
            # "stale" would let a second collector start alongside a live one.
            return pid
        except OSError:
            return None
        return pid

    def _write(self, started: datetime, heartbeat: datetime) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "mode": self.mode,
                    "started_at_utc": started.isoformat(),
                    "heartbeat_at_utc": heartbeat.isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def beat(self, now: datetime) -> None:
        if not self._acquired:
            return
        try:
            started = json.loads(self.path.read_text())["started_at_utc"]
        except Exception:
            started = now.isoformat()
        self._write(datetime.fromisoformat(started), now)

    def release(self) -> None:
        if self._acquired and self.path.is_file():
            self.path.unlink()
        self._acquired = False

    def __enter__(self) -> CollectorLock:
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


def fetch_orderbooks(
    client: httpx.Client, event_ticker: str, *, now: datetime
) -> tuple[list[dict[str, Any]], list[RawObservation]]:
    """Every market under one event, with its book and the timing of the fetch.

    Read-only endpoints only. Errors are returned as failed observations rather
    than raised: one unlisted market must not end a night's capture.
    """
    rows: list[dict[str, Any]] = []
    observations: list[RawObservation] = []

    url = f"{PUBLIC_REST_BASE}/markets"
    started = utc_now()
    try:
        response = client.get(
            url, params={"event_ticker": event_ticker, "limit": 100},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        received = utc_now()
        payload = response.json() if response.status_code == 200 else {}
        observations.append(
            RawObservation(
                source="kalshi_markets", url=url, http_status=response.status_code,
                timing=ObservationTiming(started, received, received),
                content_sha256=content_hash(response.content),
                content_bytes=len(response.content),
                request={"event_ticker": event_ticker},
            )
        )
    except httpx.HTTPError as exc:
        received = utc_now()
        observations.append(
            RawObservation(
                source="kalshi_markets", url=url, http_status=0,
                timing=ObservationTiming(started, received, received),
                content_sha256=content_hash(b""), content_bytes=0,
                request={"event_ticker": event_ticker}, error=str(exc),
            )
        )
        return rows, observations

    for market in payload.get("markets", []):
        ticker = str(market.get("ticker", ""))
        if not ticker:
            continue
        book_url = f"{PUBLIC_REST_BASE}/markets/{ticker}/orderbook"
        started = utc_now()
        try:
            book_response = client.get(book_url, timeout=REQUEST_TIMEOUT_SECONDS)
            received = utc_now()
            observations.append(
                RawObservation(
                    source="kalshi_orderbook", url=book_url,
                    http_status=book_response.status_code,
                    timing=ObservationTiming(started, received, received),
                    content_sha256=content_hash(book_response.content),
                    content_bytes=len(book_response.content),
                    request={"ticker": ticker},
                )
            )
            if book_response.status_code != 200:
                continue
            snapshot = parse_book(
                ticker, book_response.json(), received, market_payload=market
            )
            row = snapshot.to_row()
            row["event_ticker"] = event_ticker
            row["observed_at_utc"] = received.isoformat()
            rows.append(row)
        except httpx.HTTPError as exc:
            received = utc_now()
            observations.append(
                RawObservation(
                    source="kalshi_orderbook", url=book_url, http_status=0,
                    timing=ObservationTiming(started, received, received),
                    content_sha256=content_hash(b""), content_bytes=0,
                    request={"ticker": ticker}, error=str(exc),
                )
            )
    return rows, observations


def _task_key(task: Any) -> tuple[Any, str, str]:
    """Identity of one planned capture, stable across ticks."""
    return (task.game_id, task.source, task.capture_at_utc.isoformat())


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def report_states(
    archive: ReportArchive, slots: list[Any]
) -> tuple[dict[tuple[Any, str, Any], str], dict[Any, str], datetime | None]:
    """Player statuses from the newest archived slot among ``slots``.

    Keyed by ``(matchup, team, player_name)``. The collector deliberately does
    not resolve names to player ids here: the only decision it makes from this
    is the sampling rate, and resolving identity under time pressure is exactly
    where a silent fuzzy match would get made. The research join happens later,
    offline, against the alias table.
    """
    archived = [s for s in slots if archive.has(s)]
    if not archived:
        return {}, {}, None
    slot = max(archived, key=lambda s: s.report_timestamp_utc)
    try:
        parsed = parse_report_pdf(archive.pdf_path(slot))
    except Exception as exc:
        logger.warning("could not parse %s: %s", slot.filename, exc)
        return {}, {}, None
    states = {
        (entry.matchup, entry.team, entry.player_name): entry.status_normalized
        for entry in parsed.entries
    }
    names = {entry.player_name: entry.player_name for entry in parsed.entries}
    return states, names, parsed.report_timestamp_utc


@dataclass
class Collector:
    """One shift's mutable state. Everything durable is on disk."""

    settings: Settings
    horizon_hours: float = 36.0
    scheduler: CaptureScheduler = field(default_factory=CaptureScheduler)
    previous_states: dict[tuple[Any, str, Any], str] | None = None
    last_polled: dict[Any, datetime] = field(default_factory=dict)
    last_identity_refresh: datetime | None = None
    last_referee_refresh: datetime | None = None
    #: Planned captures already attempted this process, keyed by
    #: ``(game_id, source, capture_at_utc)``. ``plan_captures`` schedules each
    #: capture for a specific moment; without this, "due" means "its moment has
    #: passed", so every task re-ran on every one-second tick and the same
    #: report URL was requested once a second for as long as the slate lasted.
    attempted_captures: set[tuple[Any, str, str]] = field(default_factory=set)
    capture_referees: bool = True
    referee_assignments: int = 0
    referee_failures: int = 0
    mappings: list[Any] = field(default_factory=list)
    last_market_observation: datetime | None = None
    last_report_observation: datetime | None = None
    latest_report_source_ts: datetime | None = None
    canary_ok: bool = True
    failed_fetches: int = 0
    parse_failures: int = 0
    market_rows: int = 0
    report_captures: int = 0
    changes_detected: int = 0
    triggers_fired: int = 0

    def __post_init__(self) -> None:
        root = self.settings.paths.root
        self.archive = ReportArchive(root / "raw" / "availability" / "nba_official")
        self.store = SnapshotStore(root / "raw" / "availability" / "snapshots")
        self.capture_root = root / "raw" / "capture"

    # -- one pass ---------------------------------------------------------

    def tick(self, client: httpx.Client, now: datetime) -> None:
        games = upcoming(load_stored(self.settings), now, self.horizon_hours)
        self._refresh_identity(games, now)
        self._capture_reports(client, games, now)
        self._poll_markets(client, games, now)
        self._capture_referees(now)
        self.scheduler.prune(now)

    def _capture_referees(self, now: datetime) -> None:
        """Optional, additive, and never load-bearing.

        Wrapped whole: a missing crew costs one experimental feature, while an
        exception escaping here would cost the slate's report and market
        coverage, which is unrecoverable. That asymmetry is why this swallows
        everything and only counts the failure.
        """
        if not self.capture_referees:
            return
        due = (
            self.last_referee_refresh is None
            or (now - self.last_referee_refresh).total_seconds()
            >= REFEREE_REFRESH_SECONDS
        )
        if not due:
            return
        self.last_referee_refresh = now
        try:
            from nba_prediction_market.pipelines.capture_referee_assignments import (
                capture_once,
            )

            result = capture_once(self.settings, now=now)
            if result.get("captured"):
                self.referee_assignments = result.get("assignments", 0)
            else:
                self.referee_failures += 1
                logger.warning("referee capture failed: %s", result.get("error"))
        except Exception as exc:
            self.referee_failures += 1
            logger.warning("referee capture raised %s", type(exc).__name__)

    def _refresh_identity(self, games: list[Any], now: datetime) -> None:
        due = (
            self.last_identity_refresh is None
            or (now - self.last_identity_refresh).total_seconds()
            >= IDENTITY_REFRESH_SECONDS
        )
        if not due:
            return
        self.last_identity_refresh = now
        from nba_prediction_market.pipelines.show_upcoming_games import (
            fetch_event_tickers,
        )

        listed = fetch_event_tickers()
        if not listed:
            logger.warning("no Kalshi events listed; keeping previous identities")
            return
        self.mappings = map_games(games, listed, now=now)

    def _capture_reports(
        self, client: httpx.Client, games: list[Any], now: datetime
    ) -> None:
        """Archive any report slot now due, then look for status transitions.

        Runs for preseason too. The league is not expected to publish then, but
        "expected" is not "guaranteed", and a report that does appear is worth
        far more than the cost of a 404.
        """
        if not games:
            return
        planned = plan_captures(
            [
                {"game_id": g.source_game_id, "scheduled_tipoff_utc": g.tipoff_utc}
                for g in games
            ],
            CAPTURE_SOURCES,
        )
        # Each planned capture is attempted once. The nine offsets per game are
        # the intended retry schedule, and the runner itself retries a transient
        # error three times with backoff, so re-running a task every tick adds
        # no coverage -- it only spends the rate-limit budget, and a throttled
        # source answers 403, the same code it uses for "not published".
        due = [
            t for t in planned
            if t.capture_at_utc <= now and _task_key(t) not in self.attempted_captures
        ]
        if not due:
            return
        for task in due:
            self.attempted_captures.add(_task_key(task))

        runner = AvailabilityRunner(
            self.archive, self.store, fetch=PacedFetcher(client)
        )
        results = runner.run(due, now=now)
        # ``CaptureResult.outcome``, not ``.status``. The runner has only ever
        # exposed ``outcome`` -- ``anchor_health`` in the same module reads it --
        # and this caller drifted. Nothing caught it because every earlier smoke
        # test ran with no game inside the capture horizon, so ``due`` was empty
        # and the code below was never reached.
        captured = [r for r in results if r.outcome == "captured"]
        self.report_captures += len(captured)
        # 403 means "not published" and is counted as ``unavailable``, not
        # ``failed``, so an unpublished preseason report does not look like a
        # fetch failure.
        self.failed_fetches += runner.stats.failed
        if captured:
            self.last_report_observation = now

        unavailable = [r for r in results if r.outcome == "source_unavailable"]
        if unavailable:
            self.canary_ok = canary_is_reachable(client)

        slots = [r.slot for r in results if r.slot is not None]
        states, names, source_ts = report_states(self.archive, slots)
        if not states:
            return
        self.latest_report_source_ts = source_ts
        changes = diff_states(
            self.previous_states, states,
            detected_at_utc=now, source_report_timestamp_utc=source_ts,
            roles={}, player_names=names,
        )
        self.changes_detected += len(changes)
        fired = [c for c in changes if self.scheduler.register(c)]
        self.triggers_fired += len(fired)
        if changes:
            _append_jsonl(
                self.capture_root / "events" /
                f"{now.date().isoformat()}" / "status_changes.jsonl",
                [c.to_dict() for c in changes],
            )
        self.previous_states = states

    def _poll_markets(
        self, client: httpx.Client, games: list[Any], now: datetime
    ) -> None:
        by_game = {
            m.source_game_id: m for m in self.mappings if m.status == MATCHED
        }
        for game in games:
            mapping = by_game.get(game.source_game_id)
            if mapping is None or mapping.event_ticker is None:
                continue
            interval = self.scheduler.interval_for(game.source_game_id, now)
            last = self.last_polled.get(game.source_game_id)
            if last is not None and (now - last).total_seconds() < interval:
                continue
            rows, observations = fetch_orderbooks(
                client, mapping.event_ticker, now=now
            )
            self.last_polled[game.source_game_id] = now
            day = now.date().isoformat()
            for row in rows:
                row["source_game_id"] = game.source_game_id
                row["sampling_interval_seconds"] = interval
            _append_jsonl(
                self.capture_root / "markets" / day / "books.jsonl", rows
            )
            _append_jsonl(
                self.capture_root / "markets" / day / "observations.jsonl",
                [o.to_dict() for o in observations],
            )
            self.market_rows += len(rows)
            self.failed_fetches += sum(1 for o in observations if not o.succeeded)
            if rows:
                self.last_market_observation = now

    # -- health -----------------------------------------------------------

    def state(self, now: datetime) -> CollectorState:
        games = upcoming(load_stored(self.settings), now, self.horizon_hours)
        research = [g for g in games if g.counts_toward_research]
        matched = {
            m.source_game_id for m in self.mappings if m.status == MATCHED
        }
        research_ids = {g.source_game_id for g in research}
        missing_near = [
            m.source_game_id for m in self.mappings
            if m.status != MATCHED
            and m.source_game_id in research_ids
            and m.severity(now) == "CRITICAL"
        ]
        try:
            usage = os.statvfs(self.settings.paths.root)
            free = usage.f_bavail * usage.f_frsize
        except OSError:
            free = None
        return CollectorState(
            now_utc=now,
            heartbeat_at_utc=now,
            last_market_observation_utc=self.last_market_observation,
            last_report_observation_utc=self.last_report_observation,
            latest_report_source_timestamp_utc=self.latest_report_source_ts,
            raw_storage_writable=os.access(self.settings.paths.root, os.W_OK),
            canary_reachable=self.canary_ok,
            upcoming_games=len(games),
            research_games=len(research),
            report_publication_expected=reports_expected(games),
            games_with_market_identity=len(matched),
            games_missing_identity_near_anchor=missing_near,
            failed_fetches=self.failed_fetches,
            parse_failures=self.parse_failures,
            disk_free_bytes=free,
        )

    def write_health(self, now: datetime) -> dict[str, Any]:
        payload = summarise(self.state(now))
        payload["counters"] = {
            "report_captures": self.report_captures,
            "market_rows": self.market_rows,
            "status_changes_detected": self.changes_detected,
            "sampling_triggers_fired": self.triggers_fired,
            "games_at_event_rate": len(self.scheduler.elevated_games(now)),
            "referee_assignments": self.referee_assignments,
            "referee_capture_failures": self.referee_failures,
        }
        path = self.settings.paths.reports / "capture_health.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload


def run_pipeline(
    *,
    settings: Settings | None = None,
    horizon_hours: float = 36.0,
    duration_seconds: float | None = None,
    once: bool = False,
    tick_seconds: float = TICK_SECONDS,
    capture_referees: bool = True,
) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    if not load_stored(settings):
        raise ConfigError(
            "no forward schedule stored; run build_forward_schedule first"
        )

    lock = CollectorLock(settings.paths.root / "raw" / "capture" / "collector.lock")
    collector = Collector(
        settings, horizon_hours=horizon_hours, capture_referees=capture_referees
    )
    stopping = {"now": False}

    def _stop(signum: int, _frame: FrameType | None) -> None:
        logger.info("signal %s received; finishing this tick and exiting", signum)
        stopping["now"] = True

    previous_handlers: list[tuple[int, Any]] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Not installable off the main thread, e.g. under a test runner. That
        # only costs the graceful-stop path; the lock still releases.
        with contextlib.suppress(ValueError):
            previous_handlers.append((sig, signal.signal(sig, _stop)))

    started = utc_now()
    deadline = (
        started + timedelta(seconds=duration_seconds)
        if duration_seconds is not None else None
    )
    ticks = 0
    last_beat = started
    health: dict[str, Any] = {}

    with lock, httpx.Client(follow_redirects=True) as client:
        while True:
            now = utc_now()
            collector.tick(client, now)
            ticks += 1
            if (now - last_beat).total_seconds() >= HEARTBEAT_SECONDS or ticks == 1:
                lock.beat(now)
                health = collector.write_health(now)
                last_beat = now
            if once or stopping["now"]:
                break
            if deadline is not None and utc_now() >= deadline:
                break
            time.sleep(tick_seconds)
        health = collector.write_health(utc_now())

    for sig, handler in previous_handlers:
        signal.signal(sig, handler)

    report = {
        "started_at_utc": started.isoformat(),
        "stopped_at_utc": utc_now().isoformat(),
        "ticks": ticks,
        "horizon_hours": horizon_hours,
        "places_orders": PLACES_ORDERS,
        "report_captures": collector.report_captures,
        "market_rows": collector.market_rows,
        "status_changes_detected": collector.changes_detected,
        "sampling_triggers_fired": collector.triggers_fired,
        "failed_fetches": collector.failed_fetches,
        "referee_assignments": collector.referee_assignments,
        "referee_capture_failures": collector.referee_failures,
        "health": health,
    }
    path = settings.paths.reports / "capture_collector_run.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"] = [str(path)]
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the long-running prospective capture collector."
    )
    parser.add_argument(
        "--horizon-hours", type=float, default=36.0,
        help="Capture games tipping inside this many hours (default: 36).",
    )
    parser.add_argument(
        "--duration-seconds", type=float, default=None,
        help="Exit after this long. Omit to run until stopped.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single tick and exit. For smoke tests.",
    )
    parser.add_argument(
        "--no-referees", action="store_true",
        help="Skip referee-assignment capture. It is optional either way.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s"
    )
    try:
        report = run_pipeline(
            horizon_hours=args.horizon_hours,
            duration_seconds=args.duration_seconds,
            once=args.once,
            capture_referees=not args.no_referees,
        )
    except LockHeld as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    health = report.get("health", {})
    print(f"Ticks            : {report['ticks']}")
    print(f"Report captures  : {report['report_captures']}")
    print(f"Market rows      : {report['market_rows']}")
    print(f"Status changes   : {report['status_changes_detected']}")
    print(f"Triggers fired   : {report['sampling_triggers_fired']}")
    print(f"Health           : {health.get('status', 'unknown')}")
    for issue in health.get("issues", []):
        print(f"  [{issue['severity']:8}] {issue['code']}: {issue['detail']}")
    for path in report["written_files"]:
        print(f"\nWrote {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
