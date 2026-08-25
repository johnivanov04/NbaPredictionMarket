"""Collector health, and failures that are recorded rather than swallowed.

An unattended collector that fails quietly is worse than one that does not run:
it produces a dataset with holes nobody knows about, and holes are only
discoverable months later when the research depends on them.

So every failure becomes a row. Severity says how loudly to complain, never
whether to record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

CRITICAL: str = "CRITICAL"
WARNING: str = "WARNING"
INFO: str = "INFO"
SEVERITIES: tuple[str, ...] = (CRITICAL, WARNING, INFO)

#: Beyond this with no market observation, the feed counts as stale.
STALE_MARKET_SECONDS: float = 600.0
#: Beyond this with no collector heartbeat, the process counts as stopped.
HEARTBEAT_TIMEOUT_SECONDS: float = 300.0
#: Inside this window before an anchor a game must have a resolved market.
MARKET_IDENTITY_REQUIRED_BEFORE_SECONDS: float = 7200.0


@dataclass(frozen=True)
class HealthIssue:
    """One classified problem."""

    severity: str
    code: str
    detail: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "detail": self.detail,
            "context": self.context,
        }


@dataclass
class CollectorState:
    """What the collector knows about itself right now."""

    now_utc: datetime
    heartbeat_at_utc: datetime | None = None
    last_market_observation_utc: datetime | None = None
    last_report_observation_utc: datetime | None = None
    latest_report_source_timestamp_utc: datetime | None = None
    raw_storage_writable: bool = True
    canary_reachable: bool = True
    upcoming_games: int = 0
    games_with_market_identity: int = 0

    #: Of ``upcoming_games``, how many are regular-season games whose data the
    #: research dataset actually depends on. ``None`` means "same as
    #: ``upcoming_games``", which is the regular-season case and keeps the
    #: original behaviour byte for byte.
    #:
    #: Preseason exists to shake the collector out operationally and is never
    #: modelled, so a preseason game with no market loses nothing recoverable.
    #: Missing a *regular-season* market is unrecoverable, and stays CRITICAL.
    research_games: int | None = None

    #: Whether the NBA is expected to publish its official Injury Report for
    #: the current window at all. The report is a regular-season and playoff
    #: product; the league does not run it for preseason exhibitions.
    #:
    #: This gates *alarms only*. The collector still requests every slot and
    #: still archives anything it finds, so a preseason report that does get
    #: published is captured exactly as a regular-season one would be. What
    #: changes is that its absence stops being reported as a fault.
    report_publication_expected: bool = True

    #: Regular-season games only -- see ``research_games``.
    games_missing_identity_near_anchor: list[Any] = field(default_factory=list)
    failed_fetches: int = 0
    parse_failures: int = 0
    identity_failures: int = 0
    unresolved_players: int = 0
    disk_free_bytes: int | None = None
    model_artifact_available: bool = True


def assess(state: CollectorState) -> list[HealthIssue]:
    """Classify everything wrong, most severe first.

    The three CRITICAL conditions all mean the same thing operationally: the
    dataset is losing coverage right now and no later repair can recover it.
    """
    issues: list[HealthIssue] = []

    if state.heartbeat_at_utc is None:
        issues.append(HealthIssue(
            CRITICAL, "collector_not_running", "no heartbeat has been recorded"
        ))
    else:
        age = (state.now_utc - state.heartbeat_at_utc).total_seconds()
        if age > HEARTBEAT_TIMEOUT_SECONDS:
            issues.append(HealthIssue(
                CRITICAL, "collector_stopped",
                f"last heartbeat {age:.0f}s ago",
                {"seconds": age},
            ))

    if not state.raw_storage_writable:
        issues.append(HealthIssue(
            CRITICAL, "raw_storage_unwritable",
            "raw observations cannot be persisted; capture is being lost",
        ))

    if not state.canary_reachable:
        issues.append(HealthIssue(
            CRITICAL, "report_source_canary_blocked",
            "a URL known to exist is failing, so 403 responses cannot be read "
            "as 'not published' and reports may be silently missed",
        ))

    market_expected = (
        state.upcoming_games if state.research_games is None
        else state.research_games
    )

    if state.last_market_observation_utc is None:
        if market_expected:
            issues.append(HealthIssue(
                CRITICAL, "no_market_observations",
                "games are upcoming but no market has been observed",
            ))
    else:
        age = (state.now_utc - state.last_market_observation_utc).total_seconds()
        if age > STALE_MARKET_SECONDS and market_expected:
            issues.append(HealthIssue(
                CRITICAL, "market_feed_stale",
                f"no market observation for {age:.0f}s",
                {"seconds": age},
            ))

    if state.games_missing_identity_near_anchor:
        issues.append(HealthIssue(
            CRITICAL, "missing_market_identity_near_anchor",
            "a game is approaching its anchor with no resolved Kalshi market",
            {"games": list(state.games_missing_identity_near_anchor)},
        ))

    if not state.model_artifact_available:
        issues.append(HealthIssue(
            WARNING, "model_artifact_missing",
            "frozen model artefact not found; anchor snapshots cannot be made",
        ))
    if state.parse_failures:
        issues.append(HealthIssue(
            WARNING, "report_parse_failures",
            f"{state.parse_failures} report(s) failed to parse",
            {"count": state.parse_failures},
        ))
    if state.failed_fetches:
        issues.append(HealthIssue(
            WARNING, "fetch_failures",
            f"{state.failed_fetches} fetch(es) failed",
            {"count": state.failed_fetches},
        ))
    if state.identity_failures:
        issues.append(HealthIssue(
            WARNING, "identity_failures",
            f"{state.identity_failures} game(s) could not be mapped to a market",
            {"count": state.identity_failures},
        ))
    if state.unresolved_players:
        issues.append(HealthIssue(
            WARNING, "unresolved_players",
            f"{state.unresolved_players} designated player(s) unresolved",
            {"count": state.unresolved_players},
        ))
    if (
        state.latest_report_source_timestamp_utc is not None
        and state.report_publication_expected
    ):
        age = (
            state.now_utc - state.latest_report_source_timestamp_utc
        ).total_seconds()
        if age > 6 * 3600 and state.upcoming_games:
            issues.append(HealthIssue(
                WARNING, "availability_report_unusually_old",
                f"newest official report is {age / 3600:.1f}h old",
                {"hours": age / 3600},
            ))
    if state.disk_free_bytes is not None and state.disk_free_bytes < 5 * 2**30:
        issues.append(HealthIssue(
            WARNING, "low_disk_space",
            f"{state.disk_free_bytes / 2**30:.1f} GiB free",
            {"free_bytes": state.disk_free_bytes},
        ))

    if state.upcoming_games == 0:
        issues.append(HealthIssue(
            INFO, "no_upcoming_games", "no NBA games scheduled in the window"
        ))
    if state.last_report_observation_utc is None:
        issues.append(
            HealthIssue(
                INFO, "no_report_observed_yet", "no official report observed yet"
            )
            if state.report_publication_expected
            else HealthIssue(
                INFO, "report_publication_not_expected",
                "no official report observed, and none is expected: the NBA "
                "does not publish its Injury Report for preseason games. The "
                "canary still runs, so genuine blocking is still detected.",
            )
        )
    if state.upcoming_games and (
        state.games_with_market_identity < state.upcoming_games
    ):
        issues.append(HealthIssue(
            INFO, "markets_not_yet_listed",
            f"{state.upcoming_games - state.games_with_market_identity} "
            "upcoming game(s) have no market listed yet",
        ))

    order = {CRITICAL: 0, WARNING: 1, INFO: 2}
    return sorted(issues, key=lambda i: order[i.severity])


def overall_status(issues: list[HealthIssue]) -> str:
    """PASS, WARN or FAIL, decided by the worst issue present."""
    severities = {issue.severity for issue in issues}
    if CRITICAL in severities:
        return "FAIL"
    if WARNING in severities:
        return "WARN"
    return "PASS"


def summarise(state: CollectorState) -> dict[str, Any]:
    issues = assess(state)
    return {
        "status": overall_status(issues),
        "checked_at_utc": state.now_utc.isoformat(),
        "counts": {
            severity: sum(1 for i in issues if i.severity == severity)
            for severity in SEVERITIES
        },
        "issues": [issue.to_dict() for issue in issues],
        "observed": {
            "upcoming_games": state.upcoming_games,
            "games_with_market_identity": state.games_with_market_identity,
            "last_market_observation_utc": (
                state.last_market_observation_utc.isoformat()
                if state.last_market_observation_utc else None
            ),
            "last_report_observation_utc": (
                state.last_report_observation_utc.isoformat()
                if state.last_report_observation_utc else None
            ),
            "collector_lag_seconds": (
                (state.now_utc - state.heartbeat_at_utc).total_seconds()
                if state.heartbeat_at_utc else None
            ),
        },
    }


def storage_estimate(
    *,
    games_per_season: int = 1230,
    reports_per_day: int = 48,
    game_days: int = 170,
    report_bytes: int = 80_000,
    market_row_bytes: int = 220,
    baseline_samples_per_game: int = 360,
    event_samples_per_game: int = 720,
) -> dict[str, Any]:
    """Rough disk requirement for one captured season.

    Deliberately generous: the point is to know the order of magnitude before
    committing to a season of unattended capture, not to be exact.
    """
    reports = reports_per_day * game_days
    report_total = reports * report_bytes
    market_rows = games_per_season * 2 * (
        baseline_samples_per_game + event_samples_per_game
    )
    market_total = market_rows * market_row_bytes
    return {
        "official_report_pdfs": {
            "count": reports,
            "bytes": report_total,
            "gib": round(report_total / 2**30, 2),
        },
        "market_observations": {
            "rows": market_rows,
            "bytes": market_total,
            "gib": round(market_total / 2**30, 2),
        },
        "total_gib": round((report_total + market_total) / 2**30, 2),
        "note": (
            "raw artefacts are permanent research records and are never "
            "overwritten; derived tables can always be rebuilt"
        ),
    }
