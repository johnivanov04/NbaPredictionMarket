"""Audited capability matrix for player-availability sources.

The governing rule for this whole package: our prediction anchor is
``prediction_ts = scheduled tipoff - 30 minutes``, so an observation is usable
only if ``observed_at <= prediction_ts``. A status learned one second later is
future information.

That makes **timestamp precision** the deciding property of a source, not
whether it has "historical data". A feed can hold years of history and still be
useless if it cannot say *when* a status was known.

Every entry below was verified empirically on 2026-08-20 unless marked
``needs_credentials``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

# Temporal precision, in descending order of usefulness.
PRECISION_INTRADAY_TIMESTAMP: Final = "intraday_timestamp"
PRECISION_DATE_ONLY: Final = "date_only"
PRECISION_NONE: Final = "no_timestamp"
PRECISION_UNKNOWN: Final = "unknown_needs_credentials"

# Whether a source can reconstruct what was known at a past T-30.
ASOF_SAFE: Final = "safe"
ASOF_UNSAFE: Final = "unsafe"
ASOF_UNKNOWN: Final = "unknown"


@dataclass(frozen=True)
class AvailabilitySource:
    """One audited availability source."""

    name: str
    endpoint: str
    historical_coverage: str
    timestamp_precision: str
    has_player_status: bool
    has_projected_lineup: bool
    has_confirmed_lineup: bool
    historical_asof_safety: str
    prospective_usefulness: str
    cost_access: str
    recommendation: str
    evidence: str
    verified: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.name,
            "endpoint": self.endpoint,
            "historical_coverage": self.historical_coverage,
            "intraday_timestamp_precision": self.timestamp_precision,
            "player_status": self.has_player_status,
            "projected_lineup": self.has_projected_lineup,
            "confirmed_lineup": self.has_confirmed_lineup,
            "historical_as_of_safety": self.historical_asof_safety,
            "prospective_usefulness": self.prospective_usefulness,
            "cost_access": self.cost_access,
            "recommendation": self.recommendation,
            "evidence": self.evidence,
            "empirically_verified": self.verified,
        }


SOURCE_MATRIX: Final[tuple[AvailabilitySource, ...]] = (
    AvailabilitySource(
        name="nba_official_injury_report",
        endpoint=(
            "https://ak-static.cms.nba.com/referee/injury/"
            "Injury-Report_{YYYY-MM-DD}_{hh}_{mm}{AM|PM}.pdf"
        ),
        historical_coverage="rolling ~8 months only (see evidence)",
        timestamp_precision=PRECISION_INTRADAY_TIMESTAMP,
        has_player_status=True,
        has_projected_lineup=False,
        has_confirmed_lineup=False,
        historical_asof_safety=ASOF_UNSAFE,
        prospective_usefulness="excellent -- the reference source going forward",
        cost_access="free, public CDN, no key",
        recommendation=(
            "Primary PROSPECTIVE source. Cannot supply multi-season history: the "
            "CDN only retains a rolling window."
        ),
        evidence=(
            "Reports are published every 30 minutes around the clock. Each slot "
            "returns HTTP 200 with a Last-Modified matching the filename time "
            "exactly (04:00PM -> 21:00:05 GMT in January, 20:00:05 GMT in March, "
            "confirming Eastern time with DST). The timestamp also appears in the "
            "PDF header ('Injury Report: 01/20/26 04:00 PM'). Missing files return "
            "403, verified by requesting an invalid minute on a valid date. "
            "Retention boundary observed between 2025-12-15 (403) and 2025-12-28 "
            "(200); every date in 2016-2025 tested returned 403. PDFs extract as "
            "text with pypdf -- no OCR required."
        ),
    ),
    AvailabilitySource(
        name="sportradar_daily_injuries",
        endpoint=(
            "https://api.sportradar.com/nba/{access}/v8/{lang}/league/"
            "{year}/{month}/{day}/daily_injuries.json"
        ),
        historical_coverage="URL accepts 2013-2026; actual data depth unverified",
        timestamp_precision=PRECISION_DATE_ONLY,
        has_player_status=True,
        has_projected_lineup=False,
        has_confirmed_lineup=False,
        historical_asof_safety=ASOF_UNKNOWN,
        prospective_usefulness="moderate -- date-only stamps limit T-30 precision",
        cost_access="trial key required; not held. Returns 403 unauthenticated.",
        recommendation=(
            "Do not rely on for historical T-30 reconstruction unless a trial "
            "proves sub-daily precision. The documented ``update_date`` is a DATE, "
            "which alone cannot place a status before or after a 7:00pm anchor."
        ),
        evidence=(
            "Documentation lists fields desc/status/comment/start_date/update_date, "
            "with update_date described as 'Date of an injury update' and a 4-hour "
            "cache TTL. The URL year parameter accepting 2013 does not imply daily "
            "snapshots existed then; the endpoint is understood to be far newer. "
            "Unauthenticated probe returned HTTP 403, so nothing was verified."
        ),
        verified=False,
    ),
    AvailabilitySource(
        name="sportsdataio",
        endpoint="InjuriesByDate / StartingLineupsByDate",
        historical_coverage="historical database advertised; depth requires sales contact",
        timestamp_precision=PRECISION_UNKNOWN,
        has_player_status=True,
        has_projected_lineup=True,
        has_confirmed_lineup=True,
        historical_asof_safety=ASOF_UNKNOWN,
        prospective_usefulness="high if point-in-time state is preserved",
        cost_access="paid; trial by request. Nothing purchased.",
        recommendation=(
            "The only audited source distinguishing projected from confirmed "
            "lineups. Worth a trial specifically to test whether a historical "
            "response preserves point-in-time status or overwrites with final "
            "state -- that single question decides its value."
        ),
        evidence=(
            "Documentation states lineups are produced up to 15 days ahead, "
            "available by 9am ET on game day, and that injuries and lineups update "
            "every ten minutes in the 4-6 hours before tip. Whether a *historical* "
            "query returns the state as of then, or today's final state, is not "
            "documented and could not be tested without a key."
        ),
        verified=False,
    ),
    AvailabilitySource(
        name="balldontlie_injuries",
        endpoint="/v1/player_injuries",
        historical_coverage="none -- current state only",
        timestamp_precision=PRECISION_NONE,
        has_player_status=True,
        has_projected_lineup=False,
        has_confirmed_lineup=False,
        historical_asof_safety=ASOF_UNSAFE,
        prospective_usefulness="usable only if snapshotted forward by us",
        cost_access="included in existing GOAT subscription",
        recommendation=(
            "Snapshot prospectively as a cheap secondary source. Never backfill: "
            "there is no as-of timestamp to backfill against."
        ),
        evidence=(
            "Phase 3A3A0 verified the records carry only player, status, "
            "return_date and description -- no as-of timestamp, no history, no date "
            "filter. Descriptions discussed the 2025-26 season in the past tense, "
            "confirming the feed reports today's state."
        ),
    ),
    AvailabilitySource(
        name="balldontlie_lineups",
        endpoint="/v1/lineups",
        historical_coverage="2025-26 season only",
        timestamp_precision=PRECISION_NONE,
        has_player_status=False,
        has_projected_lineup=False,
        has_confirmed_lineup=True,
        historical_asof_safety=ASOF_UNSAFE,
        prospective_usefulness="none at T-30 -- populated only once a game begins",
        cost_access="included in existing GOAT subscription",
        recommendation=(
            "Prohibited at T-30 in any phase. Confirmed lineups arrive at tip-off, "
            "which is after our anchor by definition."
        ),
        evidence=(
            "Phase 3A3A0 verified zero rows for 2006 and 2024 games and twenty rows "
            "for a 2025-26 game, and the documentation states availability only "
            "once a game begins."
        ),
    ),
)

SOURCES_BY_NAME: Final[dict[str, AvailabilitySource]] = {s.name: s for s in SOURCE_MATRIX}

#: Sources that may never supply a historical as-of feature.
HISTORICALLY_UNSAFE: Final[dict[str, str]] = {
    s.name: s.recommendation for s in SOURCE_MATRIX if s.historical_asof_safety != ASOF_SAFE
}


def is_historically_safe(source: str) -> bool:
    """Whether a source may supply a *historical* T-30 availability feature."""
    entry = SOURCES_BY_NAME.get(source)
    return bool(entry and entry.historical_asof_safety == ASOF_SAFE)
