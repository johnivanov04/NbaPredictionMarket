"""Map scheduled games to their Kalshi contracts, ahead of capture.

A game that reaches its anchor without a resolved market is a hole in the
dataset that no later work can fill, so identity is resolved early and
re-resolved often.

The distinction that matters operationally is between *not listed yet* and
*failed to match*. Kalshi lists NBA game markets progressively -- a game in
March simply has no market in August -- and treating that as an error would
bury the real failures under months of noise. So absence is graded by how close
the game is:

* months away, no market  -> INFO, entirely normal
* approaching its anchor, no market -> escalates, because now it matters
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from nba_prediction_market.matching.team_names import resolve_team

EASTERN = ZoneInfo("America/New_York")

#: Event ticker shape: ``KXNBAGAME-26OCT20OKCSAS`` -- two-digit year, month,
#: day, then the two team codes concatenated, away first.
#:
#: Team codes are matched at **exactly three characters**, the assumption Phase
#: 1 established and verified against every market whose title carries the
#: "A at B" form. A flexible width is not merely looser, it is wrong: a greedy
#: two-to-four match splits "OKCSAS" as "OKCS" + "AS" and resolves neither.
TICKER_PATTERN = re.compile(
    r"^KXNBAGAME-(\d{2})([A-Z]{3})(\d{2})([A-Z]{3})([A-Z]{3})$"
)
MONTHS: dict[str, int] = {
    m: i + 1 for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
         "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    )
}

MATCHED: str = "matched"
NOT_YET_LISTED: str = "not_yet_listed"
AMBIGUOUS: str = "ambiguous"
UNMATCHED: str = "unmatched"

#: Inside this many hours before tipoff, a missing market stops being normal.
IDENTITY_REQUIRED_HOURS: float = 12.0
#: Inside this many hours, a missing market is critical.
IDENTITY_CRITICAL_HOURS: float = 6.0


@dataclass(frozen=True)
class MarketMapping:
    """One game's Kalshi identity, or the documented absence of one."""

    source_game_id: Any
    matchup: str
    tipoff_utc: datetime
    status: str
    event_ticker: str | None = None
    candidates: tuple[str, ...] = ()

    def severity(self, now: datetime) -> str:
        """How loudly a missing market should complain, given the clock."""
        if self.status == MATCHED:
            return "INFO"
        hours = (self.tipoff_utc - now).total_seconds() / 3600.0
        if hours <= IDENTITY_CRITICAL_HOURS:
            return "CRITICAL"
        if hours <= IDENTITY_REQUIRED_HOURS:
            return "WARNING"
        return "INFO"

    def to_dict(self, now: datetime) -> dict[str, Any]:
        return {
            "source_game_id": self.source_game_id,
            "matchup": self.matchup,
            "tipoff_utc": self.tipoff_utc,
            "status": self.status,
            "event_ticker": self.event_ticker,
            "candidates": list(self.candidates),
            "severity": self.severity(now),
            "hours_to_tipoff": round(
                (self.tipoff_utc - now).total_seconds() / 3600.0, 2
            ),
        }


def parse_event_ticker(ticker: str) -> tuple[str, str, str] | None:
    """``(et_date, away, home)`` from an event ticker, or None.

    Team codes inside the ticker are Kalshi's own; they are normalised through
    the franchise resolver so a mapping never depends on two spellings of the
    same team agreeing by luck.
    """
    match = TICKER_PATTERN.match(ticker.strip().upper())
    if not match:
        return None
    yy, mon, dd, away_raw, home_raw = match.groups()
    month = MONTHS.get(mon)
    if month is None:
        return None
    away, home = resolve_team(away_raw), resolve_team(home_raw)
    if not (away.ok and home.ok):
        return None
    return (
        f"{2000 + int(yy):04d}-{month:02d}-{int(dd):02d}",
        away.abbreviation,
        home.abbreviation,
    )


def map_games(
    games: list[Any], event_tickers: list[str], *, now: datetime
) -> list[MarketMapping]:
    """Resolve each scheduled game to a Kalshi event.

    Matching is on the Eastern game date plus both team codes -- the same key
    every earlier phase used. A key matching more than one ticker is reported
    ambiguous rather than resolved arbitrarily: attaching a quote from the
    wrong game is the one error that would silently corrupt the dataset.
    """
    index: dict[tuple[str, str, str], list[str]] = {}
    for ticker in event_tickers:
        parsed = parse_event_ticker(ticker)
        if parsed is None:
            continue
        index.setdefault(parsed, []).append(ticker)

    mappings: list[MarketMapping] = []
    for game in games:
        et_date = game.tipoff_utc.astimezone(EASTERN).date().isoformat()
        key = (et_date, game.away_team, game.home_team)
        matchup = f"{game.away_team}@{game.home_team}"
        found = index.get(key, [])
        if len(found) == 1:
            status, ticker = MATCHED, found[0]
        elif len(found) > 1:
            status, ticker = AMBIGUOUS, None
        else:
            status, ticker = NOT_YET_LISTED, None
        mappings.append(MarketMapping(
            source_game_id=game.source_game_id,
            matchup=matchup,
            tipoff_utc=game.tipoff_utc,
            status=status,
            event_ticker=ticker,
            candidates=tuple(found),
        ))
    return mappings


def summarise(mappings: list[MarketMapping], now: datetime) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    for mapping in mappings:
        by_status[mapping.status] = by_status.get(mapping.status, 0) + 1
    urgent = [
        m.to_dict(now) for m in mappings
        if m.status != MATCHED and m.severity(now) in ("WARNING", "CRITICAL")
    ]
    listable = [
        m for m in mappings
        if (m.tipoff_utc - now) <= timedelta(hours=IDENTITY_REQUIRED_HOURS)
    ]
    return {
        "games_considered": len(mappings),
        "by_status": by_status,
        "games_within_identity_window": len(listable),
        "urgent_missing_identity": urgent,
        "note": (
            "a game months away with no market is INFO and entirely normal; "
            "the same absence inside 12 hours is a WARNING and inside 6 is "
            "CRITICAL"
        ),
    }
