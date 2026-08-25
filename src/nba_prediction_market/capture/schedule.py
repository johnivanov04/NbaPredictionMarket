"""The forward schedule the collector captures against.

This is an *operational* schedule, not the modelling dataset. Phase 3A0 built
the historical frame and froze it; nothing here touches that. What the collector
needs is different: which games are coming, when their anchors fall, and what
changed since the last refresh.

Two properties of a published NBA schedule drive the design:

* **It is incomplete by design.** The league assigns only 80 of each team's 82
  games up front; the last two depend on Emirates NBA Cup results and are
  announced in December. A schedule missing 30 games is therefore *expected*,
  not corrupt, and must not be reported as a data fault.
* **It changes.** Games are rescheduled and postponed. A refresh that silently
  overwrote a tipoff would destroy the only record that it moved, so every
  change is recorded and the previous value is kept.

Preseason games are carried because they are how the collector gets tested
under real conditions. They are labelled, and every consumer that matters
excludes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from nba_prediction_market.ingestion.game_phase import (
    PHASE_PRESEASON,
    PHASE_REGULAR_SEASON,
)

#: Anchors the collector schedules capture around, in minutes before tipoff.
CAPTURE_ANCHOR_MINUTES: tuple[int, ...] = (360, 180, 60, 30)
ANCHOR_LABELS: dict[int, str] = {360: "T-6h", 180: "T-3h", 60: "T-1h", 30: "T-30m"}

#: A full NBA regular season, once the Cup-dependent games are assigned.
FULL_REGULAR_SEASON_GAMES: int = 1230
GAMES_PER_TEAM: int = 82
#: Games per team the league assigns before the Cup concludes.
INITIALLY_ASSIGNED_PER_TEAM: int = 80

CHANGE_ADDED: str = "added"
CHANGE_TIPOFF_MOVED: str = "tipoff_moved"
CHANGE_UNCHANGED: str = "unchanged"
CHANGE_DISAPPEARED: str = "disappeared_from_source"


@dataclass(frozen=True)
class ScheduledGame:
    """One game the collector may need to capture."""

    source_game_id: Any
    season: int
    phase: str
    tipoff_utc: datetime
    home_team: str
    away_team: str
    source: str
    first_seen_at_utc: datetime

    @property
    def is_capturable(self) -> bool:
        """Whether the collector should watch this game at all."""
        return self.phase in (PHASE_PRESEASON, PHASE_REGULAR_SEASON)

    @property
    def counts_toward_research(self) -> bool:
        """Preseason is infrastructure testing and counts toward nothing."""
        return self.phase == PHASE_REGULAR_SEASON

    def anchors(self) -> dict[str, datetime]:
        return {
            ANCHOR_LABELS[m]: self.tipoff_utc - timedelta(minutes=m)
            for m in CAPTURE_ANCHOR_MINUTES
        }

    def key(self) -> tuple[int, str, str, str]:
        """Identity independent of tipoff, so a moved game is still the same game."""
        return (
            self.season,
            self.tipoff_utc.date().isoformat(),
            self.away_team,
            self.home_team,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_game_id": self.source_game_id,
            "season": self.season,
            "phase": self.phase,
            "tipoff_utc": self.tipoff_utc,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "source": self.source,
            "first_seen_at_utc": self.first_seen_at_utc,
            "counts_toward_research": self.counts_toward_research,
        }


@dataclass(frozen=True)
class ScheduleChange:
    """One difference between a stored schedule and a freshly fetched one."""

    change: str
    source_game_id: Any
    matchup: str
    previous_tipoff_utc: datetime | None
    new_tipoff_utc: datetime | None
    detected_at_utc: datetime

    def to_dict(self) -> dict[str, Any]:
        moved = (
            (self.new_tipoff_utc - self.previous_tipoff_utc).total_seconds() / 3600.0
            if self.previous_tipoff_utc and self.new_tipoff_utc else None
        )
        return {
            "change": self.change,
            "source_game_id": self.source_game_id,
            "matchup": self.matchup,
            "previous_tipoff_utc": self.previous_tipoff_utc,
            "new_tipoff_utc": self.new_tipoff_utc,
            "moved_hours": round(moved, 3) if moved is not None else None,
            "detected_at_utc": self.detected_at_utc,
        }


def diff_schedules(
    stored: list[ScheduledGame],
    fetched: list[ScheduledGame],
    *,
    detected_at_utc: datetime,
) -> list[ScheduleChange]:
    """What a refresh would change.

    Games are matched on source id where possible. A tipoff that moved is a
    *change*, recorded with both values -- never a silent overwrite, because
    the fact that a game moved is exactly what a postponement study needs.

    A game that disappears from the source is reported, not deleted: a source
    briefly omitting a game must not quietly erase it from our record.
    """
    by_id_stored = {g.source_game_id: g for g in stored}
    by_id_fetched = {g.source_game_id: g for g in fetched}
    changes: list[ScheduleChange] = []

    for game_id, game in by_id_fetched.items():
        matchup = f"{game.away_team}@{game.home_team}"
        previous = by_id_stored.get(game_id)
        if previous is None:
            changes.append(ScheduleChange(
                CHANGE_ADDED, game_id, matchup, None, game.tipoff_utc,
                detected_at_utc,
            ))
        elif previous.tipoff_utc != game.tipoff_utc:
            changes.append(ScheduleChange(
                CHANGE_TIPOFF_MOVED, game_id, matchup,
                previous.tipoff_utc, game.tipoff_utc, detected_at_utc,
            ))

    for game_id, previous in by_id_stored.items():
        if game_id not in by_id_fetched:
            changes.append(ScheduleChange(
                CHANGE_DISAPPEARED, game_id,
                f"{previous.away_team}@{previous.home_team}",
                previous.tipoff_utc, None, detected_at_utc,
            ))
    return changes


def apply_refresh(
    stored: list[ScheduledGame], fetched: list[ScheduledGame]
) -> list[ScheduledGame]:
    """Merge a fetch into the stored schedule, idempotently.

    Newly published games are added and moved tipoffs are updated, but
    ``first_seen_at_utc`` is preserved from the stored record so we keep
    knowing when a game first appeared. A game absent from this fetch is
    retained rather than dropped.
    """
    merged = {g.source_game_id: g for g in stored}
    for game in fetched:
        previous = merged.get(game.source_game_id)
        merged[game.source_game_id] = (
            game if previous is None
            else ScheduledGame(
                source_game_id=game.source_game_id,
                season=game.season,
                phase=game.phase,
                tipoff_utc=game.tipoff_utc,
                home_team=game.home_team,
                away_team=game.away_team,
                source=game.source,
                first_seen_at_utc=previous.first_seen_at_utc,
            )
        )
    return sorted(merged.values(), key=lambda g: (g.tipoff_utc, g.source_game_id))


@dataclass
class CompletenessReport:
    """Whether a season's schedule is fully assigned yet."""

    season: int
    games: int
    teams: int
    min_games_per_team: int
    max_games_per_team: int
    expected_games: int = FULL_REGULAR_SEASON_GAMES

    @property
    def unassigned_games(self) -> int:
        return max(self.expected_games - self.games, 0)

    @property
    def is_awaiting_cup_assignment(self) -> bool:
        """Exactly the published-before-the-Cup pattern, not a data fault.

        Every team on 80 with the shortfall equal to the missing pairings is
        the signature of a schedule that simply has not had its Cup-dependent
        games assigned yet.
        """
        if self.games >= self.expected_games:
            return False
        per_team_shortfall = GAMES_PER_TEAM - self.max_games_per_team
        return (
            self.min_games_per_team == self.max_games_per_team
            and 0 < per_team_shortfall <= 2
            and self.unassigned_games == self.teams * per_team_shortfall // 2
        )

    @property
    def status(self) -> str:
        if self.games >= self.expected_games:
            return "complete"
        if self.is_awaiting_cup_assignment:
            return "incomplete_by_design"
        return "unexpectedly_incomplete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "games_assigned": self.games,
            "expected_games": self.expected_games,
            "unassigned_games": self.unassigned_games,
            "teams": self.teams,
            "games_per_team": [self.min_games_per_team, self.max_games_per_team],
            "status": self.status,
            "explanation": (
                "the last two games per team depend on Emirates NBA Cup results "
                "and are published later; a refresh is expected to add them"
                if self.status == "incomplete_by_design"
                else "schedule is fully assigned"
                if self.status == "complete"
                else "shortfall does not match the Cup-assignment pattern; investigate"
            ),
        }


def assess_completeness(
    games: list[ScheduledGame], season: int
) -> CompletenessReport:
    regular = [
        g for g in games if g.season == season and g.phase == PHASE_REGULAR_SEASON
    ]
    counts: dict[str, int] = {}
    for game in regular:
        counts[game.home_team] = counts.get(game.home_team, 0) + 1
        counts[game.away_team] = counts.get(game.away_team, 0) + 1
    values = list(counts.values()) or [0]
    return CompletenessReport(
        season=season,
        games=len(regular),
        teams=len(counts),
        min_games_per_team=min(values),
        max_games_per_team=max(values),
    )


def upcoming(
    games: list[ScheduledGame], now: datetime, horizon_hours: float
) -> list[ScheduledGame]:
    """Capturable games tipping inside the horizon, soonest first."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware UTC")
    limit = now + timedelta(hours=horizon_hours)
    return sorted(
        (g for g in games if g.is_capturable and now <= g.tipoff_utc <= limit),
        key=lambda g: g.tipoff_utc,
    )


def reports_expected(games: list[ScheduledGame]) -> bool:
    """Whether the NBA should be publishing official Injury Reports for these.

    The Injury Report is a regular-season and playoff product: the league's
    reporting policy does not cover preseason exhibitions, so on a preseason-
    only night the absence of a report is the expected outcome rather than a
    fault to alarm on.

    Deliberately permissive -- one research game in the window is enough to
    expect publication. Getting this wrong in the lenient direction would
    suppress a real alarm on a real slate, which is far worse than tolerating
    a stray "expected" on a mixed night.

    This never gates *capture*. The collector requests every slot regardless,
    so a preseason report that does get published is archived normally.
    """
    return any(game.counts_toward_research for game in games)
