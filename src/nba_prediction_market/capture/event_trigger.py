"""Detect meaningful availability changes and raise market sampling around them.

Phase 4A1 found the one place Kalshi behaves imperfectly: the 5-15 minutes
after an official injury report changes a rotation player's status. One-minute
candles are the coarsest lens that can see it at all.

So the collector runs at a calm baseline and *reacts*: when a newly observed
report changes a status, the affected games are sampled densely for a while.
This costs almost nothing on a quiet night and gives real resolution exactly
where the previous phase says resolution matters.

Two rules keep this research rather than trading:

* the trigger is the **report change**, never the eventual outcome;
* role weight attached to an event uses only basketball history from before the
  report, never tonight's minutes.

**Every transition is captured, not only the ones that looked profitable
historically** -- none did, so filtering on them would bake a null result into
the data collection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from nba_prediction_market.features.availability_features import (
    NOT_REPORTED,
    STATUS_RANK,
)

#: Sampling interval when nothing is happening.
BASELINE_INTERVAL_SECONDS: float = 60.0
#: Sampling interval while a game is inside an event window.
EVENT_INTERVAL_SECONDS: float = 5.0
#: How long a game stays elevated after an event.
EVENT_WINDOW_SECONDS: float = 1800.0
#: Role weight below which a change is recorded but does not raise sampling.
MEANINGFUL_ROLE_MINUTES: float = 8.0

#: Research sampling targets after an event. Targets, never guarantees: if no
#: quote exists until +42s, +42s is what gets recorded.
EVENT_TARGET_OFFSETS_SECONDS: tuple[int, ...] = (
    0, 10, 30, 60, 120, 300, 600, 900, 1800,
)


@dataclass(frozen=True)
class StatusChange:
    """One player's status change between consecutive observed reports."""

    game_id: Any
    team_code: str
    player_id: Any
    player_name: str
    from_status: str
    to_status: str
    detected_at_utc: datetime
    source_report_timestamp_utc: datetime | None
    role_minutes: float | None

    @property
    def direction(self) -> str:
        old = STATUS_RANK.get(self.from_status)
        new = STATUS_RANK.get(self.to_status)
        if old is None or new is None:
            return "unknown"
        if new > old:
            return "downgrade"
        return "upgrade" if new < old else "unchanged"

    @property
    def is_meaningful(self) -> bool:
        """Whether this change justifies raising the sampling rate.

        Unknown role counts as meaningful: a player we cannot weight might be
        important, and the cost of sampling is small next to the cost of
        missing the only phenomenon we know exists.
        """
        if self.role_minutes is None:
            return True
        return self.role_minutes >= MEANINGFUL_ROLE_MINUTES

    def to_dict(self) -> dict[str, Any]:
        return {
            "nba_game_id": self.game_id,
            "team_code": self.team_code,
            "player_id": self.player_id,
            "player_name": self.player_name,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "transition": f"{self.from_status} -> {self.to_status}",
            "direction": self.direction,
            "detected_at_utc": self.detected_at_utc.isoformat(),
            "source_report_timestamp_utc": (
                self.source_report_timestamp_utc.isoformat()
                if self.source_report_timestamp_utc else None
            ),
            "role_minutes": self.role_minutes,
            "is_meaningful": self.is_meaningful,
        }


def diff_states(
    previous: Mapping[tuple[Any, str, Any], str] | None,
    current: Mapping[tuple[Any, str, Any], str],
    *,
    detected_at_utc: datetime,
    source_report_timestamp_utc: datetime | None,
    roles: Mapping[tuple[Any, Any], float | None],
    player_names: Mapping[Any, str] | None = None,
) -> list[StatusChange]:
    """Status changes between two observed report states.

    With no previous state nothing is emitted. The first report of a night
    establishes a baseline; calling every player in it a "change" would
    manufacture a burst of events that never happened.

    A player who disappears from the report becomes ``not_reported`` rather
    than ``available`` -- absence has never meant availability in this project.
    """
    if previous is None:
        return []

    names = player_names or {}
    changes: list[StatusChange] = []
    for key, status in current.items():
        old = previous.get(key, NOT_REPORTED)
        if old == status:
            continue
        game_id, team, player = key
        changes.append(StatusChange(
            game_id=game_id, team_code=team, player_id=player,
            player_name=names.get(player, str(player)),
            from_status=old, to_status=status,
            detected_at_utc=detected_at_utc,
            source_report_timestamp_utc=source_report_timestamp_utc,
            role_minutes=roles.get((game_id, player)),
        ))
    for key, old in previous.items():
        if key in current:
            continue
        game_id, team, player = key
        changes.append(StatusChange(
            game_id=game_id, team_code=team, player_id=player,
            player_name=names.get(player, str(player)),
            from_status=old, to_status=NOT_REPORTED,
            detected_at_utc=detected_at_utc,
            source_report_timestamp_utc=source_report_timestamp_utc,
            role_minutes=roles.get((game_id, player)),
        ))
    return changes


@dataclass
class CaptureScheduler:
    """Decides how often each game's markets are sampled.

    Restart-safe by construction: state is a plain map of game to window
    expiry, so a restarted process simply resumes at the baseline rate and
    re-elevates on the next observed change.
    """

    elevated_until: dict[Any, datetime] = field(default_factory=dict)
    triggered: set[tuple[Any, Any, str, str, str]] = field(default_factory=set)

    def register(self, change: StatusChange) -> bool:
        """Record a change. Returns whether it newly elevated its game.

        Deduplicated on the change's identity, so re-observing the same report
        -- which happens on every restart and every unchanged poll -- cannot
        re-trigger a window that has already run.
        """
        key = (
            change.game_id, change.player_id,
            change.from_status, change.to_status,
            change.source_report_timestamp_utc.isoformat()
            if change.source_report_timestamp_utc else "",
        )
        if key in self.triggered:
            return False
        self.triggered.add(key)
        if not change.is_meaningful:
            return False
        expiry = change.detected_at_utc + timedelta(seconds=EVENT_WINDOW_SECONDS)
        current = self.elevated_until.get(change.game_id)
        if current is None or expiry > current:
            self.elevated_until[change.game_id] = expiry
        return True

    def interval_for(self, game_id: Any, now: datetime) -> float:
        """Seconds until this game's markets should be sampled again."""
        expiry = self.elevated_until.get(game_id)
        if expiry is not None and now < expiry:
            return EVENT_INTERVAL_SECONDS
        return BASELINE_INTERVAL_SECONDS

    def elevated_games(self, now: datetime) -> list[Any]:
        return [g for g, until in self.elevated_until.items() if now < until]

    def prune(self, now: datetime) -> int:
        expired = [g for g, until in self.elevated_until.items() if now >= until]
        for game in expired:
            del self.elevated_until[game]
        return len(expired)
