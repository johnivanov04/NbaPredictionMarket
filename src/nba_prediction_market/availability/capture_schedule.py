"""Schedule-aware capture cadence for prospective availability collection.

Polling every source constantly is wasteful and disrespectful of rate limits.
What actually matters is a simple guarantee:

    **for every game, at least one successful snapshot lands in a narrow window
    immediately BEFORE its T-30 anchor.**

Everything else is context. So the planner works backwards from each game's
scheduled tipoff and emits offsets, densifying as tip approaches, with a final
mandatory capture shortly before the anchor.

Nothing is ever captured after an anchor and labelled as that anchor: the
planner refuses to emit such an offset, and
:func:`nba_prediction_market.availability.as_of.state_at` would reject it anyway.

The NBA official report publishes on a fixed 30-minute grid, so for that source
the planner also names the exact report slot to fetch -- the latest slot at or
before the anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

#: Minutes before tipoff that the prediction anchor sits at.
ANCHOR_MINUTES_BEFORE_TIP: Final = 30

#: Offsets before tipoff, in minutes. The last entry is the guaranteed capture
#: immediately before the anchor; earlier ones provide context and redundancy.
DEFAULT_OFFSETS_MINUTES: Final[tuple[int, ...]] = (
    24 * 60, 6 * 60, 3 * 60, 120, 90, 60, 45, 35, 31,
)

#: The NBA report grid, in minutes.
REPORT_GRID_MINUTES: Final = 30


@dataclass(frozen=True)
class CaptureTask:
    """One planned capture for one game."""

    game_id: Any
    source: str
    scheduled_tipoff_utc: datetime
    anchor_utc: datetime
    capture_at_utc: datetime
    minutes_before_tip: int
    is_anchor_guarantee: bool
    report_slot_utc: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "source": self.source,
            "scheduled_tipoff_utc": self.scheduled_tipoff_utc.isoformat(),
            "anchor_utc": self.anchor_utc.isoformat(),
            "capture_at_utc": self.capture_at_utc.isoformat(),
            "minutes_before_tip": self.minutes_before_tip,
            "is_anchor_guarantee": self.is_anchor_guarantee,
            "report_slot_utc": (
                self.report_slot_utc.isoformat() if self.report_slot_utc else None
            ),
        }


def anchor_for(scheduled_tipoff_utc: datetime) -> datetime:
    """The prediction anchor for a game."""
    if scheduled_tipoff_utc.tzinfo is None:
        raise ValueError("scheduled_tipoff_utc must be timezone-aware")
    return scheduled_tipoff_utc.astimezone(UTC) - timedelta(
        minutes=ANCHOR_MINUTES_BEFORE_TIP
    )


def latest_report_slot(anchor: datetime, grid_minutes: int = REPORT_GRID_MINUTES) -> datetime:
    """The newest fixed-grid report slot at or before ``anchor``.

    The NBA publishes on the half hour, so a 19:10 anchor resolves to the 19:00
    report -- never the 19:30 one, which had not been published yet.
    """
    if anchor.tzinfo is None:
        raise ValueError("anchor must be timezone-aware")
    moment = anchor.astimezone(UTC).replace(second=0, microsecond=0)
    minutes_past = (moment.minute % grid_minutes)
    return moment - timedelta(minutes=minutes_past)


def plan_captures(
    games: list[dict[str, Any]],
    sources: list[str],
    *,
    offsets_minutes: tuple[int, ...] = DEFAULT_OFFSETS_MINUTES,
    grid_sources: frozenset[str] = frozenset({"nba_official_injury_report"}),
) -> list[CaptureTask]:
    """Plan every capture for a slate, sorted by when it should run.

    Each game contributes one task per source per offset, plus a guaranteed
    capture just before the anchor. Offsets that would land at or after the
    anchor are dropped rather than silently mislabelled.
    """
    tasks: list[CaptureTask] = []
    for game in games:
        tipoff = game["scheduled_tipoff_utc"]
        if tipoff.tzinfo is None:
            raise ValueError(f"game {game.get('game_id')} has a naive tipoff")
        tipoff = tipoff.astimezone(UTC)
        anchor = anchor_for(tipoff)
        for source in sources:
            for offset in sorted(set(offsets_minutes), reverse=True):
                capture_at = tipoff - timedelta(minutes=offset)
                if capture_at >= anchor:
                    # Would be at or after the anchor: not usable for this game.
                    continue
                tasks.append(
                    CaptureTask(
                        game_id=game.get("game_id"),
                        source=source,
                        scheduled_tipoff_utc=tipoff,
                        anchor_utc=anchor,
                        capture_at_utc=capture_at,
                        minutes_before_tip=offset,
                        is_anchor_guarantee=(offset == min(
                            o for o in offsets_minutes
                            if tipoff - timedelta(minutes=o) < anchor
                        )),
                        report_slot_utc=(
                            latest_report_slot(anchor) if source in grid_sources else None
                        ),
                    )
                )
    return sorted(tasks, key=lambda t: (t.capture_at_utc, str(t.game_id), t.source))


def verify_anchor_guarantee(
    tasks: list[CaptureTask], games: list[dict[str, Any]]
) -> dict[str, Any]:
    """Confirm every game has at least one capture strictly before its anchor."""
    by_game: dict[Any, list[CaptureTask]] = {}
    for task in tasks:
        by_game.setdefault(task.game_id, []).append(task)
    uncovered = []
    for game in games:
        planned = by_game.get(game.get("game_id"), [])
        if not any(t.capture_at_utc < t.anchor_utc for t in planned):
            uncovered.append(game.get("game_id"))
    return {
        "games": len(games),
        "games_with_pre_anchor_capture": len(games) - len(uncovered),
        "uncovered_games": uncovered,
        "guaranteed": not uncovered,
    }
