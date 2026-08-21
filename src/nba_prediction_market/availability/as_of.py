"""As-of queries: what was known at an instant, and nothing later.

One rule, enforced in one place::

    observed_at <= anchor

An observation exactly at the anchor is allowed; one second later is not. This
is deliberately the same boundary convention as the Phase 2 candle selector, so
"at or before" means the same thing everywhere in the project.

A second rule matters just as much: an observation whose source only gives a
calendar date **cannot** satisfy a T-30 anchor, because a date does not place it
before or after 7:00pm. Such records are refused by
:func:`state_at`, not silently accepted.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from nba_prediction_market.availability.events import (
    STATUS_UNKNOWN,
    AvailabilityEvent,
)


class AnchorViolationError(RuntimeError):
    """Raised when an observation after the anchor would be used."""


@dataclass(frozen=True)
class PlayerState:
    """The latest known state for one player at an anchor."""

    player_key: Any
    player_name: str
    status_normalized: str
    status_raw: str | None
    observed_at_utc: datetime
    source: str
    staleness_seconds: float
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_key": self.player_key,
            "player_name": self.player_name,
            "status_normalized": self.status_normalized,
            "status_raw": self.status_raw,
            "observed_at_utc": self.observed_at_utc.isoformat(),
            "source": self.source,
            "staleness_seconds": self.staleness_seconds,
            "reason": self.reason,
        }


def _player_key(event: AvailabilityEvent) -> Any:
    """Prefer a stable id; fall back to the name only when no id exists."""
    return (
        event.nba_reference_id
        if event.nba_reference_id is not None
        else event.player_id
        if event.player_id is not None
        else event.player_name
    )


def observations_at_or_before(
    events: Iterable[AvailabilityEvent], anchor: datetime
) -> list[AvailabilityEvent]:
    """Every observation known at ``anchor``, oldest first.

    The single chokepoint that prevents future information entering a
    prediction.
    """
    if anchor.tzinfo is None:
        raise ValueError("anchor must be timezone-aware")
    kept = [e for e in events if e.observed_at_utc <= anchor]
    return sorted(kept, key=lambda e: e.observed_at_utc)


def state_at(
    events: Sequence[AvailabilityEvent],
    anchor: datetime,
    *,
    require_anchor_safe_precision: bool = True,
) -> dict[Any, PlayerState]:
    """Latest known state per player at ``anchor``.

    ``require_anchor_safe_precision`` refuses date-only observations, which is
    the default because the anchor is a time of day. Set it False only for
    coarse day-level analysis, never for a T-30 feature.
    """
    eligible = observations_at_or_before(events, anchor)
    latest: dict[Any, PlayerState] = {}
    for event in eligible:
        if require_anchor_safe_precision and not event.anchor_safe:
            continue
        key = _player_key(event)
        # Events are sorted ascending, so a later one legitimately supersedes.
        latest[key] = PlayerState(
            player_key=key,
            player_name=event.player_name,
            status_normalized=event.status_normalized,
            status_raw=event.status_raw,
            observed_at_utc=event.observed_at_utc,
            source=event.source,
            staleness_seconds=(anchor - event.observed_at_utc).total_seconds(),
            reason=event.reason,
        )
    return latest


def status_for(
    events: Sequence[AvailabilityEvent], anchor: datetime, player_key: Any
) -> str:
    """One player's status at the anchor, or ``unknown`` if never observed.

    Absence is never converted into availability.
    """
    state = state_at(events, anchor).get(player_key)
    return state.status_normalized if state else STATUS_UNKNOWN


def assert_no_future_observation(
    events: Iterable[AvailabilityEvent], anchor: datetime
) -> None:
    """Raise if any supplied observation postdates the anchor."""
    offenders = [e for e in events if e.observed_at_utc > anchor]
    if offenders:
        first = offenders[0]
        raise AnchorViolationError(
            f"{len(offenders)} observation(s) postdate the anchor {anchor.isoformat()}; "
            f"first is {first.player_name} from {first.source} at "
            f"{first.observed_at_utc.isoformat()}. These are future information."
        )


@dataclass(frozen=True)
class AnchorCoverage:
    """How well a game's anchor is covered by captured observations."""

    anchor: datetime
    observations_before_anchor: int
    latest_observation_utc: datetime | None
    staleness_seconds: float | None
    anchor_safe_sources: list[str]
    refused_date_only_sources: list[str]

    @property
    def usable(self) -> bool:
        return self.latest_observation_utc is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_utc": self.anchor.isoformat(),
            "observations_before_anchor": self.observations_before_anchor,
            "latest_observation_utc": (
                self.latest_observation_utc.isoformat()
                if self.latest_observation_utc else None
            ),
            "staleness_seconds": self.staleness_seconds,
            "anchor_safe_sources": self.anchor_safe_sources,
            "refused_date_only_sources": self.refused_date_only_sources,
            "usable": self.usable,
        }


def coverage_at(
    events: Sequence[AvailabilityEvent], anchor: datetime
) -> AnchorCoverage:
    """Diagnose whether an anchor is adequately covered."""
    eligible = observations_at_or_before(events, anchor)
    safe = [e for e in eligible if e.anchor_safe]
    refused = sorted({e.source for e in eligible if not e.anchor_safe})
    latest = max((e.observed_at_utc for e in safe), default=None)
    return AnchorCoverage(
        anchor=anchor,
        observations_before_anchor=len(safe),
        latest_observation_utc=latest,
        staleness_seconds=(anchor - latest).total_seconds() if latest else None,
        anchor_safe_sources=sorted({e.source for e in safe}),
        refused_date_only_sources=refused,
    )
