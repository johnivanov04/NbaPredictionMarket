"""Normalized availability observations and status vocabulary.

Two conservative rules:

* **Absence is not availability.** A player missing from an injury report is
  ``unknown``, not ``available``, unless the source explicitly enumerates
  healthy players. Treating silence as a green light would systematically
  under-count unavailability.
* **Raw is always preserved.** ``status_raw`` keeps whatever the source said;
  ``status_normalized`` is our interpretation and may be ``unknown``.

Every observation carries an ``observed_at_utc`` (when *we* learned it) and an
optional ``source_updated_at_utc`` (when the source says the state changed),
plus an explicit :class:`TemporalPrecision`. A date-only record can never
masquerade as safe for a T-30 anchor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

STATUS_AVAILABLE: Final = "available"
STATUS_PROBABLE: Final = "probable"
STATUS_QUESTIONABLE: Final = "questionable"
STATUS_DOUBTFUL: Final = "doubtful"
STATUS_OUT: Final = "out"
STATUS_UNKNOWN: Final = "unknown"

NORMALIZED_STATUSES: Final[tuple[str, ...]] = (
    STATUS_AVAILABLE, STATUS_PROBABLE, STATUS_QUESTIONABLE,
    STATUS_DOUBTFUL, STATUS_OUT, STATUS_UNKNOWN,
)

#: Rough probability of playing, for ordering and diagnostics only. These are
#: not calibrated estimates and must not be used as model inputs directly.
STATUS_SEVERITY: Final[dict[str, int]] = {
    STATUS_AVAILABLE: 0, STATUS_PROBABLE: 1, STATUS_QUESTIONABLE: 2,
    STATUS_DOUBTFUL: 3, STATUS_OUT: 4, STATUS_UNKNOWN: -1,
}

#: Exact phrases observed across sources. Matching is exact after
#: normalisation -- never fuzzy, for the same reason team names never were.
_STATUS_ALIASES: Final[dict[str, str]] = {
    "available": STATUS_AVAILABLE,
    "active": STATUS_AVAILABLE,
    "probable": STATUS_PROBABLE,
    "questionable": STATUS_QUESTIONABLE,
    "doubtful": STATUS_DOUBTFUL,
    "out": STATUS_OUT,
    "out for season": STATUS_OUT,
    "out indefinitely": STATUS_OUT,
    "inactive": STATUS_OUT,
    "day to day": STATUS_QUESTIONABLE,
    "day-to-day": STATUS_QUESTIONABLE,
    "unknown": STATUS_UNKNOWN,
    "not yet submitted": STATUS_UNKNOWN,
}

_WHITESPACE = re.compile(r"\s+")


class TemporalPrecision:
    """How precisely an observation can be placed in time."""

    EXACT: Final = "exact_timestamp"
    DATE_ONLY: Final = "date_only"
    UNKNOWN: Final = "unknown"

    #: Only this precision may satisfy a T-30 anchor.
    ANCHOR_SAFE: Final = frozenset({EXACT})


def normalize_status(raw: str | None) -> str:
    """Map a source status onto the normalized vocabulary.

    Unrecognised text becomes ``unknown`` rather than being guessed at. A new
    phrase should be added deliberately after being seen in real data.
    """
    if raw is None:
        return STATUS_UNKNOWN
    key = _WHITESPACE.sub(" ", str(raw)).strip().casefold()
    if not key:
        return STATUS_UNKNOWN
    return _STATUS_ALIASES.get(key, STATUS_UNKNOWN)


@dataclass(frozen=True)
class AvailabilityEvent:
    """One normalized availability observation from one source."""

    source: str
    observed_at_utc: datetime
    player_name: str
    status_raw: str | None
    status_normalized: str
    temporal_precision: str
    source_updated_at_utc: datetime | None = None
    game_id: Any | None = None
    team_id: Any | None = None
    player_id: Any | None = None
    nba_reference_id: Any | None = None
    reason: str | None = None
    report_type: str | None = None
    is_projected_lineup: bool = False
    is_confirmed_lineup: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.observed_at_utc.tzinfo is None:
            raise ValueError("observed_at_utc must be timezone-aware")
        if self.status_normalized not in NORMALIZED_STATUSES:
            raise ValueError(f"unknown normalized status {self.status_normalized!r}")
        if self.temporal_precision not in {
            TemporalPrecision.EXACT, TemporalPrecision.DATE_ONLY, TemporalPrecision.UNKNOWN
        }:
            raise ValueError(f"unknown temporal precision {self.temporal_precision!r}")

    @property
    def anchor_safe(self) -> bool:
        """Whether this observation may be used against a T-30 anchor."""
        return self.temporal_precision in TemporalPrecision.ANCHOR_SAFE

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "observed_at_utc": self.observed_at_utc.isoformat(),
            "source_updated_at_utc": (
                self.source_updated_at_utc.isoformat()
                if self.source_updated_at_utc else None
            ),
            "temporal_precision": self.temporal_precision,
            "anchor_safe": self.anchor_safe,
            "game_id": self.game_id,
            "team_id": self.team_id,
            "player_id": self.player_id,
            "nba_reference_id": self.nba_reference_id,
            "player_name": self.player_name,
            "status_raw": self.status_raw,
            "status_normalized": self.status_normalized,
            "reason": self.reason,
            "report_type": self.report_type,
            "is_projected_lineup": self.is_projected_lineup,
            "is_confirmed_lineup": self.is_confirmed_lineup,
        }


def event_from_row(
    source: str,
    observed_at_utc: datetime,
    *,
    player_name: str,
    status_raw: str | None,
    temporal_precision: str,
    **kwargs: Any,
) -> AvailabilityEvent:
    """Build a normalized event, preserving the raw status."""
    return AvailabilityEvent(
        source=source,
        observed_at_utc=observed_at_utc.astimezone(UTC),
        player_name=player_name,
        status_raw=status_raw,
        status_normalized=normalize_status(status_raw),
        temporal_precision=temporal_precision,
        **kwargs,
    )
