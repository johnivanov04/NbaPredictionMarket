"""Source time versus observation time.

Every external fact has two clocks, and conflating them is the quiet way a
latency study becomes fiction:

* **source time** -- what the publisher says. An NBA report stamped 17:30 says
  it describes the world at 17:30.
* **observation time** -- when our system could actually see it. The same report
  might not be fetchable until 17:30:12, and nobody could have acted on it
  before then.

For prospective research a fact may influence our state only from
``first_observed_at_utc`` onward. Source time is preserved because it is what
the league published; it is *not* evidence of public availability, and this
module refuses to treat it as such.

The distinction only matters if it is measured, so every record carries the
request and response instants and the derived latency rather than a single
timestamp standing in for all of them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

#: Reasons a source timestamp may not be treated as publicly actionable.
SOURCE_TIME_IS_NOT_AVAILABILITY: str = (
    "the publisher's own timestamp is not proof the artefact was fetchable "
    "then; only first_observed_at_utc establishes when we could have acted"
)


def content_hash(payload: bytes) -> str:
    """Stable digest used to recognise an unchanged artefact."""
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ObservationTiming:
    """The clocks around a single fetch."""

    request_started_at_utc: datetime
    response_received_at_utc: datetime
    first_observed_at_utc: datetime
    source_timestamp_utc: datetime | None = None
    http_last_modified_utc: datetime | None = None

    def __post_init__(self) -> None:
        for name in (
            "request_started_at_utc",
            "response_received_at_utc",
            "first_observed_at_utc",
        ):
            value = getattr(self, name)
            if value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware UTC")
        if self.response_received_at_utc < self.request_started_at_utc:
            raise ValueError("response cannot precede its own request")

    @property
    def round_trip_seconds(self) -> float:
        return (
            self.response_received_at_utc - self.request_started_at_utc
        ).total_seconds()

    @property
    def publication_latency_seconds(self) -> float | None:
        """How long after the source's own stamp we first saw it.

        Negative would mean we observed an artefact before the publisher says
        it existed, which indicates a clock problem rather than prescience, so
        it is surfaced rather than clamped to zero.
        """
        if self.source_timestamp_utc is None:
            return None
        return (
            self.first_observed_at_utc - self.source_timestamp_utc
        ).total_seconds()

    def actionable_from(self) -> datetime:
        """The earliest instant this fact may influence our state."""
        return self.first_observed_at_utc

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_started_at_utc": self.request_started_at_utc.isoformat(),
            "response_received_at_utc": self.response_received_at_utc.isoformat(),
            "first_observed_at_utc": self.first_observed_at_utc.isoformat(),
            "source_timestamp_utc": (
                self.source_timestamp_utc.isoformat()
                if self.source_timestamp_utc else None
            ),
            "http_last_modified_utc": (
                self.http_last_modified_utc.isoformat()
                if self.http_last_modified_utc else None
            ),
            "round_trip_seconds": round(self.round_trip_seconds, 6),
            "publication_latency_seconds": (
                round(self.publication_latency_seconds, 3)
                if self.publication_latency_seconds is not None else None
            ),
            "actionable_from_utc": self.actionable_from().isoformat(),
            "note": SOURCE_TIME_IS_NOT_AVAILABILITY,
        }


@dataclass(frozen=True)
class RawObservation:
    """One immutable captured artefact plus its provenance."""

    source: str
    url: str
    http_status: int
    timing: ObservationTiming
    content_sha256: str
    content_bytes: int
    request: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and 200 <= self.http_status < 300

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "url": self.url,
            "http_status": self.http_status,
            "content_sha256": self.content_sha256,
            "content_bytes": self.content_bytes,
            "request": self.request,
            "error": self.error,
            "succeeded": self.succeeded,
            **self.timing.to_dict(),
        }


def visible_at(
    observations: list[RawObservation], moment: datetime
) -> list[RawObservation]:
    """Observations our system could actually have known by ``moment``.

    Filters on ``first_observed_at_utc``, never on source time. Replaying with
    source time would let a report influence a decision taken before anyone
    could have fetched it.
    """
    if moment.tzinfo is None:
        raise ValueError("moment must be timezone-aware UTC")
    return [o for o in observations if o.timing.first_observed_at_utc <= moment]
