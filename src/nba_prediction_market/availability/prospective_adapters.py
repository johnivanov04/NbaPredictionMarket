"""Secondary availability feeds, captured prospectively only.

The official NBA report is the primary source. These adapters exist so that a
second, independent view of availability accumulates alongside it from today
forward — useful for cross-checking the report parser, and as a fallback for
anchors the report misses.

Both feeds share one hard property: **they publish only current state.** Neither
carries an as-of timestamp or a history endpoint, so the observation time is the
moment *we* fetched it and nothing else. That makes them safe going forward and
unusable backward. Attaching today's injury list to a past game would import
information that did not exist at that game's anchor, so these adapters refuse
to write any observation whose effective time precedes the fetch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from nba_prediction_market.availability.snapshot_store import Snapshot, SnapshotStore
from nba_prediction_market.availability.sources import (
    ASOF_SAFE,
    PRECISION_NONE,
)

#: These feeds report present state, so a captured snapshot is only evidence
#: about anchors at or after the fetch instant.
CAPTURE_PRECISION = PRECISION_NONE


class SnapshotFetcher(Protocol):
    """Returns the decoded payload for one capture."""

    def __call__(self) -> Any: ...


@dataclass(frozen=True)
class CaptureOutcome:
    """Result of one prospective capture."""

    source: str
    status: str
    retrieved_at_utc: datetime | None = None
    records: int = 0
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status,
            "retrieved_at_utc": (
                self.retrieved_at_utc.isoformat() if self.retrieved_at_utc else None
            ),
            "records": self.records,
            "detail": self.detail,
        }


class ProspectiveAdapter:
    """Base adapter: fetch current state, stamp it with the fetch time, store it.

    Subclasses supply ``source_name`` and a fetch callable. The base class owns
    the part that must not vary: the observation time is the fetch time, and it
    is never taken from the payload.
    """

    source_name: str = "override-me"
    #: Set False for adapters whose credentials or contract are unverified.
    enabled: bool = True
    disabled_reason: str | None = None

    def __init__(
        self,
        store: SnapshotStore,
        *,
        fetch: SnapshotFetcher,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._fetch = fetch
        if now is None:
            from nba_prediction_market.ingestion.raw_store import utc_now

            now = utc_now
        self._now = now

    def request_key(self) -> str:
        """Identifies this capture in the store. Kept stable per adapter."""
        return self.source_name

    def _record_count(self, payload: Any) -> int:
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                return len(data)
        return len(payload) if isinstance(payload, list) else 0

    def capture(self) -> CaptureOutcome:
        """Fetch current state once and append it to the snapshot store."""
        if not self.enabled:
            return CaptureOutcome(
                source=self.source_name,
                status="disabled",
                detail=self.disabled_reason,
            )
        retrieved_at = self._now()
        if retrieved_at.tzinfo is None:
            raise ValueError("capture clock must return timezone-aware UTC datetimes")
        try:
            payload = self._fetch()
        except Exception as exc:  # an outage must not kill the capture run
            return CaptureOutcome(
                source=self.source_name,
                status="failed",
                retrieved_at_utc=retrieved_at,
                detail=f"{type(exc).__name__}: {exc}",
            )

        self._store.append(
            Snapshot(
                source=self.source_name,
                retrieved_at_utc=retrieved_at,
                request=self.request_key(),
                payload=payload,
                # No as-of timestamp exists in either feed. Leaving these unset
                # is what stops a consumer from back-dating the observation.
                source_report_timestamp=None,
                source_effective_date=None,
                content_type="application/json",
                metadata={
                    "precision": CAPTURE_PRECISION,
                    "asof_class": ASOF_SAFE,
                    "prospective_only": True,
                    "note": (
                        "Current-state feed. Valid only for anchors at or after "
                        "retrieved_at_utc; never attach to an earlier game."
                    ),
                },
            )
        )
        return CaptureOutcome(
            source=self.source_name,
            status="captured",
            retrieved_at_utc=retrieved_at,
            records=self._record_count(payload),
        )


class BallDontLieInjuriesAdapter(ProspectiveAdapter):
    """Snapshots ``/v1/player_injuries`` forward in time.

    The audit found this feed carries player, status, return_date and
    description and nothing else — no as-of stamp, no history, no date filter.
    Its *historical* use is prohibited by the project constraints and by the
    feed's own shape; snapshotting it forward is the only safe use.
    """

    source_name = "balldontlie_player_injuries"

    @classmethod
    def from_client(
        cls,
        store: SnapshotStore,
        client: Any,
        *,
        season: int,
        now: Callable[[], datetime] | None = None,
    ) -> BallDontLieInjuriesAdapter:
        def fetch() -> dict[str, Any]:
            records: list[dict[str, Any]] = []
            for page in client.iter_paid_records("/v1/player_injuries", season):
                records.extend(page)
            return {"data": records}

        return cls(store, fetch=fetch, now=now)


class SportsDataIoAdapter(ProspectiveAdapter):
    """Placeholder for a SportsDataIO availability feed.

    Deliberately inert. SportsDataIO publishes a timestamped injury feed that
    would be a genuinely useful second source, but this project has no
    subscription and nothing about the contract has been verified against live
    responses. Enabling it on the strength of the vendor's documentation alone
    would put unverified data into the availability store, so it stays disabled
    until someone supplies a key and confirms the payload shape.
    """

    source_name = "sportsdataio_injuries"
    enabled = False
    disabled_reason = (
        "No subscription and no verified response shape. Supply a key, capture a "
        "live payload, confirm the as-of semantics, then set enabled=True."
    )

    def __init__(
        self,
        store: SnapshotStore,
        *,
        fetch: SnapshotFetcher | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(store, fetch=fetch or _refuse, now=now)


def _refuse() -> Any:
    raise NotImplementedError(SportsDataIoAdapter.disabled_reason)


def capture_all(adapters: Iterable[ProspectiveAdapter]) -> list[CaptureOutcome]:
    """Run every adapter, letting one failure not stop the others."""
    return [adapter.capture() for adapter in adapters]
