"""Third-party availability archives, normalized onto one schema.

Two public datasets were audited as possible ways to extend availability
history backwards. They are kept here behind a single normalized event shape so
that source-specific quirks stay visible rather than being flattened away.

The distinction this module exists to enforce is **as-of class**: whether a
record can answer "what was known 30 minutes before tip". That is a property of
the source's timestamp, not of its size or its coverage.

* ``T30_SAFE`` -- carries an exact report timestamp, so the latest observation
  at or before an anchor can be selected. Safe does not mean *fresh*: a source
  publishing five snapshots a day is T-30 safe but often hours stale.
* ``EARLY_DAY_ONLY`` -- a fixed early-in-the-day snapshot. Usable for a morning
  anchor, never a substitute for T-30.
* ``DATE_ONLY`` -- a calendar date and nothing finer. Which snapshot it came
  from is unknown, so no anchor can be answered from it.
* ``UNSAFE_FINAL_STATE`` -- reflects the eventual outcome.
* ``UNAVAILABLE`` -- no usable availability content.

**A date-only record is never promoted to an anchor state**, and a status is
never carried forward from an early snapshot as though it still held at tip.
Forward-filling would manufacture exactly the certainty the anchor rule exists
to deny.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from nba_prediction_market.availability.nba_official import (
    EASTERN,
    LEGACY_SLOT_MINUTE,
)

T30_SAFE: Final = "T30_SAFE"
EARLY_DAY_ONLY: Final = "EARLY_DAY_ONLY"
DATE_ONLY: Final = "DATE_ONLY"
UNSAFE_FINAL_STATE: Final = "UNSAFE_FINAL_STATE"
UNAVAILABLE: Final = "UNAVAILABLE"

PRECISION_EXACT: Final = "exact_timestamp"
PRECISION_DATE_ONLY: Final = "date_only"

#: Redistribution posture. Nothing external is committed to the repository;
#: raw downloads live under the gitignored data tree.
LICENSE_NONE_DECLARED: Final = "no licence declared - treat as all rights reserved"
LICENSE_UNSTATED: Final = "no terms stated on the distribution page"


@dataclass(frozen=True)
class ExternalSource:
    """An audited third-party availability archive."""

    name: str
    origin: str
    seasons: str
    asof_class: str
    timestamp_precision: str
    license_status: str
    redistributable: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.name,
            "origin": self.origin,
            "seasons": self.seasons,
            "asof_class": self.asof_class,
            "timestamp_precision": self.timestamp_precision,
            "license_status": self.license_status,
            "redistributable": self.redistributable,
            "notes": self.notes,
        }


PROFESSOR_PETE = ExternalSource(
    name="professor_pete_2024_25",
    origin="github.com/Professor-Pete/2024-2025-injury_analysis",
    seasons="2024-25",
    asof_class=T30_SAFE,
    timestamp_precision=PRECISION_EXACT,
    license_status=LICENSE_NONE_DECLARED,
    redistributable=False,
    notes=(
        "Derived CSV only; no PDFs retained. Keeps snapshot_date and the "
        "filename slot, so an exact report timestamp is recoverable. Covers "
        "five slots a day (01PM/03PM/05PM/08PM/11PM) out of the 24 the league "
        "published, so it is T-30 safe but materially staler than the source."
    ),
)

STATSURGE = ExternalSource(
    name="statsurge_2021_24",
    origin="statsurge.substack.com downloadable NBA injury datasets",
    seasons="2021-22, 2022-23, 2023-24",
    asof_class=DATE_ONLY,
    timestamp_precision=PRECISION_DATE_ONLY,
    license_status=LICENSE_UNSTATED,
    redistributable=False,
    notes=(
        "Schema is PLAYER, STATUS, REASON, TEAM, GAME, DATE -- there is no "
        "time field of any kind, and exactly one row per player and game. The "
        "publisher describes an approximately 2 PM report, but that claim "
        "cannot be checked from the file itself because no timestamp is "
        "stored, so the data is classed by what it carries, not by the claim."
    ),
)

EXTERNAL_SOURCES: Final[tuple[ExternalSource, ...]] = (PROFESSOR_PETE, STATSURGE)


def legacy_slot_timestamp(report_date: str, slot: str) -> datetime:
    """Exact Eastern timestamp for a legacy filename slot such as ``05PM``.

    The filename carries only the hour; the report inside is stamped at
    :30 past it. Reading ``05PM`` as 17:00 would place the observation half an
    hour earlier than it happened, which is the direction that lets a report
    published *after* an anchor be accepted for it.
    """
    hour_text, meridiem = slot[:2], slot[2:].upper()
    if meridiem not in ("AM", "PM"):
        raise ValueError(f"unrecognised slot {slot!r}")
    hour = int(hour_text) % 12 + (12 if meridiem == "PM" else 0)
    year, month, day = (int(part) for part in report_date.split("-"))
    return datetime(year, month, day, hour, LEGACY_SLOT_MINUTE, tzinfo=EASTERN)


@dataclass(frozen=True)
class ExternalEvent:
    """One availability observation from a third-party archive."""

    source: str
    source_report_id: str
    observed_at_utc: datetime | None
    timestamp_precision: str
    asof_class: str
    game_date: str
    away_team: str | None
    home_team: str | None
    team_raw: str
    player_name_raw: str
    status_raw: str
    status_normalized: str
    reason_raw: str

    def usable_for_anchor(self, anchor_utc: datetime) -> bool:
        """Whether this observation may answer an anchor at all.

        Requires both an exact timestamp and that the observation precede the
        anchor. A date-only record returns False whatever the anchor, which is
        what stops it standing in for a T-30 state.
        """
        if self.asof_class != T30_SAFE or self.observed_at_utc is None:
            return False
        return self.observed_at_utc <= anchor_utc

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_report_id": self.source_report_id,
            "observed_at_utc": (
                self.observed_at_utc.isoformat() if self.observed_at_utc else None
            ),
            "timestamp_precision": self.timestamp_precision,
            "asof_class": self.asof_class,
            "game_date": self.game_date,
            "away_team": self.away_team,
            "home_team": self.home_team,
            "team_raw": self.team_raw,
            "player_name_raw": self.player_name_raw,
            "status_raw": self.status_raw,
            "status_normalized": self.status_normalized,
            "reason_raw": self.reason_raw,
        }


def _split_matchup(matchup: str) -> tuple[str | None, str | None]:
    if "@" not in matchup:
        return (None, None)
    away, home = matchup.split("@", 1)
    return (away.strip() or None, home.strip() or None)


def _iso_game_date(text: str) -> str:
    month, day, year = text.split("/")
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def professor_pete_events(rows: Any) -> list[ExternalEvent]:
    """Normalize Professor-Pete ``season_rows.csv`` records.

    Team-level "NOT YET SUBMITTED" markers carry no player and are skipped
    here; they describe a filing state, not a player's availability.
    """
    from nba_prediction_market.availability.events import normalize_status

    out: list[ExternalEvent] = []
    for row in rows:
        player = (row.get("player") or "").strip()
        if not player:
            continue
        slot = row["snapshot_time"]
        observed = legacy_slot_timestamp(row["snapshot_date"], slot)
        away, home = _split_matchup(row.get("matchup", ""))
        status = (row.get("status") or "").strip()
        out.append(
            ExternalEvent(
                source=PROFESSOR_PETE.name,
                source_report_id=f"{row['snapshot_date']}_{slot}",
                observed_at_utc=observed.astimezone(UTC),
                timestamp_precision=PRECISION_EXACT,
                asof_class=T30_SAFE,
                game_date=_iso_game_date(row["game_date"]),
                away_team=away,
                home_team=home,
                team_raw=(row.get("team") or "").strip(),
                player_name_raw=player,
                status_raw=status,
                status_normalized=normalize_status(status),
                reason_raw=(row.get("reason") or "").strip(),
            )
        )
    return out


def statsurge_events(rows: Any) -> list[ExternalEvent]:
    """Normalize StatSurge records.

    ``observed_at_utc`` is deliberately left ``None``. The file has no time
    field, so any timestamp here would be invented -- and an invented timestamp
    is precisely what would let a date-only record be mistaken for an anchor
    observation.
    """
    from nba_prediction_market.availability.events import normalize_status

    out: list[ExternalEvent] = []
    for row in rows:
        player = (row.get("PLAYER") or "").strip()
        if not player:
            continue
        away, home = _split_matchup(row.get("GAME", ""))
        game_date = _iso_game_date(row["DATE"])
        status = (row.get("STATUS") or "").strip()
        out.append(
            ExternalEvent(
                source=STATSURGE.name,
                source_report_id=f"{game_date}_{row.get('GAME', '')}",
                observed_at_utc=None,
                timestamp_precision=PRECISION_DATE_ONLY,
                asof_class=DATE_ONLY,
                game_date=game_date,
                away_team=away,
                home_team=home,
                team_raw=(row.get("TEAM") or "").strip(),
                player_name_raw=player,
                status_raw=status,
                status_normalized=normalize_status(status),
                reason_raw=(row.get("REASON") or "").strip(),
            )
        )
    return out
