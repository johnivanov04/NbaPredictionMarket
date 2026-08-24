"""Reconciling the injury reports against the trusted schedule.

Two things make a scheduled game have no usable report rows, and they are not
the same: the game was **postponed** (the reports describe a date it was not
played on), or the league simply **omitted** it from its reports. Both are
recorded here with independent evidence, so a gap is classified rather than
left looking like a parser defect.

## Postponements

Phase 3A3B1 recorded four report-vs-schedule mismatches without resolving them.
All four turned out to be the same thing, and **neither source was wrong**:

* the official injury report describes the date a game was *originally
  scheduled* for, and keeps describing it right through that evening;
* the trusted schedule holds the date the game was actually *played*.

A postponement makes those differ. Verified independently against the ESPN
scoreboard API, which reports the original date with ``STATUS_POSTPONED`` and
the replay date with ``STATUS_FINAL``.

The modelling consequence is the point of this module. Availability rows filed
against a postponed occurrence describe a game that did not happen, observed in
a different context -- two of these were replayed weeks later. They must not be
attached to the replayed game, so they stay unmatched. Recording them here
turns that from an unexplained gap into a classified, evidenced one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Final

#: Independent source used to establish every entry below.
ESPN_SCOREBOARD: Final = "ESPN public scoreboard API"


@dataclass(frozen=True)
class Postponement:
    """One postponed occurrence and the game that eventually replaced it."""

    original_date: date
    away_team: str
    home_team: str
    replayed_date: date
    evidence: str
    source: str = ESPN_SCOREBOARD

    @property
    def key(self) -> tuple[str, str, str]:
        """Matches the ``(game_date, away, home)`` key reports are joined on."""
        return (self.original_date.isoformat(), self.away_team, self.home_team)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_date": self.original_date.isoformat(),
            "away_team": self.away_team,
            "home_team": self.home_team,
            "replayed_date": self.replayed_date.isoformat(),
            "evidence": self.evidence,
            "source": self.source,
            "modeling_treatment": (
                "rows filed against the original date are not attached to the "
                "replayed game; they describe an occurrence that did not happen"
            ),
        }


#: 2021-22 and the COVID-era rescheduling that followed. Each was found by
#: sweeping every unmatched report key through the ESPN scoreboard, then taking
#: the replay date from the schedule: for most of these the source keeps the
#: original ``date`` and moves ``game_datetime_utc`` to the replay, so the
#: retained row names both dates at once.
POSTPONEMENTS_COVID_ERA: Final[tuple[Postponement, ...]] = (
    Postponement(
        date(2021, 12, 19), "CLE", "ATL", date(2022, 3, 31),
        "ESPN 2021-12-19 lists CLE@ATL STATUS_POSTPONED; the schedule row still "
        "dated 2021-12-19 carries a 2022-03-31 tipoff",
    ),
    Postponement(
        date(2021, 12, 19), "DEN", "BKN", date(2022, 1, 26),
        "ESPN 2021-12-19 lists DEN@BKN STATUS_POSTPONED; the schedule row still "
        "dated 2021-12-19 carries a 2022-01-26 tipoff",
    ),
    Postponement(
        date(2021, 12, 19), "NOP", "PHI", date(2022, 1, 25),
        "ESPN 2021-12-19 lists NO@PHI STATUS_POSTPONED; the schedule row still "
        "dated 2021-12-19 carries a 2022-01-25 tipoff",
    ),
    Postponement(
        date(2021, 12, 22), "TOR", "CHI", date(2022, 1, 26),
        "ESPN 2021-12-22 lists TOR@CHI STATUS_POSTPONED; the schedule row still "
        "dated 2021-12-22 carries a 2022-01-26 tipoff",
    ),
    Postponement(
        date(2021, 12, 29), "MIA", "SAS", date(2022, 2, 3),
        "ESPN 2021-12-29 lists MIA@SA STATUS_POSTPONED; the schedule row still "
        "dated 2021-12-29 carries a 2022-02-03 tipoff",
    ),
    Postponement(
        date(2021, 12, 30), "GSW", "DEN", date(2022, 3, 7),
        "ESPN 2021-12-30 lists GS@DEN STATUS_POSTPONED; the schedule row still "
        "dated 2021-12-30 carries a 2022-03-07 tipoff",
    ),
    Postponement(
        date(2023, 2, 1), "WAS", "DET", date(2023, 3, 7),
        "ESPN 2023-02-01 lists WSH@DET STATUS_POSTPONED; the schedule row still "
        "dated 2023-02-01 carries a 2023-03-07 tipoff",
    ),
    Postponement(
        date(2024, 1, 17), "GSW", "UTA", date(2024, 2, 12),
        "ESPN 2024-01-17 lists GS@UTAH STATUS_POSTPONED; here the source moved "
        "both date and tipoff, and the next scheduled meeting is 2024-02-12",
    ),
)

#: 2024-25. Five games were postponed in January 2025; the Los Angeles cluster
#: coincides with the wildfires that month. Each verified the same way.
POSTPONEMENTS_2024_25: Final[tuple[Postponement, ...]] = (
    Postponement(
        date(2025, 1, 9), "CHA", "LAL", date(2025, 2, 19),
        "ESPN 2025-01-09 lists CHA@LAL STATUS_POSTPONED; the trusted schedule "
        "carries the matchup only on 2025-02-19",
    ),
    Postponement(
        date(2025, 1, 11), "CHA", "LAC", date(2025, 3, 16),
        "ESPN 2025-01-11 lists CHA@LAC STATUS_POSTPONED; the trusted schedule "
        "carries the matchup only on 2025-03-16",
    ),
    Postponement(
        date(2025, 1, 11), "SAS", "LAL", date(2025, 1, 13),
        "ESPN 2025-01-11 lists SA@LAL STATUS_POSTPONED; replayed two days later "
        "on 2025-01-13, which the trusted schedule carries",
    ),
    Postponement(
        date(2025, 1, 11), "HOU", "ATL", date(2025, 1, 28),
        "ESPN 2025-01-11 lists HOU@ATL STATUS_POSTPONED; the trusted schedule "
        "carries the matchup only on 2025-01-28",
    ),
    Postponement(
        date(2025, 1, 22), "MIL", "NOP", date(2025, 4, 6),
        "ESPN 2025-01-22 lists MIL@NO STATUS_POSTPONED; the trusted schedule "
        "carries the matchup only on 2025-04-06",
    ),
)

#: 2025-26.
POSTPONEMENTS_2025_26: Final[tuple[Postponement, ...]] = (
    Postponement(
        date(2026, 1, 8), "MIA", "CHI", date(2026, 1, 29),
        "ESPN 2026-01-08 lists MIA@CHI STATUS_POSTPONED; 2026-01-29 lists it "
        "STATUS_FINAL, which is the date the trusted schedule carries",
    ),
    Postponement(
        date(2026, 1, 24), "GSW", "MIN", date(2026, 1, 25),
        "ESPN 2026-01-24 lists GS@MIN STATUS_POSTPONED; 2026-01-25 lists it "
        "STATUS_FINAL. The reports corroborate the correction directly: the "
        "01/24 listing stops appearing between 19:30Z and 20:00Z on 2026-01-24 "
        "and is replaced by an 01/25 listing at the same 05:30 PM tip time",
    ),
    Postponement(
        date(2026, 1, 25), "DEN", "MEM", date(2026, 3, 18),
        "ESPN 2026-01-25 lists DEN@MEM STATUS_POSTPONED; 2026-03-18 lists it "
        "STATUS_FINAL, matching the trusted schedule",
    ),
    Postponement(
        date(2026, 1, 25), "DAL", "MIL", date(2026, 3, 31),
        "ESPN 2026-01-25 lists DAL@MIL STATUS_POSTPONED; 2026-03-31 lists it "
        "STATUS_FINAL, matching the trusted schedule",
    ),
)

POSTPONEMENTS: Final[tuple[Postponement, ...]] = (
    POSTPONEMENTS_COVID_ERA + POSTPONEMENTS_2024_25 + POSTPONEMENTS_2025_26
)

POSTPONED_KEYS: Final[dict[tuple[str, str, str], Postponement]] = {
    p.key: p for p in POSTPONEMENTS
}


def postponement_for(
    game_date: str, away_team: str, home_team: str
) -> Postponement | None:
    """The postponement matching a report's game key, if there is one."""
    return POSTPONED_KEYS.get((game_date, away_team, home_team))


@dataclass(frozen=True)
class ReportOmission:
    """A game that was played but which the league never put in a report."""

    game_date: date
    away_team: str
    home_team: str
    evidence: str
    source: str = ESPN_SCOREBOARD

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.game_date.isoformat(), self.away_team, self.home_team)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_date": self.game_date.isoformat(),
            "away_team": self.away_team,
            "home_team": self.home_team,
            "evidence": self.evidence,
            "source": self.source,
            "modeling_treatment": (
                "no availability state exists for this game at any anchor; it "
                "is reported as uncovered and never filled from another game"
            ),
        }


#: Games the reports never covered. A gap in the source, not in the parser.
REPORT_OMISSIONS: Final[tuple[ReportOmission, ...]] = (
    ReportOmission(
        date(2023, 10, 25), "BOS", "NYK",
        "ESPN lists BOS@NY on 2023-10-25 as STATUS_FINAL, one of twelve games "
        "that day, and the trusted schedule agrees. The injury reports carry "
        "eleven of those twelve and omit this one: none of the 55 reports "
        "archived across 2023-10-23..26 mentions the matchup at all, in "
        "entries or in not-yet-submitted markers",
    ),
)

OMITTED_KEYS: Final[dict[tuple[str, str, str], ReportOmission]] = {
    o.key: o for o in REPORT_OMISSIONS
}


def omission_for(
    game_date: str, away_team: str, home_team: str
) -> ReportOmission | None:
    """The recorded omission for a game key, if there is one."""
    return OMITTED_KEYS.get((game_date, away_team, home_team))
