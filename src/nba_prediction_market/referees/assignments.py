"""Prospective capture of the NBA's published referee assignments.

The league posts the day's crews at official.nba.com/referee-assignments/
around 9:00 AM ET, hours before this project's earliest T-6h anchor. That
makes the crew genuinely *pregame* information rather than another rearranged
box score, which is the whole reason this family is worth testing.

Three properties this module exists to guarantee:

* **As-of semantics.** Every crew carries the moment we first observed it. A
  crew discovered at 14:00 is not information that existed at 09:00, and it is
  never backfilled into an earlier anchor.
* **Changes are additive.** Assignments do change during the day. Both states
  are kept, each with its own first-observed time, so a prediction can be
  reconstructed from what was actually known at its cutoff.
* **Optional by construction.** Nothing here is on the critical path of the
  October collector. A failure to capture assignments is recorded and the
  slate carries on.

Unlike the historical Basketball-Reference source, this page **does** carry
positions -- Crew Chief, Referee, Umpire, Alternate -- so capture starts
recording them now even though no historical model can use them yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

#: The published page. No query string: official.nba.com's robots.txt
#: disallows ``/*?*``, so the date-filtered form of this page is off limits.
#: That is also why no *historical* archive is retrieved from here.
ASSIGNMENTS_URL: Final = "https://official.nba.com/referee-assignments/"

#: Their stated Crawl-Delay, with margin.
CRAWL_DELAY_SECONDS: Final = 1.5

#: Roughly when the league posts. Used only to describe expectations in health
#: output -- never to assume a crew exists before it has been observed.
TYPICAL_PUBLICATION_HOUR_ET: Final = 9

POSITIONS: Final[tuple[str, ...]] = ("crew_chief", "referee", "umpire", "alternate")

_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
_TAG = re.compile(r"<[^>]+>")


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", _TAG.sub(" ", html)).replace("&nbsp;", " ").strip()


@dataclass(frozen=True)
class AssignmentRow:
    """One game's published crew, as observed at a moment in time."""

    matchup: str
    crew_chief: str | None
    referee: str | None
    umpire: str | None
    alternate: str | None
    first_observed_at_utc: datetime
    source_date_et: str | None = None

    @property
    def officials(self) -> dict[str, str | None]:
        return {
            "crew_chief": self.crew_chief,
            "referee": self.referee,
            "umpire": self.umpire,
            "alternate": self.alternate,
        }

    @property
    def assigned_names(self) -> tuple[str, ...]:
        """Named officials excluding the alternate, who does not work the game."""
        return tuple(
            n for n in (self.crew_chief, self.referee, self.umpire) if n
        )

    def identity(self) -> tuple[Any, ...]:
        """What makes this a *different* assignment, ignoring when we saw it."""
        return (self.matchup, self.crew_chief, self.referee,
                self.umpire, self.alternate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matchup": self.matchup,
            **self.officials,
            "first_observed_at_utc": self.first_observed_at_utc.isoformat(),
            "source_date_et": self.source_date_et,
        }


def parse_assignments(
    html: str, *, observed_at_utc: datetime, source_date_et: str | None = None
) -> list[AssignmentRow]:
    """Rows from the published table.

    An empty table is a normal state -- there are no assignments in the
    offseason, and none before the league posts them in the morning -- so it
    returns an empty list rather than raising. The caller distinguishes "no
    games" from "fetch failed"; this function only reports what the page said.
    """
    rows: list[AssignmentRow] = []
    for raw in _ROW.findall(html):
        cells = [_text(c) for c in _CELL.findall(raw)]
        if len(cells) < 4:
            continue
        if cells[0].lower() == "game":  # header
            continue
        matchup = cells[0]
        if not matchup:
            continue
        values = (cells + [""] * 5)[1:5]
        crew_chief, referee, umpire, alternate = (v or None for v in values)
        rows.append(AssignmentRow(
            matchup=matchup, crew_chief=crew_chief, referee=referee,
            umpire=umpire, alternate=alternate,
            first_observed_at_utc=observed_at_utc,
            source_date_et=source_date_et,
        ))
    return rows


CHANGE_NEW: Final = "new"
CHANGE_UNCHANGED: Final = "unchanged"
CHANGE_REASSIGNED: Final = "reassigned"
CHANGE_DISAPPEARED: Final = "disappeared"


@dataclass(frozen=True)
class AssignmentChange:
    matchup: str
    change: str
    previous: AssignmentRow | None
    current: AssignmentRow | None
    detected_at_utc: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "matchup": self.matchup,
            "change": self.change,
            "previous": self.previous.to_dict() if self.previous else None,
            "current": self.current.to_dict() if self.current else None,
            "detected_at_utc": self.detected_at_utc.isoformat(),
        }


@dataclass
class AssignmentLedger:
    """Every observed state of every crew, in the order they were observed.

    Never overwrites. A reassignment appends a second state and keeps the
    first, because the earlier state is what an earlier anchor legitimately
    saw and deleting it would make that prediction unreconstructable.
    """

    states: dict[str, list[AssignmentRow]] = field(default_factory=dict)

    def observe(
        self, rows: list[AssignmentRow], *, now: datetime
    ) -> list[AssignmentChange]:
        changes: list[AssignmentChange] = []
        seen: set[str] = set()
        for row in rows:
            seen.add(row.matchup)
            history = self.states.setdefault(row.matchup, [])
            if not history:
                history.append(row)
                changes.append(AssignmentChange(
                    row.matchup, CHANGE_NEW, None, row, now
                ))
                continue
            latest = history[-1]
            if latest.identity() == row.identity():
                continue  # same crew re-observed; first_observed stands
            history.append(row)
            changes.append(AssignmentChange(
                row.matchup, CHANGE_REASSIGNED, latest, row, now
            ))
        for matchup, history in self.states.items():
            if matchup not in seen and history:
                changes.append(AssignmentChange(
                    matchup, CHANGE_DISAPPEARED, history[-1], None, now
                ))
        return changes

    def known_at(self, matchup: str, cutoff: datetime) -> AssignmentRow | None:
        """The latest crew observed at or before ``cutoff``.

        This is the only accessor a prediction may use. A crew first observed
        after the cutoff is invisible here, which is what stops a late
        discovery from being backfilled into an earlier anchor.
        """
        eligible = [
            row for row in self.states.get(matchup, [])
            if row.first_observed_at_utc <= cutoff
        ]
        return eligible[-1] if eligible else None

    def was_known_at(self, matchup: str, cutoff: datetime) -> bool:
        return self.known_at(matchup, cutoff) is not None

    def summary(self) -> dict[str, Any]:
        return {
            "matchups": len(self.states),
            "reassigned": sum(1 for h in self.states.values() if len(h) > 1),
            "total_states": sum(len(h) for h in self.states.values()),
        }
