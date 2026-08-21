"""Deterministic parser for official NBA injury-report PDFs.

The PDFs carry a real text layer, so **no OCR is used**. The report is a fixed
seven-column table:

    Game Date | Game Time | Matchup | Team | Player Name | Current Status | Reason

Parsing works from **glyph coordinates**, not from layout-mode character
columns. That choice is forced by the documents themselves:

* **The header row is printed on page 1 only.** Pages 2..N continue the table
  with no header of their own.
* **Layout-mode extraction rescales the character grid per page**, so page 2's
  column positions bear no relation to page 1's. An earlier character-offset
  implementation therefore parsed page 1 and silently dropped every later page
  -- 15 entries out of ~120, with no warning. Absolute glyph coordinates are
  identical across pages, so header anchors taken from page 1 apply to all.

Three further quirks, all observed in real reports:

* **The page content matrix is a vertical flip** (``0 -1`` with a 595.35
  offset), so *increasing* text-space y is the visual reading order: title,
  header row, then data rows top to bottom, page marker last.
* **Group columns print once.** Game date, time, matchup and team appear on the
  first row of each block and are blank on the rest, so they are carried
  forward -- across page boundaries too, since a team's block can span pages.
* **Reasons wrap.** A continuation row has only the reason column filled and
  belongs to the player above it.

Column anchors are read from the header row rather than hard-coded, so a layout
shift is absorbed instead of silently mis-slicing every field.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

from pypdf import PdfReader

from nba_prediction_market.availability.events import normalize_status
from nba_prediction_market.availability.nba_official import EASTERN

logger = logging.getLogger(__name__)

#: Column labels in order. Header detection requires all of them.
COLUMNS: Final[tuple[str, ...]] = (
    "game_date", "game_time", "matchup", "team", "player_name", "status", "reason",
)
_HEADER_TOKENS: Final[tuple[str, ...]] = (
    "Game", "Date", "Time", "Matchup", "Team", "Player", "Name", "Current", "Status", "Reason",
)

_TIMESTAMP = re.compile(
    r"Injury\s*Report:\s*(\d{2}/\d{2}/\d{2})\s+(\d{2}:\d{2})\s*(AM|PM)", re.IGNORECASE
)
_MATCHUP = re.compile(r"^([A-Z]{2,3})@([A-Z]{2,3})$")
_GAME_DATE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_SPACES = re.compile(r"\s+")

#: Statuses the league actually prints. Anything else is preserved raw and
#: normalizes to ``unknown``.
KNOWN_STATUSES: Final[frozenset[str]] = frozenset(
    {"available", "probable", "questionable", "doubtful", "out", "not yet submitted"}
)


class ReportParseError(RuntimeError):
    """Raised when a report cannot be parsed through its text layer."""


def _tidy(value: str) -> str:
    """Collapse the extra spaces layout mode inserts inside words."""
    return _SPACES.sub(" ", value).strip()


def _tidy_name(value: str) -> str:
    """Tidy a player name, repairing two artifacts of glyph extraction.

    A hyphenated surname is drawn as separate chunks ("Caldwell-" then "Pope"),
    and joining them on whitespace yields "Caldwell- Pope", which matches no
    player. No real name carries a space after an internal hyphen, so closing
    it is safe -- and it is a repair of our own join, not a fuzzy match.
    """
    text = _tidy(value)
    text = re.sub(r",\s*", ", ", text)
    return re.sub(r"(\w)-\s+(\w)", r"\1-\2", text)


@dataclass
class PlayerEntry:
    """One player row from a report."""

    game_date: str | None
    game_time_et: str | None
    matchup: str | None
    away_team: str | None
    home_team: str | None
    team: str
    player_name: str
    status_raw: str
    status_normalized: str
    reason_raw: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_date": self.game_date,
            "game_time_et": self.game_time_et,
            "matchup": self.matchup,
            "away_team": self.away_team,
            "home_team": self.home_team,
            "team": self.team,
            "player_name": self.player_name,
            "status_raw": self.status_raw,
            "status_normalized": self.status_normalized,
            "reason_raw": self.reason_raw,
        }


@dataclass(frozen=True)
class NotSubmitted:
    """A team whose report was still outstanding for one specific game."""

    game_date: str | None
    matchup: str | None
    team: str

    def to_dict(self) -> dict[str, Any]:
        return {"game_date": self.game_date, "matchup": self.matchup, "team": self.team}


@dataclass
class ParsedReport:
    """One fully parsed injury report."""

    source_filename: str
    report_timestamp_et: datetime
    report_date: date
    entries: list[PlayerEntry] = field(default_factory=list)
    #: Fingerprint of the header geometry this report was parsed with. Derived
    #: from the observed column offsets rather than assumed, so a report whose
    #: layout differs from the December baseline shows up as a distinct variant
    #: instead of being silently parsed against the wrong columns.
    layout_variant: str = "unknown"
    #: Header column anchors this report was parsed with, kept so geometry
    #: drift can be measured across the archive rather than mistaken for a
    #: layout change.
    column_offsets: tuple[float, ...] = ()
    #: Teams whose filing was outstanding at this timestamp, each tied to the
    #: specific game it was outstanding for. Their players are absent from the
    #: table, which is unknown availability, never "available". One report
    #: covers several dates, so the game context is what makes this joinable --
    #: a team can be pending for tomorrow's game and already filed for today's.
    teams_not_submitted: list[NotSubmitted] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def report_timestamp_utc(self) -> datetime:
        return self.report_timestamp_et.astimezone(UTC)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_filename": self.source_filename,
            "report_timestamp_et": self.report_timestamp_et.isoformat(),
            "report_timestamp_utc": self.report_timestamp_utc.isoformat(),
            "report_date": self.report_date.isoformat(),
            "layout_variant": self.layout_variant,
            "column_offsets": list(self.column_offsets),
            "teams_not_submitted": [n.to_dict() for n in self.teams_not_submitted],
            "entries": len(self.entries),
            "warnings": self.warnings,
        }


HEADER_LABEL_SEQUENCE: Final[tuple[str, ...]] = (
    "Game", "Game", "Matchup", "Team", "Player", "Current", "Reason",
)

#: Glyphs land a hair right of their header label (426.0 vs 425.0), so a cell is
#: assigned to the last anchor at or left of its x, within this slack.
_ANCHOR_SLACK: Final = 2.0

#: Chunks whose text-space y differs by less than this belong to one visual row.
_ROW_TOLERANCE: Final = 1.0

_PAGE_MARKER = re.compile(r"^Page\s+\d+\s+of\s+\d+$", re.IGNORECASE)
_NOT_SUBMITTED = re.compile(r"NOT\s+YET\s+SUBMITTED", re.IGNORECASE)


@dataclass(frozen=True)
class TextRow:
    """One visual row of a report page, as positioned glyph chunks."""

    page: int
    y: float
    cells: tuple[tuple[float, str], ...]

    @property
    def text(self) -> str:
        return _tidy(" ".join(text for _, text in self.cells))


def extract_page_rows(page: Any, page_number: int) -> list[TextRow]:
    """Group one page's glyphs into visual rows, in reading order.

    Rows are returned by increasing text-space y, which the page's flipping
    content matrix makes equivalent to reading top to bottom.
    """
    chunks: list[tuple[float, float, str]] = []

    def visitor(text: str, cm: Any, tm: Any, font_dict: Any, font_size: Any) -> None:
        stripped = text.strip()
        if stripped:
            chunks.append((float(tm[4]), float(tm[5]), stripped))

    page.extract_text(visitor_text=visitor)

    rows: list[TextRow] = []
    for _, y, _text in sorted(chunks, key=lambda c: (c[1], c[0])):
        if rows and abs(rows[-1].y - y) < _ROW_TOLERANCE:
            continue
        group = [c for c in chunks if abs(c[1] - y) < _ROW_TOLERANCE]
        rows.append(
            TextRow(
                page=page_number,
                y=y,
                cells=tuple((x, text) for x, _y, text in sorted(group)),
            )
        )
    return rows


def header_anchors(row: TextRow) -> list[float] | None:
    """Column x-anchors from a header row, or None if this is not one."""
    anchors: list[float] = []
    remaining = list(row.cells)
    for label in HEADER_LABEL_SEQUENCE:
        for index, (x, text) in enumerate(remaining):
            if text == label:
                anchors.append(x)
                remaining = remaining[index + 1:]
                break
        else:
            return None
    return anchors


def layout_fingerprint(anchors: list[float]) -> str:
    """Name the layout by its column *structure*, not its exact geometry.

    Anchor positions shift slightly between reports; what defines the layout is
    the column count. Per-report anchors are read from that report's own header,
    so drift is handled correctly regardless.
    """
    return f"columnar_v1_{len(anchors)}col"


def _row_fields(row: TextRow, anchors: list[float]) -> dict[str, str]:
    """Bucket a row's chunks into the seven named columns."""
    buckets: list[list[str]] = [[] for _ in anchors]
    for x, text in row.cells:
        index = bisect_right(anchors, x + _ANCHOR_SLACK) - 1
        if index < 0:
            index = 0
        buckets[index].append(text)
    return {name: _tidy(" ".join(buckets[i])) for i, name in enumerate(COLUMNS)}


def parse_report_rows(
    rows: list[TextRow], source_filename: str
) -> ParsedReport:
    """Parse a report from its glyph rows, in page then reading order."""
    if not rows:
        raise ReportParseError(f"{source_filename}: no text layer content")

    match = _TIMESTAMP.search(_tidy(" ".join(row.text for row in rows)))
    if not match:
        raise ReportParseError(f"{source_filename}: no report timestamp found")
    stamp = datetime.strptime(
        f"{match.group(1)} {match.group(2)} {match.group(3).upper()}", "%m/%d/%y %I:%M %p"
    ).replace(tzinfo=EASTERN)

    anchors: list[float] | None = None
    header_position: int | None = None
    for index, row in enumerate(rows):
        found = header_anchors(row)
        if found is not None:
            anchors, header_position = found, index
            break
    if anchors is None or header_position is None:
        raise ReportParseError(f"{source_filename}: header row not found")

    report = ParsedReport(
        source_filename=source_filename,
        report_timestamp_et=stamp,
        report_date=stamp.date(),
        layout_variant=layout_fingerprint(anchors),
        column_offsets=tuple(round(a, 1) for a in anchors),
    )

    current: dict[str, str | None] = {
        "game_date": None, "game_time": None, "matchup": None, "team": None,
    }
    for row in rows[header_position + 1:]:
        line = row.text
        if _PAGE_MARKER.match(line) or _TIMESTAMP.search(line):
            continue
        # Pages 2..N repeat neither header nor anything else structural, so a
        # row that looks like a header again is simply skipped.
        if header_anchors(row) is not None:
            continue

        fields = _row_fields(row, anchors)
        if fields["game_date"] and _GAME_DATE.match(fields["game_date"]):
            current["game_date"] = fields["game_date"]
        if fields["game_time"]:
            current["game_time"] = fields["game_time"].replace("(ET)", "").strip()
        if fields["matchup"] and _MATCHUP.match(fields["matchup"]):
            current["matchup"] = fields["matchup"]
        if fields["team"]:
            current["team"] = fields["team"]

        player = _tidy_name(fields["player_name"])
        status = fields["status"]
        reason = fields["reason"]

        # A team that has not filed yet prints the marker with no player. That
        # is a statement about the team, not a wrapped reason for whoever came
        # before, so it must never be appended to the previous entry.
        if not player and _NOT_SUBMITTED.search(f"{status} {reason}"):
            if current["team"]:
                pending = NotSubmitted(
                    game_date=current["game_date"],
                    matchup=current["matchup"],
                    team=current["team"],
                )
                if pending not in report.teams_not_submitted:
                    report.teams_not_submitted.append(pending)
            continue

        if not player:
            # A continuation row carries only more reason text.
            if reason and report.entries:
                report.entries[-1].reason_raw = _tidy(
                    f"{report.entries[-1].reason_raw} {reason}"
                )
            elif status and not reason and report.entries:
                report.entries[-1].reason_raw = _tidy(
                    f"{report.entries[-1].reason_raw} {status}"
                )
            continue

        if not current["team"]:
            report.warnings.append(f"player {player!r} appeared before any team")
            continue

        away, home = (None, None)
        if current["matchup"]:
            parts = _MATCHUP.match(current["matchup"])
            away, home = parts.group(1), parts.group(2)

        normalized = normalize_status(status)
        if status and status.strip().casefold() not in KNOWN_STATUSES:
            report.warnings.append(f"unrecognised status {status!r} for {player!r}")
        report.entries.append(
            PlayerEntry(
                game_date=current["game_date"],
                game_time_et=current["game_time"],
                matchup=current["matchup"],
                away_team=away,
                home_team=home,
                team=current["team"],
                player_name=player,
                status_raw=status or None,
                status_normalized=normalized,
                reason_raw=reason,
            )
        )
    return report


def parse_report_pdf(path: Path) -> ParsedReport:
    """Parse one archived report PDF through its embedded text layer."""
    path = Path(path)
    try:
        reader = PdfReader(str(path))
        rows: list[TextRow] = []
        for number, page in enumerate(reader.pages, start=1):
            rows.extend(extract_page_rows(page, number))
    except Exception as exc:  # pragma: no cover - pypdf raises many types
        raise ReportParseError(f"{path.name}: could not read PDF ({exc})") from exc
    return parse_report_rows(rows, path.name)
