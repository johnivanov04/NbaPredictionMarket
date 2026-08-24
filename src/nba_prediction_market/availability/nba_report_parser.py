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


def _rejoin_hyphens(value: str) -> str:
    """Close a hyphen that glyph extraction split across chunks.

    "Two-Way" arrives as "Two-" and "Way" and joins to "Two- Way". A hyphen
    directly attached to the preceding word never takes a space after it, so
    closing that gap is safe -- and it leaves the report's own " - " separator
    alone, since that hyphen has a space on both sides.
    """
    return re.sub(r"(\w)-\s+(\w)", r"\1-\2", value)


def _tidy_name(value: str) -> str:
    """Tidy a player name, repairing two artifacts of glyph extraction.

    A hyphenated surname is drawn as separate chunks ("Caldwell-" then "Pope"),
    and joining them on whitespace yields "Caldwell- Pope", which matches no
    player. No real name carries a space after an internal hyphen, so closing
    it is safe -- and it is a repair of our own join, not a fuzzy match.
    """
    text = _tidy(value)
    text = re.sub(r",\s*", ", ", text)
    return _rejoin_hyphens(text)


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


@dataclass(frozen=True)
class LayoutVariant:
    """One published column layout, named by its header labels in order."""

    name: str
    labels: tuple[str, ...]
    #: Canonical field each column feeds. ``None`` keeps the column's text
    #: without letting it drive the seven-field row model.
    fields: tuple[str | None, ...]

    def index_of(self, field: str) -> int | None:
        return self.fields.index(field) if field in self.fields else None


#: The seven-column layout published from the 2019-20 season onward.
LAYOUT_V1: Final = LayoutVariant(
    name="columnar_v1_7col",
    labels=("Game Date", "Game Time", "Matchup", "Team",
            "Player Name", "Current Status", "Reason"),
    fields=("game_date", "game_time", "matchup", "team",
            "player_name", "status", "reason"),
)

#: The nine-column layout published in 2018-19. It splits what later became a
#: single Reason string into "Category" plus "Reason", orders Reason before
#: Current Status, and carries a Previous Status column that was later dropped.
LAYOUT_V0: Final = LayoutVariant(
    name="columnar_v0_9col",
    labels=("Game Date", "Game Time", "Matchup", "Team", "Player Name",
            "Category", "Reason", "Current Status", "Previous Status"),
    fields=("game_date", "game_time", "matchup", "team", "player_name",
            "category", "reason", "status", "previous_status"),
)

#: Tried in order; the first whose labels all appear wins.
LAYOUT_VARIANTS: Final[tuple[LayoutVariant, ...]] = (LAYOUT_V0, LAYOUT_V1)

#: Glyphs land a hair right of their header label (426.0 vs 425.0), so a cell is
#: assigned to the last anchor at or left of its x, within this slack.
_ANCHOR_SLACK: Final = 2.0

#: Chunks whose text-space y differs by less than this belong to one visual row.
_ROW_TOLERANCE: Final = 1.0

#: The page marker is not always drawn as one piece. In the portrait-rotated
#: era the trailing page count sits on its own baseline, so the row reads
#: "Page 1 of" -- and on some layouts a stray count leads it ("6 Page 1 of").
#: Requiring the complete "Page 1 of 8" let the fragment fall into the Player
#: column and become a player named "Page 1 of", 34,000 rows of them.
_PAGE_MARKER = re.compile(r"^\d*\s*Page\s+\d+\s+of(\s+\d+)?$", re.IGNORECASE)
#: A row carrying nothing but a number is the page marker's detached count.
_BARE_NUMBER = re.compile(r"^\d{1,3}$")
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


def display_position(cm: Any, tm: Any, rotation: int) -> tuple[float, float]:
    """Map a glyph to ``(column, row)`` keys in the page's *displayed* frame.

    Working in displayed coordinates rather than raw text space is what lets one
    code path read every era of the report. The league has shipped the same
    table under two quite different page geometries:

    * **2024 onward** -- landscape media box, no ``/Rotate``, content matrix a
      vertical flip ``(1,0,0,-1)``.
    * **through 2023** -- *portrait* media box with ``/Rotate 90``, content
      drawn sideways under ``(0,1,-1,0)``. Read without the rotation the table
      comes out transposed: each apparent row is an entire column, so a whole
      report's statuses arrive concatenated into one string.

    Composing the content matrix with the page rotation resolves both. Only the
    ordering of the returned keys is meaningful, not their origin: rotation
    preserves distance, so the row and column tolerances stay valid in points.
    """
    a, b, c, d, e, f = (float(v) for v in cm)
    tx, ty = float(tm[4]), float(tm[5])
    ux = a * tx + c * ty + e
    uy = b * tx + d * ty + f
    # Row keys are returned so that *ascending* order is top-to-bottom reading
    # order, which the rest of the parser assumes.
    match (rotation or 0) % 360:
        case 90:
            return (uy, ux)
        case 180:
            return (-ux, uy)
        case 270:
            return (-uy, -ux)
        case _:
            return (ux, -uy)


def extract_page_rows(page: Any, page_number: int) -> list[TextRow]:
    """Group one page's glyphs into visual rows, in reading order."""
    chunks: list[tuple[float, float, str]] = []
    rotation = int(page.get("/Rotate") or 0)

    def visitor(text: str, cm: Any, tm: Any, font_dict: Any, font_size: Any) -> None:
        stripped = text.strip()
        if stripped:
            column, row = display_position(cm, tm, rotation)
            chunks.append((column, row, stripped))

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


def _squash(value: str) -> str:
    return value.replace(" ", "").casefold()


def _anchors_for_labels(row: TextRow, labels: tuple[str, ...]) -> list[float] | None:
    """x-anchors for a specific label sequence, or None if the row lacks it.

    A label may span several consecutive chunks; the anchor is the x of the
    first chunk that starts it.
    """
    anchors: list[float] = []
    cells = list(row.cells)
    position = 0
    for label in labels:
        target = _squash(label)
        matched = False
        while position < len(cells) and not matched:
            start_x = cells[position][0]
            accumulated = ""
            for end in range(position, len(cells)):
                accumulated += _squash(cells[end][1])
                if accumulated == target:
                    anchors.append(start_x)
                    position = end + 1
                    matched = True
                    break
                if not target.startswith(accumulated):
                    break
            if not matched:
                position += 1
        if not matched:
            return None
    return anchors


def match_header(row: TextRow) -> tuple[LayoutVariant, list[float]] | None:
    """Identify which published layout a header row belongs to.

    The nine-column layout is tried first: its labels are a superset of the
    seven-column one, so testing the shorter sequence first would match a
    2018-19 header and then silently read Category as Current Status.
    """
    for variant in LAYOUT_VARIANTS:
        anchors = _anchors_for_labels(row, variant.labels)
        if anchors is not None:
            return variant, anchors
    return None


def header_anchors(row: TextRow) -> list[float] | None:
    """Column x-anchors from a header row, or None if this is not one."""
    found = match_header(row)
    return found[1] if found else None


def _reason_pieces_by_owner(
    body: list[TextRow], anchors: list[float], variant: LayoutVariant
) -> dict[int, str]:
    """Rebuild each player's Reason cell from individually placed glyphs.

    The Reason cell is drawn vertically centred inside its row, while the
    player name sits on the row's own baseline. For a wrapped reason the two do
    not line up: its first line can be drawn *above* the player it describes
    and its second below, so grouping text into rows by y and then reading a
    Reason column mixes fragments from neighbouring players -- and a single
    grouped row can hold the tail of one reason and the head of the next.

    Assigning each reason glyph to the nearest player baseline on its own page
    reconstructs the cell as drawn. Only the Reason column is treated this way;
    the grouping columns print on their block's first row, not centred, which
    is why they carry forward correctly.
    """
    reason_index = variant.index_of("reason")
    if reason_index is None:
        return {}

    baselines = [
        (index, row.y, row.page)
        for index, row in enumerate(body)
        if _tidy(_row_fields(row, anchors, variant)["player_name"])
    ]
    if not baselines:
        return {}

    collected: dict[int, list[tuple[float, float, str]]] = {}
    for row in body:
        if (_PAGE_MARKER.match(row.text) or _BARE_NUMBER.match(row.text)
                or _TIMESTAMP.search(row.text)):
            continue
        if match_header(row) is not None:
            continue
        # A team's "not yet submitted" marker is drawn in the Reason column but
        # describes the team, not any player. Folding it into the nearest
        # player's reason would attribute a filing state to an individual.
        row_fields = _row_fields(row, anchors, variant)
        if not _tidy(row_fields["player_name"]) and _NOT_SUBMITTED.search(row.text):
            continue
        for x, text in row.cells:
            bucket = bisect_right(anchors, x + _ANCHOR_SLACK) - 1
            if max(bucket, 0) != reason_index:
                continue
            same_page = [b for b in baselines if b[2] == row.page]
            if not same_page:
                continue
            owner = min(same_page, key=lambda b: abs(b[1] - row.y))[0]
            collected.setdefault(owner, []).append((row.y, x, text))

    return {
        owner: _rejoin_hyphens(
            _tidy(" ".join(text for _, _, text in sorted(pieces)))
        )
        for owner, pieces in collected.items()
    }


def _continuation_owners(
    body: list[TextRow], anchors: list[float], variant: LayoutVariant
) -> dict[int, int]:
    """Map each continuation row to the player row that owns it.

    Reason text is drawn vertically centred inside its cell, so a wrapped
    reason straddles the row boundary: one line can sit above the player name
    it belongs to and the next below it. Assigning continuation lines by their
    position in the stream therefore hands the first line to the *previous*
    player and produces two players' reasons welded together.

    Nearest player row by vertical distance recovers the intended owner, and
    ownership never crosses a page boundary.
    """
    player_rows: list[tuple[int, float, int]] = []
    for index, row in enumerate(body):
        fields = _row_fields(row, anchors, variant)
        if _tidy(fields["player_name"]):
            player_rows.append((index, row.y, row.page))
    if not player_rows:
        return {}

    owners: dict[int, int] = {}
    for index, row in enumerate(body):
        fields = _row_fields(row, anchors, variant)
        if _tidy(fields["player_name"]):
            continue
        same_page = [p for p in player_rows if p[2] == row.page]
        if not same_page:
            continue
        nearest = min(same_page, key=lambda p: abs(p[1] - row.y))
        owners[index] = nearest[0]
    return owners


def _is_bare_team_fragment(fields: dict[str, str]) -> bool:
    """True when a row carries a team cell and nothing else at all."""
    return not any(
        fields.get(name)
        for name in ("game_date", "game_time", "matchup", "player_name",
                     "status", "reason", "category", "previous_status")
    )


def _row_fields(
    row: TextRow, anchors: list[float], variant: LayoutVariant
) -> dict[str, str]:
    """Bucket a row's chunks into canonical fields for this layout."""
    buckets: list[list[str]] = [[] for _ in anchors]
    for x, text in row.cells:
        index = bisect_right(anchors, x + _ANCHOR_SLACK) - 1
        buckets[max(index, 0)].append(text)

    fields = dict.fromkeys(COLUMNS, "")
    extra: dict[str, str] = {}
    for position, name in enumerate(variant.fields):
        if name is None or position >= len(buckets):
            continue
        value = _tidy(" ".join(buckets[position]))
        if name in fields:
            fields[name] = value
        else:
            extra[name] = value

    # The 2018-19 layout splits what later became one Reason string into
    # Category plus Reason. Rejoining them with " - " reproduces the modern
    # convention exactly; both halves are kept raw alongside it.
    if extra.get("category") and fields["reason"]:
        fields["reason"] = f"{extra['category']} - {fields['reason']}"
    elif extra.get("category"):
        fields["reason"] = extra["category"]
    fields.update(extra)
    return fields


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
    variant: LayoutVariant | None = None
    for index, row in enumerate(rows):
        found = match_header(row)
        if found is not None:
            variant, anchors = found
            header_position = index
            break
    if anchors is None or header_position is None or variant is None:
        raise ReportParseError(f"{source_filename}: header row not found")

    report = ParsedReport(
        source_filename=source_filename,
        report_timestamp_et=stamp,
        report_date=stamp.date(),
        layout_variant=variant.name,
        column_offsets=tuple(round(a, 1) for a in anchors),
    )

    body = rows[header_position + 1:]
    owners = _continuation_owners(body, anchors, variant)
    reason_by_owner = _reason_pieces_by_owner(body, anchors, variant)

    current: dict[str, str | None] = {
        "game_date": None, "game_time": None, "matchup": None, "team": None,
    }
    entry_for_row: dict[int, PlayerEntry] = {}
    for index, row in enumerate(body):
        line = row.text
        if (_PAGE_MARKER.match(line) or _BARE_NUMBER.match(line)
                or _TIMESTAMP.search(line)):
            continue
        # Pages 2..N repeat neither header nor anything else structural, so a
        # row that looks like a header again is simply skipped.
        if match_header(row) is not None:
            continue

        fields = _row_fields(row, anchors, variant)
        if fields["game_date"] and _GAME_DATE.match(fields["game_date"]):
            current["game_date"] = fields["game_date"]
        if fields["game_time"]:
            current["game_time"] = fields["game_time"].replace("(ET)", "").strip()
        if fields["matchup"] and _MATCHUP.match(fields["matchup"]):
            current["matchup"] = fields["matchup"]
        if fields["team"]:
            # A long team name wraps onto its own row in the narrower
            # portrait-era layouts ("Minnesota" then "Timberwolves"). A row
            # carrying nothing but a team fragment continues the name above it
            # rather than starting a new block, so appending is what keeps the
            # following players attached to the right franchise.
            if _is_bare_team_fragment(fields) and current["team"]:
                current["team"] = _tidy(f"{current['team']} {fields['team']}")
            else:
                current["team"] = fields["team"]

        player = _tidy_name(fields["player_name"])
        status = fields["status"]
        reason = _rejoin_hyphens(fields["reason"])

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
            # A continuation row carries only more reason text. It belongs to
            # the player whose row band it falls in -- which is not always the
            # row above. The reason cell is vertically centred, so a two-line
            # reason straddles the boundary and its first line can sit higher
            # than the player name it describes; attaching by position in the
            # stream silently gives that line to the previous player.
            # Reason text was already reassembled per player from glyph
            # positions, so nothing is appended here. Only a stray status-column
            # fragment with no reason text of its own is folded in.
            if status and not reason and not reason_by_owner:
                owner_index = owners.get(index)
                target = (
                    entry_for_row.get(owner_index) if owner_index is not None else None
                )
                if target is None and report.entries:
                    target = report.entries[-1]
                if target is not None:
                    target.reason_raw = _tidy(f"{target.reason_raw} {status}").strip()
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
        entry = PlayerEntry(
            game_date=current["game_date"],
            game_time_et=current["game_time"],
            matchup=current["matchup"],
            away_team=away,
            home_team=home,
            team=current["team"],
            player_name=player,
            status_raw=status or None,
            status_normalized=normalized,
            reason_raw=reason_by_owner.get(index, reason),
        )
        report.entries.append(entry)
        entry_for_row[index] = entry
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
