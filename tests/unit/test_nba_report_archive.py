"""Report URL construction, immutable archiving, parsing, and the runner."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from nba_prediction_market.availability.capture_schedule import plan_captures
from nba_prediction_market.availability.nba_official import (
    NOT_AVAILABLE_STATUS,
    ReportArchive,
    ReportSlot,
    latest_slot_at_or_before,
    slot_from_filename,
    slots_for_date,
)
from nba_prediction_market.availability.nba_report_parser import (
    ReportParseError,
    TextRow,
    header_anchors,
    parse_report_rows,
)
from nba_prediction_market.availability.runner import (
    AvailabilityRunner,
    anchor_health,
)
from nba_prediction_market.availability.snapshot_store import SnapshotStore

# --- URL construction ------------------------------------------------------


def test_a_slot_builds_the_documented_url() -> None:
    slot = ReportSlot(date(2026, 1, 20), 4, 0, "PM")
    assert slot.filename == "Injury-Report_2026-01-20_04_00PM.pdf"
    assert slot.url.endswith("/referee/injury/Injury-Report_2026-01-20_04_00PM.pdf")


def test_forty_eight_slots_per_day_in_order() -> None:
    slots = slots_for_date(date(2026, 1, 20))
    assert len(slots) == 48
    assert slots[0].filename.endswith("12_00AM.pdf")
    assert slots[-1].filename.endswith("11_30PM.pdf")
    assert [s.report_timestamp_utc for s in slots] == sorted(
        s.report_timestamp_utc for s in slots
    )


@pytest.mark.parametrize(
    ("hour_12", "meridiem", "hour_24"),
    [(12, "AM", 0), (1, "AM", 1), (11, "AM", 11), (12, "PM", 12), (1, "PM", 13), (11, "PM", 23)],
)
def test_twelve_hour_clock_converts_correctly(hour_12, meridiem, hour_24) -> None:
    assert ReportSlot(date(2026, 1, 20), hour_12, 0, meridiem).hour_24 == hour_24


def test_filenames_round_trip() -> None:
    for slot in slots_for_date(date(2026, 3, 15)):
        assert slot_from_filename(slot.filename).filename == slot.filename


def test_a_foreign_filename_does_not_parse() -> None:
    assert slot_from_filename("something-else.pdf") is None


def test_slot_timestamps_respect_eastern_daylight_saving() -> None:
    """4pm ET is 21:00 UTC in winter and 20:00 UTC in summer."""
    winter = ReportSlot(date(2026, 1, 20), 4, 0, "PM").report_timestamp_utc
    summer = ReportSlot(date(2026, 6, 15), 4, 0, "PM").report_timestamp_utc
    assert winter.hour == 21
    assert summer.hour == 20


@pytest.mark.parametrize(
    ("anchor_minute", "expected"), [(0, 0), (10, 0), (29, 0), (30, 30), (59, 30)]
)
def test_latest_slot_never_overshoots_the_anchor(anchor_minute, expected) -> None:
    anchor = datetime(2026, 1, 21, 0, anchor_minute, tzinfo=UTC)
    slot = latest_slot_at_or_before(anchor)
    assert slot.report_timestamp_utc <= anchor
    assert slot.report_timestamp_utc.minute == expected


def test_a_naive_anchor_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        latest_slot_at_or_before(datetime(2026, 1, 21))


# --- immutable archive -----------------------------------------------------


SLOT = ReportSlot(date(2026, 1, 20), 4, 0, "PM")
NOW = datetime(2026, 1, 20, 21, 5, tzinfo=UTC)


def test_a_report_is_archived_with_provenance(tmp_path: Path) -> None:
    archive = ReportArchive(tmp_path)
    row = archive.store(
        SLOT, b"%PDF-1.4 fake", http_status=200,
        headers={"last-modified": "Tue, 20 Jan 2026 21:00:05 GMT", "content-length": "13"},
        retrieved_at_utc=NOW,
    )
    assert archive.pdf_path(SLOT).is_file()
    assert row["available"] is True
    assert row["sha256"]
    assert row["last_modified"].startswith("Tue, 20 Jan 2026")
    assert row["report_timestamp_utc"] == "2026-01-20T21:00:00+00:00"
    assert row["retrieved_at_utc"] == "2026-01-20T21:05:00+00:00"
    sidecar = json.loads(archive.sidecar_path(SLOT).read_text())
    assert sidecar["sha256"] == row["sha256"]


def test_the_archive_is_organised_by_date(tmp_path: Path) -> None:
    archive = ReportArchive(tmp_path)
    archive.store(SLOT, b"x", http_status=200, headers={}, retrieved_at_utc=NOW)
    assert archive.pdf_path(SLOT).parent == tmp_path / "2026" / "01" / "20"


def test_an_identical_redownload_is_deduplicated(tmp_path: Path) -> None:
    archive = ReportArchive(tmp_path)
    archive.store(SLOT, b"same", http_status=200, headers={}, retrieved_at_utc=NOW)
    archive.store(
        SLOT, b"same", http_status=200, headers={},
        retrieved_at_utc=NOW + timedelta(hours=1),
    )
    assert archive.stats.archived == 1
    assert archive.stats.already_present == 1
    assert len(list((tmp_path / "2026" / "01" / "20").glob("*.pdf"))) == 1


def test_differing_content_for_one_slot_preserves_both_and_flags_it(tmp_path: Path) -> None:
    """Never silently replace an archived report."""
    archive = ReportArchive(tmp_path)
    archive.store(SLOT, b"original", http_status=200, headers={}, retrieved_at_utc=NOW)
    archive.store(SLOT, b"different", http_status=200, headers={}, retrieved_at_utc=NOW)
    assert archive.stats.hash_conflicts == 1
    pdfs = sorted((tmp_path / "2026" / "01" / "20").glob("*.pdf"))
    assert len(pdfs) == 2
    assert any(".conflict-" in p.name for p in pdfs)
    assert archive.pdf_path(SLOT).read_bytes() == b"original"


def test_an_unavailable_slot_is_recorded_not_treated_as_no_injuries(tmp_path: Path) -> None:
    archive = ReportArchive(tmp_path)
    row = archive.unavailable_row(SLOT, NOT_AVAILABLE_STATUS, NOW)
    assert row["available"] is False
    assert row["http_status"] == 403
    assert row["sha256"] is None
    assert not archive.pdf_path(SLOT).exists()


def test_resume_skips_already_archived_slots(tmp_path: Path) -> None:
    archive = ReportArchive(tmp_path)
    assert archive.has(SLOT) is False
    archive.store(SLOT, b"x", http_status=200, headers={}, retrieved_at_utc=NOW)
    assert archive.has(SLOT) is True
    assert [s.filename for s in archive.archived_slots()] == [SLOT.filename]


# --- parser ----------------------------------------------------------------

# Reports are parsed from glyph coordinates, so the fixture is built the same
# way: real column anchors, real reading order (increasing text-space y, which
# the page's flipping content matrix makes top-to-bottom), and the header row
# on page 1 only.
ANCHORS = (23.1, 119.6, 200.0, 264.2, 425.0, 585.7, 666.1)
_COLUMN_X = {
    "game_date": 24.1, "game_time": 120.6, "matchup": 201.0, "team": 265.2,
    "player": 426.0, "status": 586.7, "reason": 667.1,
}

HEADER_ROW = TextRow(page=1, y=115.2, cells=(
    (23.1, "Game"), (53.0, "Date"), (119.6, "Game"), (149.4, "Time"),
    (200.0, "Matchup"), (264.2, "Team"), (425.0, "Player"), (456.9, "Name"),
    (585.7, "Current"), (624.1, "Status"), (666.1, "Reason"),
))
TIMESTAMP_ROW = TextRow(page=1, y=62.0, cells=(
    (289.1, "Injury"), (355.6, "Report:"), (438.3, "12/24/25"),
    (539.8, "05:30"), (602.5, "PM"),
))


def _row(y: float, page: int = 1, **cells: str) -> TextRow:
    """Place cell text at the real column anchors for that page."""
    placed: list[tuple[float, str]] = []
    for key, value in cells.items():
        if value:
            placed.append((_COLUMN_X[key], value))
    return TextRow(page=page, y=y, cells=tuple(sorted(placed)))


def _page_marker(y: float, page: int, total: int) -> TextRow:
    return TextRow(page=page, y=y, cells=(
        (398.5, "Page"), (420.4, str(page)), (427.8, "of"), (438.3, str(total)),
    ))


# One page: title, header, then rows in reading order.
REPORT_ROWS = [
    TIMESTAMP_ROW,
    HEADER_ROW,
    _row(136.3, game_date="12/24/2025", game_time="07:00 (ET)", matchup="MIN@DEN",
         team="Minnesota Timberwolves", player="Edwards, Anthony",
         status="Probable", reason="Injury/Illness - Ankle"),
    _row(158.3, team="Denver Nuggets", player="Jokic, Nikola",
         status="Questionable", reason="Injury/Illness - Right Wrist"),
    _row(180.3, player="Porter Jr., Michael", status="Out",
         reason="Injury/Illness - Left Knee;"),
    _row(202.3, reason="Soreness"),
    _page_marker(545.4, 1, 1),
]


def parsed(rows=None):
    return parse_report_rows(
        list(rows or REPORT_ROWS), "Injury-Report_2025-12-24_05_30PM.pdf"
    )


def test_the_report_timestamp_is_extracted_in_eastern() -> None:
    report = parsed()
    assert report.report_timestamp_et.hour == 17
    assert report.report_timestamp_et.minute == 30
    assert report.report_timestamp_utc == datetime(2025, 12, 24, 22, 30, tzinfo=UTC)
    assert report.report_date == date(2025, 12, 24)


def test_players_teams_and_matchups_are_extracted() -> None:
    report = parsed()
    names = {e.player_name for e in report.entries}
    assert "Porter Jr., Michael" in names
    assert "Jokic, Nikola" in names
    assert "Edwards, Anthony" in names
    denver = [e for e in report.entries if e.team == "Denver Nuggets"]
    assert denver
    assert denver[0].matchup == "MIN@DEN"
    assert denver[0].away_team == "MIN"
    assert denver[0].home_team == "DEN"
    assert denver[0].game_date == "12/24/2025"
    assert denver[0].game_time_et == "07:00"


def test_statuses_are_normalized_and_raw_is_kept() -> None:
    by_name = {e.player_name: e for e in parsed().entries}
    assert by_name["Porter Jr., Michael"].status_normalized == "out"
    assert by_name["Porter Jr., Michael"].status_raw == "Out"
    assert by_name["Jokic, Nikola"].status_normalized == "questionable"
    assert by_name["Edwards, Anthony"].status_normalized == "probable"


def test_a_wrapped_reason_is_joined_to_its_player() -> None:
    by_name = {e.player_name: e for e in parsed().entries}
    assert "Soreness" in by_name["Porter Jr., Michael"].reason_raw
    assert "Left Knee" in by_name["Porter Jr., Michael"].reason_raw


def test_group_columns_carry_forward_to_later_rows() -> None:
    by_name = {e.player_name: e for e in parsed().entries}
    # Only the Timberwolves row printed the matchup and date.
    assert by_name["Porter Jr., Michael"].team == "Denver Nuggets"
    assert by_name["Porter Jr., Michael"].matchup == "MIN@DEN"
    assert by_name["Porter Jr., Michael"].game_date == "12/24/2025"


def test_the_page_marker_never_becomes_a_player() -> None:
    names = {e.player_name for e in parsed().entries}
    assert not any("Page" in name for name in names)
    assert len(parsed().entries) == 3


def test_a_report_without_a_timestamp_is_a_parse_error() -> None:
    with pytest.raises(ReportParseError, match="no report timestamp"):
        parse_report_rows([_row(10.0, player="Nobody")], "x.pdf")


def test_a_report_without_a_header_row_is_a_parse_error() -> None:
    with pytest.raises(ReportParseError, match="header row not found"):
        parse_report_rows([TIMESTAMP_ROW, _row(200.0, player="Nobody")], "x.pdf")


def test_empty_text_is_a_parse_error() -> None:
    with pytest.raises(ReportParseError, match="no text layer"):
        parse_report_rows([], "x.pdf")


def test_an_unrecognised_status_is_flagged_but_kept() -> None:
    rows = list(REPORT_ROWS)
    rows[2] = _row(136.3, game_date="12/24/2025", game_time="07:00 (ET)",
                   matchup="MIN@DEN", team="Minnesota Timberwolves",
                   player="Edwards, Anthony", status="Banana",
                   reason="Injury/Illness - Ankle")
    report = parsed(rows)
    unknown = [e for e in report.entries if e.status_normalized == "unknown"]
    assert unknown
    assert unknown[0].status_raw == "Banana"
    assert any("Banana" in w for w in report.warnings)


# --- multi-page reports ----------------------------------------------------

# The header prints on page 1 only, and a team's block can run past the page
# break. An earlier character-offset parser dropped every page after the first
# and reported no warning, so these are regression tests.
MULTIPAGE_ROWS = [
    TIMESTAMP_ROW,
    HEADER_ROW,
    _row(136.3, game_date="12/24/2025", game_time="07:00 (ET)", matchup="MIN@DEN",
         team="Minnesota Timberwolves", player="Edwards, Anthony", status="Out",
         reason="Injury/Illness - Ankle"),
    _row(158.3, player="Gobert, Rudy", status="Available", reason="-"),
    _page_marker(545.4, 1, 2),
    # Page 2: no header, no repeated team label -- Reid continues Minnesota.
    TextRow(page=2, y=62.0, cells=TIMESTAMP_ROW.cells),
    _row(136.3, page=2, player="Reid, Naz", status="Questionable",
         reason="Injury/Illness - Wrist"),
    _row(158.3, page=2, team="Denver Nuggets", player="Murray, Jamal",
         status="Probable", reason="Injury/Illness - Hamstring"),
    _page_marker(545.4, 2, 2),
]


def test_every_page_is_parsed_not_only_the_first() -> None:
    names = {e.player_name for e in parse_report_rows(MULTIPAGE_ROWS, "x.pdf").entries}
    assert names == {"Edwards, Anthony", "Gobert, Rudy", "Reid, Naz", "Murray, Jamal"}


def test_a_team_block_carries_across_a_page_break() -> None:
    by_name = {e.player_name: e for e in parse_report_rows(MULTIPAGE_ROWS, "x.pdf").entries}
    assert by_name["Reid, Naz"].team == "Minnesota Timberwolves"
    assert by_name["Reid, Naz"].matchup == "MIN@DEN"
    assert by_name["Murray, Jamal"].team == "Denver Nuggets"


def test_the_repeated_page_timestamp_does_not_end_parsing() -> None:
    report = parse_report_rows(MULTIPAGE_ROWS, "x.pdf")
    assert len(report.entries) == 4
    assert report.warnings == []


# --- teams that have not filed ---------------------------------------------


def test_a_not_yet_submitted_team_is_recorded_separately() -> None:
    rows = [
        *REPORT_ROWS[:-1],
        _row(224.3, team="LA Clippers", reason="NOT YET SUBMITTED"),
        _page_marker(545.4, 1, 1),
    ]
    report = parsed(rows)
    assert [n.team for n in report.teams_not_submitted] == ["LA Clippers"]
    # The pending filing is tied to the game it was pending for, because one
    # report covers several dates.
    pending = report.teams_not_submitted[0]
    assert pending.game_date == "12/24/2025"
    assert pending.matchup == "MIN@DEN"


def test_not_yet_submitted_never_leaks_into_the_previous_player_reason() -> None:
    rows = [
        *REPORT_ROWS[:-1],
        _row(224.3, team="LA Clippers", reason="NOT YET SUBMITTED"),
        _page_marker(545.4, 1, 1),
    ]
    report = parsed(rows)
    last = report.entries[-1]
    assert "NOT YET SUBMITTED" not in last.reason_raw
    assert last.player_name == "Porter Jr., Michael"


# --- column assignment -----------------------------------------------------


def test_multi_word_cells_stay_in_their_own_column() -> None:
    row = TextRow(page=1, y=300.0, cells=(
        (265.2, "Cleveland"), (307.2, "Cavaliers"),
        (426.0, "Garland,"), (462.9, "Darius"),
        (586.7, "Out"),
        (667.1, "Injury/Illness"), (721.8, "-"), (727.1, "Right"), (750.4, "Toe"),
    ))
    report = parse_report_rows([TIMESTAMP_ROW, HEADER_ROW, row], "x.pdf")
    entry = report.entries[0]
    assert entry.team == "Cleveland Cavaliers"
    assert entry.player_name == "Garland, Darius"
    assert entry.status_raw == "Out"
    assert entry.reason_raw == "Injury/Illness - Right Toe"


def test_a_hyphenated_surname_split_across_chunks_is_rejoined() -> None:
    # The PDF draws "Gilgeous-Alexander" as two chunks; joining on whitespace
    # would produce "Gilgeous- Alexander", which matches no player.
    row = TextRow(page=1, y=300.0, cells=(
        (265.2, "Oklahoma"), (300.0, "City"), (320.0, "Thunder"),
        (426.0, "Gilgeous-"), (470.0, "Alexander,"), (520.0, "Shai"),
        (586.7, "Out"), (667.1, "Rest"),
    ))
    report = parse_report_rows([TIMESTAMP_ROW, HEADER_ROW, row], "x.pdf")
    assert report.entries[0].player_name == "Gilgeous-Alexander, Shai"


def test_a_hyphenated_given_name_is_also_rejoined() -> None:
    row = TextRow(page=1, y=300.0, cells=(
        (265.2, "New"), (285.0, "York"), (306.0, "Knicks"),
        (426.0, "Towns,"), (470.0, "Karl-"), (500.0, "Anthony"),
        (586.7, "Available"), (667.1, "-"),
    ))
    report = parse_report_rows([TIMESTAMP_ROW, HEADER_ROW, row], "x.pdf")
    assert report.entries[0].player_name == "Towns, Karl-Anthony"


def test_the_comma_repair_still_applies() -> None:
    row = TextRow(page=1, y=300.0, cells=(
        (265.2, "Chicago"), (300.0, "Bulls"),
        (426.0, "Collins,Zach"), (586.7, "Out"), (667.1, "-"),
    ))
    report = parse_report_rows([TIMESTAMP_ROW, HEADER_ROW, row], "x.pdf")
    assert report.entries[0].player_name == "Collins, Zach"


def test_header_anchors_are_read_from_the_header_row() -> None:
    assert header_anchors(HEADER_ROW) == list(ANCHORS)


def test_a_non_header_row_yields_no_anchors() -> None:
    assert header_anchors(_row(300.0, player="Somebody")) is None


# --- runner ----------------------------------------------------------------


def game(gid=1, tipoff=datetime(2026, 1, 21, 0, 30, tzinfo=UTC)):
    return {"game_id": gid, "scheduled_tipoff_utc": tipoff}


def make_runner(tmp_path: Path, responses):
    calls: list[str] = []

    def fetch(url: str):
        calls.append(url)
        result = responses.pop(0) if isinstance(responses, list) else responses
        if isinstance(result, Exception):
            raise result
        return result

    runner = AvailabilityRunner(
        ReportArchive(tmp_path / "archive"),
        SnapshotStore(tmp_path / "snapshots"),
        fetch=fetch, sleep=lambda _s: None,
    )
    return runner, calls


def test_a_capture_stores_the_pdf_and_a_snapshot(tmp_path: Path) -> None:
    runner, _ = make_runner(tmp_path, (200, b"%PDF", {"last-modified": "x"}))
    tasks = plan_captures([game()], ["nba_official_injury_report"])
    result = runner.run_task(tasks[-1])

    assert result.outcome == "captured"
    assert result.report_timestamp_utc <= result.task.anchor_utc
    assert runner.stats.captured == 1
    assert runner.store.iter_snapshots("nba_official")


def test_source_and_retrieval_timestamps_are_both_preserved(tmp_path: Path) -> None:
    """Fetching a 6:30 report at 6:59 does not make it a 6:59 observation."""
    runner, _ = make_runner(tmp_path, (200, b"%PDF", {}))
    tasks = plan_captures([game()], ["nba_official_injury_report"])
    fetched_at = datetime(2026, 1, 21, 0, 29, tzinfo=UTC)
    result = runner.run_task(tasks[-1], now=fetched_at)

    snapshot = runner.store.iter_snapshots("nba_official")[0]
    assert snapshot["retrieved_at_utc"].startswith("2026-01-21T00:29")
    assert snapshot["source_report_timestamp"] == result.report_timestamp_utc.isoformat()
    assert snapshot["source_report_timestamp"] != snapshot["retrieved_at_utc"]


def test_a_restart_does_not_recapture_or_lose_history(tmp_path: Path) -> None:
    runner, _ = make_runner(tmp_path, (200, b"%PDF", {}))
    tasks = plan_captures([game()], ["nba_official_injury_report"])
    runner.run_task(tasks[-1])

    restarted, restart_calls = make_runner(tmp_path, (200, b"%PDF", {}))
    result = restarted.run_task(tasks[-1])
    assert result.outcome == "already_present"
    assert restart_calls == [], "a restart must not refetch an archived slot"
    assert restarted.archive.has(result.slot)


def test_a_transient_failure_is_retried_then_succeeds(tmp_path: Path) -> None:
    responses = [RuntimeError("boom"), RuntimeError("boom"), (200, b"%PDF", {})]
    runner, calls = make_runner(tmp_path, responses)
    result = runner.run_task(plan_captures([game()], ["s"])[-1])
    assert result.outcome == "captured"
    assert len(calls) == 3


def test_a_persistent_failure_is_recorded_not_raised(tmp_path: Path) -> None:
    runner, _ = make_runner(tmp_path, [RuntimeError("down")] * 5)
    result = runner.run_task(plan_captures([game()], ["s"])[-1])
    assert result.outcome == "failed"
    assert runner.stats.failed == 1
    assert "down" in result.error


def test_a_source_outage_is_distinguished_from_a_failure(tmp_path: Path) -> None:
    runner, _ = make_runner(tmp_path, (NOT_AVAILABLE_STATUS, b"", {}))
    result = runner.run_task(plan_captures([game()], ["s"])[-1])
    assert result.outcome == "source_unavailable"
    assert runner.stats.unavailable == 1
    assert runner.stats.failed == 0


def test_duplicate_tasks_capture_once(tmp_path: Path) -> None:
    runner, calls = make_runner(tmp_path, [(200, b"%PDF", {}), (200, b"%PDF", {})])
    task = plan_captures([game()], ["s"])[-1]
    runner.run_task(task)
    runner.run_task(task)
    assert runner.stats.captured == 1
    assert runner.stats.already_present == 1
    assert len(calls) == 1


def test_anchor_health_flags_an_uncovered_game(tmp_path: Path) -> None:
    runner, _ = make_runner(tmp_path, (NOT_AVAILABLE_STATUS, b"", {}))
    tasks = plan_captures([game()], ["nba_official_injury_report"])
    results = [runner.run_task(tasks[-1])]
    health = anchor_health(results)
    assert health["healthy"] is False
    assert health["anchors_uncovered"] == 1


def test_anchor_health_passes_when_a_fresh_report_was_captured(tmp_path: Path) -> None:
    runner, _ = make_runner(tmp_path, (200, b"%PDF", {}))
    tasks = plan_captures([game()], ["nba_official_injury_report"])
    health = anchor_health([runner.run_task(tasks[-1])])
    assert health["healthy"] is True
    assert health["anchors_covered"] == 1


def test_no_capture_uses_a_report_published_after_its_anchor(tmp_path: Path) -> None:
    """The core guarantee, asserted over a full slate."""
    games = [
        game(1, datetime(2026, 1, 21, 0, 0, tzinfo=UTC)),
        game(2, datetime(2026, 1, 21, 0, 40, tzinfo=UTC)),
        game(3, datetime(2026, 1, 21, 3, 10, tzinfo=UTC)),
    ]
    runner, _ = make_runner(tmp_path, (200, b"%PDF", {}))
    for result in runner.run(plan_captures(games, ["nba_official_injury_report"])):
        if result.report_timestamp_utc:
            assert result.report_timestamp_utc <= result.task.anchor_utc
