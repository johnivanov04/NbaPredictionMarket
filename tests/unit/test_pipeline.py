"""End-to-end pipeline: normalize -> match -> write artefacts -> report.

The two network-facing fetch functions are replaced with stubs that return
captured-shape payloads, so this exercises everything downstream of the HTTP
layer without touching the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from tests.conftest import bdl_game, kalshi_pair

from nba_prediction_market.config import Paths, Settings
from nba_prediction_market.ingestion.raw_store import RawStore
from nba_prediction_market.pipelines import build_dataset
from nba_prediction_market.pipelines.build_dataset import (
    KalshiFetch,
    build_arg_parser,
    main,
    run_pipeline,
)

CUTOFF = {"market_settled_ts": "2026-06-20T00:00:00Z"}

# Three games, one of which has no Kalshi counterpart.
GAMES = [
    bdl_game(id=1, date="2025-10-21"),  # OKC (home) vs HOU
    bdl_game(
        id=2,
        date="2025-10-22",
        home_team={"id": 14, "abbreviation": "LAL", "full_name": "Los Angeles Lakers"},
        visitor_team={"id": 10, "abbreviation": "GSW", "full_name": "Golden State Warriors"},
        home_team_score=109,
        visitor_team_score=119,
    ),
    bdl_game(
        id=3,
        date="2026-06-13",
        postseason=True,
        home_team={"id": 27, "abbreviation": "SAS", "full_name": "San Antonio Spurs"},
        visitor_team={"id": 20, "abbreviation": "NYK", "full_name": "New York Knicks"},
        home_team_score=95,
        visitor_team_score=100,
    ),
]


def kalshi_payload() -> tuple[list[dict], list[dict]]:
    """Two of the three games get markets; one Kalshi event has no NBA game."""
    markets: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for event_ticker, sub_title, away, home, rules_date in (
        ("KXNBAGAME-25OCT21HOUOKC", "HOU at OKC (Oct 21)", "HOU", "OKC", "Oct 21, 2025"),
        ("KXNBAGAME-26JUN13NYKSAS", "NYK at SAS (Jun 13)", "NYK", "SAS", "Jun 13, 2026"),
        # An event with no corresponding NBA game in this pull.
        ("KXNBAGAME-25NOV01ORLMIA", "ORL at MIA (Nov 1)", "ORL", "MIA", "Nov 1, 2025"),
    ):
        pair_markets, event = kalshi_pair(
            event_ticker=event_ticker,
            sub_title=sub_title,
            away=away,
            home=home,
            rules_date=rules_date,
        )
        markets.extend(pair_markets)
        events.append(event)
    return markets, events


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace both fetch functions; raw payloads are still written to disk."""

    def fake_nba(settings: Settings, season: int, raw_store: RawStore):
        pages = [{"data": GAMES, "meta": {"next_cursor": None}}]
        snapshot = raw_store.write(
            f"balldontlie_games_season_{season}", pages, record_count=len(GAMES)
        )
        return GAMES, snapshot

    def fake_kalshi(settings: Settings, series_ticker: str, raw_store: RawStore) -> KalshiFetch:
        markets, events = kalshi_payload()
        # The same ticker arriving from both stores must not double-count.
        duplicated = [*markets, *markets]
        sources = {m["ticker"]: ["historical", "markets"] for m in markets}
        snapshot = raw_store.write(
            f"kalshi_markets_{series_ticker}",
            [{"markets": duplicated}],
            record_count=len(duplicated),
        )
        return KalshiFetch(
            markets=duplicated,
            events=events,
            sources_by_ticker=sources,
            cutoff=CUTOFF,
            snapshots=[snapshot],
        )

    monkeypatch.setattr(build_dataset, "fetch_nba_games", fake_nba)
    monkeypatch.setattr(build_dataset, "fetch_kalshi", fake_kalshi)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(paths=Paths(tmp_path / "data"), balldontlie_api_key="test-key")


def test_pipeline_produces_every_documented_artefact(stubbed, settings: Settings) -> None:
    result = run_pipeline(2025, settings=settings)
    processed = settings.paths.processed

    for name in (
        "nba_games_2025_26",
        "kalshi_nba_markets_2025_26",
        "nba_kalshi_matches_2025_26",
    ):
        assert (processed / f"{name}.parquet").is_file()
        assert (processed / f"{name}.csv").is_file()
    assert (settings.paths.reports / "match_report.json").is_file()
    assert all(path.exists() for path in result.written_files)


def test_parquet_output_round_trips(stubbed, settings: Settings) -> None:
    result = run_pipeline(2025, settings=settings)
    reloaded = pd.read_parquet(settings.paths.processed / "nba_games_2025_26.parquet")

    assert len(reloaded) == len(result.games) == 3
    assert list(reloaded.columns) == list(result.games.columns)
    assert set(reloaded["home_team_code"]) == {"OKC", "LAL", "SAS"}


def test_duplicate_markets_from_both_stores_are_collapsed(stubbed, settings: Settings) -> None:
    result = run_pipeline(2025, settings=settings)
    # 3 events x 2 team markets, fed in twice.
    assert len(result.markets) == 6
    assert result.markets["ticker"].is_unique
    assert len(result.kalshi_events) == 3


def test_matching_counts_are_reported_per_category(stubbed, settings: Settings) -> None:
    result = run_pipeline(2025, settings=settings)
    counts = result.report["counts"]

    assert counts["matched"] == 2          # OKC/HOU and NYK/SAS
    assert counts["unmatched_nba"] == 1    # LAL/GSW has no Kalshi event
    assert counts["unmatched_kalshi"] == 1  # ORL/MIA has no NBA game
    assert counts["ambiguous"] == 0
    assert counts["total_rows"] == len(result.matches) == 4


def result_matched(result) -> pd.DataFrame:
    return result.matches[result.matches["match_status"] == "matched"]


def test_matched_rows_carry_the_market_tickers_needed_to_attach_a_price(
    stubbed, settings: Settings
) -> None:
    """The whole point of the matched table: a game -> its two market tickers."""
    matched = result_matched(run_pipeline(2025, settings=settings))
    row = matched[matched["nba_game_id"] == 1].iloc[0]

    assert row["kalshi_home_market_ticker"] == "KXNBAGAME-25OCT21HOUOKC-OKC"
    assert row["kalshi_away_market_ticker"] == "KXNBAGAME-25OCT21HOUOKC-HOU"
    assert row["kalshi_open_time_utc"] is not None
    assert bool(row["orientation_agrees"]) is True


def test_report_is_valid_json_with_examples_for_every_category(
    stubbed, settings: Settings
) -> None:
    run_pipeline(2025, settings=settings)
    report = json.loads((settings.paths.reports / "match_report.json").read_text())

    assert report["season"] == 2025
    assert report["season_label"] == "2025-26"
    assert report["kalshi_series_ticker"] == "KXNBAGAME"
    assert report["season_verification"]["verified"] is True
    assert report["inputs"]["kalshi_historical_cutoff"] == CUTOFF

    for status in ("matched", "unmatched_nba", "unmatched_kalshi", "ambiguous"):
        section = report["by_status"][status]
        assert section["count"] == report["counts"][status]
        assert len(section["examples"]) == min(section["count"], 5)

    assert report["by_status"]["matched"]["examples"][0]["kalshi_event_ticker"]
    assert report["raw_snapshots"]


def test_report_records_provenance_for_the_raw_pulls(stubbed, settings: Settings) -> None:
    result = run_pipeline(2025, settings=settings)
    names = {s["name"] for s in result.report["raw_snapshots"]}

    assert names == {"balldontlie_games_season_2025", "kalshi_markets_KXNBAGAME"}
    for snapshot in result.report["raw_snapshots"]:
        assert Path(snapshot["path"]).is_file()


def test_quality_checks_surface_cross_source_agreement(stubbed, settings: Settings) -> None:
    checks = run_pipeline(2025, settings=settings).report["quality_checks"]
    assert checks["nba_games_final"] == 3
    assert checks["nba_games_postseason"] == 1
    assert checks["matched_orientation_agrees"] == 2
    assert checks["matched_orientation_disagrees"] == 0
    assert checks["kalshi_markets_flagged_non_nba"] == 0


def test_no_csv_flag_writes_only_parquet(stubbed, settings: Settings) -> None:
    run_pipeline(2025, settings=settings, write_csv=False)
    processed = settings.paths.processed

    assert (processed / "nba_games_2025_26.parquet").is_file()
    assert not (processed / "nba_games_2025_26.csv").exists()


def test_rerunning_is_reproducible(stubbed, settings: Settings) -> None:
    first = run_pipeline(2025, settings=settings)
    second = run_pipeline(2025, settings=settings)

    assert first.report["counts"] == second.report["counts"]
    pd.testing.assert_frame_equal(first.matches, second.matches)


# --- CLI -------------------------------------------------------------------


def test_cli_defaults_to_the_2025_26_season() -> None:
    args = build_arg_parser().parse_args([])
    assert args.season == 2025
    assert args.series_ticker == "KXNBAGAME"
    assert args.no_csv is False


def test_cli_runs_the_pipeline_and_prints_a_summary(
    stubbed, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setenv("BALLDONTLIE_API_KEY", "test-key")
    exit_code = main(
        ["--season", "2025", "--data-dir", str(tmp_path / "data"), "--log-level", "ERROR"]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Season 2025-26" in output
    assert "matched             : 2" in output
    assert "unmatched_nba       : 1" in output
    assert (tmp_path / "data" / "reports" / "match_report.json").is_file()


def test_cli_reports_a_missing_api_key_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.delenv("BALLDONTLIE_API_KEY", raising=False)
    monkeypatch.setattr(
        build_dataset, "load_settings",
        lambda *a, **k: Settings(paths=Paths(tmp_path / "data"), balldontlie_api_key=None),
    )
    exit_code = main(["--data-dir", str(tmp_path / "data"), "--log-level", "ERROR"])

    assert exit_code == 2
    assert "BALLDONTLIE_API_KEY is not set" in capsys.readouterr().err
