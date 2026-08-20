"""History pipeline: normalization, audits, and the modelling dataset.

End-to-end through MockTransport; no network, no generated artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from nba_prediction_market.clients.balldontlie import BallDontLieClient
from nba_prediction_market.config import ConfigError, Paths, Settings
from nba_prediction_market.ingestion.game_phase import (
    PHASE_NBA_CUP_CHAMPIONSHIP,
    PHASE_OTHER_SPECIAL,
    PHASE_PLAY_IN,
    PHASE_PLAYOFFS,
    PHASE_REGULAR_SEASON,
)
from nba_prediction_market.pipelines.build_history import (
    HISTORY_COLUMNS,
    audit_chronology,
    audit_identity,
    audit_season,
    build_arg_parser,
    build_history_frame,
    parse_seasons,
    run_pipeline,
)

_UNSET = object()


def game(
    gid: int,
    season: int,
    day: str,
    *,
    home_id: int = 21,
    away_id: int = 11,
    home_name: str = "Oklahoma City Thunder",
    away_name: str = "Houston Rockets",
    home_abbr: str = "OKC",
    away_abbr: str = "HOU",
    home_score: int | None = 110,
    away_score: int | None = 100,
    postseason: bool = False,
    ist_stage: str | None = None,
    dt: Any = _UNSET,
    status: str = "Final",
) -> dict[str, Any]:
    return {
        "id": gid,
        "date": day,
        "season": season,
        "status": status,
        "status_state": "final" if status == "Final" else "scheduled",
        "period": 4,
        "postseason": postseason,
        "postponed": False,
        "ist_stage": ist_stage,
        "home_team_score": home_score,
        "visitor_team_score": away_score,
        "datetime": f"{day}T23:30:00.000Z" if dt is _UNSET else dt,
        "home_team": {"id": home_id, "abbreviation": home_abbr, "full_name": home_name},
        "visitor_team": {"id": away_id, "abbreviation": away_abbr, "full_name": away_name},
    }


# --- normalization ---------------------------------------------------------


def test_frame_has_the_documented_schema() -> None:
    frame = build_history_frame({2006: [game(1, 2006, "2006-11-01")]})
    assert list(frame.columns) == HISTORY_COLUMNS
    row = frame.iloc[0]
    assert row["nba_game_id"] == 1
    assert row["season"] == 2006
    assert row["season_label"] == "2006-07"
    assert row["home_team_source_id"] == 21
    assert row["home_franchise_id"] == 21
    assert row["home_team"] == "OKC"
    assert row["game_phase"] == PHASE_REGULAR_SEASON
    assert bool(row["home_win"]) is True


def test_sonics_era_game_carries_present_day_identity() -> None:
    """BALLDONTLIE returns OKC for 2007-08 Seattle games; that is the franchise id."""
    frame = build_history_frame({2007: [game(1, 2007, "2007-11-01")]})
    row = frame.iloc[0]
    assert row["home_franchise_id"] == 21
    assert row["home_team"] == "OKC"


def test_non_franchise_opponent_gets_no_franchise_id_and_is_special() -> None:
    frame = build_history_frame(
        {2025: [game(1, 2025, "2025-10-13", away_id=5193,
                     away_abbr="GUA", away_name="Guangzhou Long-Lions")]}
    )
    row = frame.iloc[0]
    assert row["away_team_source_id"] == 5193
    assert row["away_franchise_id"] is None
    assert row["game_phase"] == PHASE_OTHER_SPECIAL


def test_frame_is_sorted_by_season_then_date_then_id() -> None:
    frame = build_history_frame({
        2007: [game(9, 2007, "2007-11-02")],
        2006: [game(5, 2006, "2006-11-02"), game(3, 2006, "2006-11-01")],
    })
    assert list(frame["nba_game_id"]) == [3, 5, 9]


def test_phases_across_eras() -> None:
    frame = build_history_frame({
        2011: [game(1, 2011, "2011-12-25")],
        2019: [game(2, 2019, "2020-08-15")],
        2025: [
            game(3, 2025, "2026-04-15"),
            game(4, 2025, "2025-12-16", ist_stage="Championship"),
            game(5, 2025, "2026-04-20", postseason=True),
        ],
    })
    phases = dict(zip(frame["nba_game_id"], frame["game_phase"], strict=True))
    assert phases[1] == PHASE_REGULAR_SEASON       # lockout season opener
    assert phases[2] == PHASE_PLAY_IN              # 2020 one-off play-in
    assert phases[3] == PHASE_PLAY_IN
    assert phases[4] == PHASE_NBA_CUP_CHAMPIONSHIP
    assert phases[5] == PHASE_PLAYOFFS


# --- audits ----------------------------------------------------------------


def test_season_audit_reports_counts_and_quality() -> None:
    games = [game(i, 2006, "2006-11-01") for i in range(1, 4)]
    games.append(game(4, 2006, "2007-04-25", postseason=True))
    games.append(game(5, 2006, "2006-11-02", home_score=None, away_score=None))
    audit = audit_season(build_history_frame({2006: games}), 2006)

    assert audit["season_label"] == "2006-07"
    assert audit["structure"] == "standard"
    assert audit["raw_games_returned"] == 5
    assert audit["playoff_games"] == 1
    assert audit["play_in_games"] == 0
    assert audit["data_quality"]["final_games_missing_scores"] == 1
    assert audit["era"]["play_in_expected"] is False
    assert audit["era"]["nba_cup_expected"] is False


def test_season_audit_flags_duplicate_ids() -> None:
    games = [game(1, 2006, "2006-11-01"), game(1, 2006, "2006-11-02")]
    audit = audit_season(build_history_frame({2006: games}), 2006)
    assert audit["data_quality"]["duplicate_game_ids"] == [1]


def test_season_audit_flags_a_regular_season_game_outside_the_window() -> None:
    """Guards the declared boundaries against being quietly wrong."""
    frame = build_history_frame({2006: [game(1, 2006, "2006-11-01")]})
    frame.loc[0, "date"] = pd.Timestamp("2007-09-01").date()
    audit = audit_season(frame, 2006)
    assert audit["data_quality"]["regular_season_games_outside_declared_window"]


def test_season_audit_flags_ties() -> None:
    frame = build_history_frame(
        {2006: [game(1, 2006, "2006-11-01", home_score=100, away_score=100)]}
    )
    assert audit_season(frame, 2006)["data_quality"]["tied_final_scores"] == 1


def test_identity_audit_lists_every_observed_combination() -> None:
    frame = build_history_frame({
        2007: [game(1, 2007, "2007-11-01")],
        2025: [game(2, 2025, "2026-01-15", away_id=5193, away_abbr="GUA",
                    away_name="Guangzhou Long-Lions")],
    })
    identity = audit_identity(frame)

    okc = identity[identity["source_team_id"] == 21].iloc[0]
    assert okc["canonical_abbreviation"] == "OKC"
    assert bool(okc["is_nba_franchise"]) is True
    assert okc["first_season"] == 2007

    guangzhou = identity[identity["source_team_id"] == 5193].iloc[0]
    assert bool(guangzhou["is_nba_franchise"]) is False
    assert pd.isna(guangzhou["canonical_franchise_id"])


def test_chronology_audit_counts_missing_and_shared_timestamps() -> None:
    frame = build_history_frame({2006: [
        game(1, 2006, "2006-11-01", dt="2006-11-02T00:00:00.000Z"),
        game(2, 2006, "2006-11-01", dt="2006-11-02T00:00:00.000Z", home_id=2, away_id=5,
             home_abbr="BOS", away_abbr="CHI", home_name="Boston Celtics",
             away_name="Chicago Bulls"),
        game(3, 2006, "2006-11-02", dt=None, home_id=7, away_id=8, home_abbr="DAL",
             away_abbr="DEN", home_name="Dallas Mavericks", away_name="Denver Nuggets"),
    ]})
    audit = audit_chronology(frame)

    assert audit["total_games"] == 3
    assert audit["with_datetime"] == 2
    assert audit["missing_datetime"] == 1
    assert audit["max_games_at_one_timestamp"] == 2
    assert audit["timezone_aware"] is True
    assert audit["same_team_twice_at_one_timestamp"] == []


def test_chronology_audit_detects_a_team_playing_twice_at_one_instant() -> None:
    """Impossible ordering that would corrupt sequential features."""
    frame = build_history_frame({2006: [
        game(1, 2006, "2006-11-01", dt="2006-11-02T00:00:00.000Z"),
        game(2, 2006, "2006-11-01", dt="2006-11-02T00:00:00.000Z", away_id=5,
             away_abbr="CHI", away_name="Chicago Bulls"),
    ]})
    impossible = audit_chronology(frame)["same_team_twice_at_one_timestamp"]
    assert any(x["team"] == "OKC" for x in impossible)


# --- end to end ------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(paths=Paths(tmp_path / "data"), balldontlie_api_key="k")


def stub_client(by_season: dict[int, list[dict]]) -> BallDontLieClient:
    def handler(request: httpx.Request) -> httpx.Response:
        season = int(request.url.params["seasons[]"])
        return httpx.Response(
            200, json={"data": by_season.get(season, []), "meta": {"next_cursor": None}}
        )

    return BallDontLieClient(
        "k", base_url="https://api.test", min_interval=0.0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_pipeline_writes_datasets_and_report(settings: Settings) -> None:
    by_season = {
        2006: [game(1, 2006, "2006-11-01"), game(2, 2006, "2007-04-25", postseason=True)],
        2011: [game(3, 2011, "2011-12-25")],
    }
    result = run_pipeline([2006, 2011], settings=settings, client=stub_client(by_season))

    processed = settings.paths.processed
    assert (processed / "nba_regular_season_games_2006_12.parquet").is_file()
    assert (processed / "nba_all_games_2006_12.parquet").is_file()
    assert (processed / "nba_team_identity_2006_12.parquet").is_file()
    assert (settings.paths.reports / "historical_nba_2006_12_report.json").is_file()
    assert all(p.exists() for p in result.written_files)

    assert len(result.all_games) == 3
    assert len(result.regular_season) == 2
    assert set(result.regular_season["game_phase"]) == {PHASE_REGULAR_SEASON}


def test_modelling_dataset_contains_only_regular_season(settings: Settings) -> None:
    by_season = {2025: [
        game(1, 2025, "2026-01-15"),
        game(2, 2025, "2026-04-15"),                       # play-in
        game(3, 2025, "2025-12-16", ist_stage="Championship"),
        game(4, 2025, "2026-04-20", postseason=True),
        game(5, 2025, "2025-10-13", away_id=5193, away_abbr="GUA",
             away_name="Guangzhou Long-Lions"),            # exhibition
    ]}
    result = run_pipeline([2025], settings=settings, client=stub_client(by_season))

    assert list(result.regular_season["nba_game_id"]) == [1]
    # ...but every other phase survives in the broader table.
    assert len(result.all_games) == 5
    assert set(result.all_games["game_phase"]) == {
        PHASE_REGULAR_SEASON, PHASE_PLAY_IN, PHASE_NBA_CUP_CHAMPIONSHIP,
        PHASE_PLAYOFFS, PHASE_OTHER_SPECIAL,
    }


def test_report_covers_every_required_section(settings: Settings) -> None:
    by_season = {2006: [game(1, 2006, "2006-11-01")], 2011: [game(2, 2011, "2011-12-25")]}
    run_pipeline([2006, 2011], settings=settings, client=stub_client(by_season))
    report = json.loads(
        (settings.paths.reports / "historical_nba_2006_12_report.json").read_text()
    )

    assert report["seasons"] == [2006, 2011]
    assert set(report["totals"]) >= {
        "all_games", "regular_season_games", "playoff_games", "play_in_games",
        "nba_cup_championship_games", "other_special_games", "unclassified_games",
    }
    assert "chronology" in report and "identity" in report and "data_quality" in report
    assert len(report["seasons_detail"]) == 2
    for detail in report["seasons_detail"]:
        assert set(detail) >= {
            "raw_games_returned", "regular_season_games", "playoff_games", "play_in_games",
            "teams", "games_per_team", "earliest_regular_season_date",
            "latest_regular_season_date", "validation_status", "notes",
        }
    assert report["season_structures"]["shortened"] == [2011]


def test_pipeline_refuses_an_undeclared_season(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="no declared metadata"):
        run_pipeline([1999], settings=settings, client=stub_client({}))


def test_pipeline_is_reproducible(settings: Settings) -> None:
    by_season = {2006: [game(1, 2006, "2006-11-01")]}
    first = run_pipeline([2006], settings=settings, client=stub_client(by_season))
    second = run_pipeline([2006], settings=settings, client=stub_client(by_season))
    pd.testing.assert_frame_equal(first.regular_season, second.regular_season)
    assert second.report["cache"]["hits"] >= 1


# --- CLI -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("2006-2008", [2006, 2007, 2008]),
        ("2006,2011,2019", [2006, 2011, 2019]),
        ("2006-2007,2025", [2006, 2007, 2025]),
        ("2011", [2011]),
    ],
)
def test_parse_seasons(spec: str, expected: list[int]) -> None:
    assert parse_seasons(spec) == expected


def test_parse_seasons_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no seasons parsed"):
        parse_seasons("  ")


def test_cli_defaults_to_the_full_historical_range() -> None:
    args = build_arg_parser().parse_args([])
    assert parse_seasons(args.seasons) == list(range(2006, 2026))
    assert args.refresh is False
