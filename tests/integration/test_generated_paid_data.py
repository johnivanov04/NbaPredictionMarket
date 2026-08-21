"""Invariants of the generated Phase 3A3A0 paid datasets. ``pytest -m dataset``."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_prediction_market.ingestion.paid_endpoints import (
    PROHIBITED_FOR_HISTORICAL_FEATURES,
)
from nba_prediction_market.ingestion.paid_stats import (
    ADVANCED_COLUMNS,
    BOUNDED_FRACTIONS,
    PLAYER_GAME_COLUMNS,
)
from nba_prediction_market.matching.franchises import FRANCHISE_IDS

pytestmark = pytest.mark.dataset

PROCESSED = Path("data/processed")
PLAYERS = PROCESSED / "nba_player_game_stats_2006_26.parquet"
ADVANCED = PROCESSED / "nba_game_advanced_stats_2006_26.parquet"
GAMES = PROCESSED / "nba_regular_season_games_2006_26.parquet"
REPORT = Path("data/reports/paid_data_audit.json")


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        pytest.skip(f"{path} not generated; run build_paid_data first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def players() -> pd.DataFrame:
    return _load(PLAYERS)


@pytest.fixture(scope="module")
def advanced() -> pd.DataFrame:
    return _load(ADVANCED)


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    frame = _load(GAMES)
    return frame[frame["modeling_eligible"]]


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT.is_file():
        pytest.skip("audit report not generated")
    return json.loads(REPORT.read_text())


# --- schema ----------------------------------------------------------------


def test_player_dataset_has_the_documented_schema(players: pd.DataFrame) -> None:
    assert list(players.columns) == list(PLAYER_GAME_COLUMNS)
    assert len(players) > 500_000


def test_advanced_dataset_has_the_documented_schema(advanced: pd.DataFrame) -> None:
    assert list(advanced.columns) == list(ADVANCED_COLUMNS)
    assert len(advanced) > 500_000


def test_all_twenty_seasons_are_present(players: pd.DataFrame) -> None:
    assert sorted(players["season"].unique()) == list(range(2006, 2026))


def test_one_row_per_game_and_player(players: pd.DataFrame, advanced: pd.DataFrame) -> None:
    for frame in (players, advanced):
        assert not frame.duplicated(subset=["nba_game_id", "player_id"]).any()


# --- joins against the trusted history -------------------------------------


def test_player_stats_cover_the_trusted_games(
    players: pd.DataFrame, games: pd.DataFrame
) -> None:
    trusted = set(games["nba_game_id"])
    covered = trusted & set(players["nba_game_id"])
    assert len(covered) / len(trusted) > 0.99


def test_advanced_stats_cover_the_trusted_games(
    advanced: pd.DataFrame, games: pd.DataFrame
) -> None:
    trusted = set(games["nba_game_id"])
    covered = trusted & set(advanced["nba_game_id"])
    assert len(covered) / len(trusted) > 0.99


def test_every_team_id_is_a_known_franchise(players: pd.DataFrame) -> None:
    teams = set(players["team_id"].dropna().astype(int))
    assert teams <= set(FRANCHISE_IDS)


def test_home_away_is_resolved_for_every_row(players: pd.DataFrame) -> None:
    assert players["is_home"].notna().all()


def test_each_game_has_exactly_two_teams(players: pd.DataFrame) -> None:
    per_game = players.groupby("nba_game_id")["team_id"].nunique()
    assert set(per_game.unique()) == {2}


# --- value sanity ----------------------------------------------------------


@pytest.mark.parametrize("column", ["fg_pct", "fg3_pct", "ft_pct"])
def test_player_percentages_are_bounded(players: pd.DataFrame, column: str) -> None:
    values = players[column].dropna()
    assert values.between(0.0, 1.0).all()


def test_advanced_fractions_are_bounded(advanced: pd.DataFrame) -> None:
    for column in BOUNDED_FRACTIONS:
        if column in advanced.columns:
            values = advanced[column].dropna()
            assert values.between(0.0, 1.0).all(), column


def test_minutes_are_plausible(players: pd.DataFrame) -> None:
    minutes = players["minutes"].dropna()
    assert minutes.min() >= 0.0
    # A quadruple-overtime game runs 68 minutes, so allow headroom but not absurdity.
    assert minutes.max() <= 80.0


def test_pace_is_plausible_for_players_who_actually_played(
    advanced: pd.DataFrame, players: pd.DataFrame
) -> None:
    """Non-participants carry 0.0, not null, so they must be excluded first."""
    joined = advanced.merge(
        players[["nba_game_id", "player_id", "minutes"]],
        on=["nba_game_id", "player_id"], how="left",
    )
    pace = joined[joined["minutes"].fillna(0.0) > 0.0]["pace"].dropna()
    assert pace.quantile(0.01) > 60.0
    assert pace.quantile(0.99) < 130.0


def test_zero_pace_rows_are_non_participants(report: dict) -> None:
    """The zero-inflation is explained, not a mystery: 0.0 means did not play."""
    consistency = report["consistency"]
    zero = consistency["advanced_rows_with_zero_pace"]
    assert zero > 0
    explained = consistency["advanced_zero_pace_with_no_minutes"]
    assert explained / zero > 0.999
    assert consistency["zero_inflation_note"]


def test_player_points_sum_to_team_scores(report: dict) -> None:
    """The strongest cross-check that the feed joins to the right games."""
    consistency = report["consistency"]
    assert consistency["team_point_totals_exact_pct"] > 99.0


def test_the_corrected_4ot_games_are_consistent(report: dict) -> None:
    """Phase 3A0.1 corrected these scores; the paid feed must agree."""
    consistency = report["consistency"]
    assert consistency["corrected_4ot_games_checked"] > 0
    assert (
        consistency["corrected_4ot_games_point_totals_match"]
        == consistency["corrected_4ot_games_checked"]
    )


# --- audit report ----------------------------------------------------------


def test_the_prohibited_feeds_are_recorded_and_not_ingested(report: dict) -> None:
    prohibited = report["prohibited_for_historical_features"]
    for name in ("lineups", "player_injuries", "season_averages"):
        assert name in prohibited
        assert prohibited[name].strip()
    ingested = {
        e["name"] for e in report["endpoint_capability_matrix"]
        if e["ingested_in_phase_3a3a0"]
    }
    assert ingested.isdisjoint(set(PROHIBITED_FOR_HISTORICAL_FEATURES))


def test_the_coverage_matrix_is_per_metric_per_season(report: dict) -> None:
    coverage = report["coverage"]["advanced_stats_v1"]
    matrix = coverage["per_season_non_null_share"]
    assert len(matrix) == 20
    for season, metrics in matrix.items():
        assert 2006 <= int(season) <= 2025
        assert all(0.0 <= share <= 1.0 for share in metrics.values())
    assert coverage["verdict"]


def test_join_statistics_are_reported_per_season(report: dict) -> None:
    for feed in ("player_game_stats", "advanced_stats_v1"):
        join = report["joins"][feed]
        assert join["trusted_games"] == 24038
        assert len(join["per_season"]) == 20
        assert join["coverage_pct"] > 99.0


def test_no_paid_column_leaked_into_the_model_feature_datasets() -> None:
    """This phase is data foundation only -- nothing merges into 3A1/3A2 yet."""
    for name in ("nba_model_features_2006_26", "nba_team_strength_features_2006_26"):
        path = PROCESSED / f"{name}.parquet"
        if not path.is_file():
            continue
        columns = set(pd.read_parquet(path).columns)
        for paid in ("pace", "offensive_rating", "true_shooting_percentage",
                     "usage_percentage", "minutes", "plus_minus"):
            assert paid not in columns, f"{paid} leaked into {name}"
