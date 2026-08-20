"""Invariants of the generated 20-season historical artefacts.

Reads ``data/`` rather than the network; deselected by default because it needs a
completed pipeline run. Run after regenerating::

    pytest -m dataset
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nba_prediction_market.ingestion.game_phase import PHASE_REGULAR_SEASON
from nba_prediction_market.ingestion.season_metadata import (
    HISTORICAL_SEASONS,
    NBA_TEAM_COUNT,
    SEASON_METADATA,
    STRUCTURE_STANDARD,
)
from nba_prediction_market.matching.franchises import FRANCHISE_IDS

pytestmark = pytest.mark.dataset

# Known upstream data defects, pinned exactly. A NEW defect fails these tests;
# the known ones are preserved in the dataset and reported, never silently
# dropped. See the Phase 3A0 README section.

#: Games recorded with equal "final" scores -- impossible in the NBA. Each is an
#: overtime game whose stored score is an end-of-period snapshot, so the winner
#: is not recoverable and ``home_win`` is null.
CORRUPT_TIE_GAME_IDS = {28012, 32587, 34714, 48851}
#: Regular-season games with no tipoff timestamp from the API.
MISSING_TIMESTAMP_COUNT = 13
#: Games whose ``date`` still holds the original schedule after a postponement.
TIPOFF_DIVERGENCE_COUNT = 49

PROCESSED = Path("data/processed")
REGULAR = PROCESSED / "nba_regular_season_games_2006_26.parquet"
ALL_GAMES = PROCESSED / "nba_all_games_2006_26.parquet"
IDENTITY = PROCESSED / "nba_team_identity_2006_26.parquet"
REPORT = Path("data/reports/historical_nba_2006_26_report.json")


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        pytest.skip(f"{path} not generated; run build_history first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def regular() -> pd.DataFrame:
    return _load(REGULAR)


@pytest.fixture(scope="module")
def all_games() -> pd.DataFrame:
    return _load(ALL_GAMES)


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT.is_file():
        pytest.skip("report not generated")
    return json.loads(REPORT.read_text())


# --- coverage and uniqueness ----------------------------------------------


def test_all_twenty_seasons_are_present(regular: pd.DataFrame) -> None:
    assert sorted(regular["season"].unique()) == list(HISTORICAL_SEASONS)


def test_every_game_id_is_unique(regular: pd.DataFrame, all_games: pd.DataFrame) -> None:
    assert regular["nba_game_id"].is_unique
    assert all_games["nba_game_id"].is_unique


def test_modelling_dataset_contains_only_regular_season(regular: pd.DataFrame) -> None:
    assert set(regular["game_phase"]) == {PHASE_REGULAR_SEASON}


def test_no_unclassified_games_anywhere(all_games: pd.DataFrame) -> None:
    """A silently unclassified game is the failure mode this phase exists to prevent."""
    assert int((all_games["game_phase"] == "unclassified").sum()) == 0


def test_regular_season_is_a_subset_of_all_games(
    regular: pd.DataFrame, all_games: pd.DataFrame
) -> None:
    assert set(regular["nba_game_id"]) <= set(all_games["nba_game_id"])
    assert len(regular) < len(all_games), "playoffs must be preserved separately"


# --- per-season structure --------------------------------------------------


@pytest.mark.parametrize("season", HISTORICAL_SEASONS)
def test_each_season_matches_its_declared_structure(
    regular: pd.DataFrame, season: int
) -> None:
    info = SEASON_METADATA[season]
    sub = regular[regular["season"] == season]
    assert len(sub) > 0

    if info.expected_regular_season_games is not None:
        assert len(sub) == info.expected_regular_season_games, (
            f"season {season} ({info.structure}) expected "
            f"{info.expected_regular_season_games} games, got {len(sub)}"
        )


@pytest.mark.parametrize("season", HISTORICAL_SEASONS)
def test_each_season_has_thirty_teams(regular: pd.DataFrame, season: int) -> None:
    sub = regular[regular["season"] == season]
    teams = pd.concat([sub["home_team"], sub["away_team"]]).nunique()
    assert teams == NBA_TEAM_COUNT


@pytest.mark.parametrize("season", HISTORICAL_SEASONS)
def test_games_per_team_matches_the_declared_expectation(
    regular: pd.DataFrame, season: int
) -> None:
    info = SEASON_METADATA[season]
    if info.expected_games_per_team is None:
        pytest.skip(f"season {season} has no uniform per-team expectation")
    sub = regular[regular["season"] == season]
    counts = pd.concat([sub["home_team"], sub["away_team"]]).value_counts().to_dict()

    for team, played in counts.items():
        allowed = info.known_team_game_count_exceptions.get(
            team, info.expected_games_per_team
        )
        assert played == allowed, f"season {season}: {team} played {played}, expected {allowed}"
    # Every declared exception must actually be present, or it is stale.
    for team, expected in info.known_team_game_count_exceptions.items():
        assert counts.get(team) == expected


@pytest.mark.parametrize("season", HISTORICAL_SEASONS)
def test_regular_season_dates_fall_inside_the_declared_window(
    regular: pd.DataFrame, season: int
) -> None:
    info = SEASON_METADATA[season]
    dates = pd.to_datetime(regular[regular["season"] == season]["date"]).dt.date
    assert dates.min() >= info.regular_season_start
    assert dates.max() <= info.regular_season_end


def test_standard_seasons_are_all_1230_games(regular: pd.DataFrame) -> None:
    for season in HISTORICAL_SEASONS:
        if SEASON_METADATA[season].structure == STRUCTURE_STANDARD:
            assert len(regular[regular["season"] == season]) == 1230, f"season {season}"


def test_the_three_irregular_seasons_are_not_1230(regular: pd.DataFrame) -> None:
    """Forcing these to 1230 would be inventing or dropping real games."""
    for season in (2011, 2019, 2020):
        assert len(regular[regular["season"] == season]) != 1230


# --- era gating ------------------------------------------------------------


def test_no_play_in_games_before_2019(all_games: pd.DataFrame) -> None:
    early = all_games[all_games["season"] < 2019]
    assert int((early["game_phase"] == "play_in").sum()) == 0


def test_no_nba_cup_games_before_2023(all_games: pd.DataFrame) -> None:
    early = all_games[all_games["season"] < 2023]
    assert int((early["game_phase"] == "nba_cup_championship").sum()) == 0
    assert early["ist_stage"].isna().all()


# --- franchise identity ----------------------------------------------------


def test_every_regular_season_team_is_one_of_the_thirty_franchises(
    regular: pd.DataFrame,
) -> None:
    ids = set(regular["home_franchise_id"].dropna().astype(int)) | set(
        regular["away_franchise_id"].dropna().astype(int)
    )
    assert ids == set(FRANCHISE_IDS)
    assert regular["home_franchise_id"].notna().all()
    assert regular["away_franchise_id"].notna().all()


def test_franchise_ids_are_stable_across_all_seasons(regular: pd.DataFrame) -> None:
    """A relocation must not create a second id, or Elo would reset."""
    per_id = regular.groupby("home_franchise_id")["home_team"].nunique()
    assert set(per_id.unique()) == {1}, "one franchise id must map to one abbreviation"


def test_identity_table_records_the_normalization(report: dict) -> None:
    identity = _load(IDENTITY)
    franchises = identity[identity["is_nba_franchise"]]
    assert len(franchises) == NBA_TEAM_COUNT
    # BALLDONTLIE returns present-day identity for every era, so no id should
    # carry more than one label.
    assert report["identity"]["ids_with_multiple_labels"] == []


# --- data quality ----------------------------------------------------------


def test_no_duplicate_date_matchups(report: dict) -> None:
    assert report["data_quality"]["duplicate_date_matchup_count"] == 0


def test_every_regular_season_game_has_scores(regular: pd.DataFrame) -> None:
    assert regular["home_score"].notna().all()
    assert regular["away_score"].notna().all()


def test_only_the_known_corrupt_records_lack_a_result(regular: pd.DataFrame) -> None:
    """Impossible tied 'final' scores; the winner is not recoverable."""
    unresolved = set(regular[regular["home_win"].isna()]["nba_game_id"])
    assert unresolved == CORRUPT_TIE_GAME_IDS

    tied = set(
        regular[regular["home_score"] == regular["away_score"]]["nba_game_id"]
    )
    assert tied == CORRUPT_TIE_GAME_IDS
    # They are preserved, not dropped.
    assert len(regular[regular["nba_game_id"].isin(CORRUPT_TIE_GAME_IDS)]) == 4


def test_home_win_agrees_with_the_scores(regular: pd.DataFrame) -> None:
    usable = regular[~regular["nba_game_id"].isin(CORRUPT_TIE_GAME_IDS)]
    derived = usable["home_score"] > usable["away_score"]
    assert (derived == usable["home_win"]).all()
    assert usable["home_win"].notna().all()


# --- chronology ------------------------------------------------------------


def test_timestamps_are_utc_and_missing_only_where_known(regular: pd.DataFrame) -> None:
    stamps = pd.to_datetime(regular["game_datetime_utc"], utc=True, errors="coerce")
    assert stamps.dt.tz is not None
    assert int(stamps.isna().sum()) == MISSING_TIMESTAMP_COUNT


def test_ordering_must_use_the_tipoff_not_the_scheduled_date(report: dict) -> None:
    """`date` holds the original schedule for postponed games, so it cannot sort."""
    chronology = report["chronology"]
    assert chronology["ordering_field"] == "game_datetime_utc"
    # Sorting by tipoff is physically consistent...
    assert chronology["same_team_twice_at_one_timestamp"] == []
    # ...whereas sorting by `date` is not.
    assert len(chronology["same_team_twice_on_one_scheduled_date"]) == 5


def test_tipoff_divergence_is_confined_to_the_known_covid_seasons(
    report: dict, regular: pd.DataFrame
) -> None:
    chronology = report["chronology"]
    assert chronology["tipoff_diverges_from_scheduled_date"] == TIPOFF_DIVERGENCE_COUNT
    assert set(chronology["tipoff_divergence_by_season"]) == {"2020", "2021", "2022"}
    flagged = regular[regular["tipoff_date_matches_scheduled_date"] == False]  # noqa: E712
    assert len(flagged) == TIPOFF_DIVERGENCE_COUNT


def test_no_team_plays_twice_at_the_same_instant(report: dict) -> None:
    assert report["chronology"]["same_team_twice_at_one_timestamp"] == []


def test_report_totals_agree_with_the_frames(
    report: dict, regular: pd.DataFrame, all_games: pd.DataFrame
) -> None:
    assert report["totals"]["regular_season_games"] == len(regular)
    assert report["totals"]["all_games"] == len(all_games)
    assert report["totals"]["unclassified_games"] == 0
