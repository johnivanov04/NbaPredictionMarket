"""Invariants of the generated 2025-26 artefacts.

These read ``data/`` rather than the network, and are deselected by default
because they require a completed pipeline run. Run them after regenerating::

    pytest -m dataset

They exist to pin the scope of the primary modelling set: the regular season is
1,230 games, and neither play-in nor NBA Cup final games may enter it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_prediction_market.ingestion.game_phase import (
    EXPECTED_REGULAR_SEASON_GAMES,
    NBA_TEAM_COUNT,
    PHASE_NBA_CUP_CHAMPIONSHIP,
    PHASE_PLAY_IN,
    PHASE_PLAYOFFS,
    PHASE_REGULAR_SEASON,
    REGULAR_SEASON_GAMES_PER_TEAM,
)

pytestmark = pytest.mark.dataset

PROCESSED = Path("data/processed")
GAMES = PROCESSED / "nba_games_2025_26.parquet"
MATCHES = PROCESSED / "nba_kalshi_matches_2025_26.parquet"
QUOTES = PROCESSED / "nba_kalshi_pregame_t30_2025_26.parquet"


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        pytest.skip(f"{path} not generated; run the pipelines first")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    return _load(GAMES)


@pytest.fixture(scope="module")
def quotes() -> pd.DataFrame:
    return _load(QUOTES)


# --- phase classification on the real season -------------------------------


def test_season_splits_into_the_expected_phases(games: pd.DataFrame) -> None:
    counts = games["game_phase"].value_counts().to_dict()
    assert counts.get(PHASE_REGULAR_SEASON) == EXPECTED_REGULAR_SEASON_GAMES == 1230
    assert counts.get(PHASE_PLAY_IN) == 6
    assert counts.get(PHASE_NBA_CUP_CHAMPIONSHIP) == 1
    assert counts.get(PHASE_PLAYOFFS) == 85
    assert sum(counts.values()) == len(games)
    assert "unclassified" not in counts


def test_every_team_plays_exactly_82_regular_season_games(games: pd.DataFrame) -> None:
    """The league invariant that exposed the play-in leak."""
    regular = games[games["game_phase"] == PHASE_REGULAR_SEASON]
    per_team = pd.concat([regular["home_team_code"], regular["visitor_team_code"]]).value_counts()

    assert len(per_team) == NBA_TEAM_COUNT
    assert set(per_team.unique()) == {REGULAR_SEASON_GAMES_PER_TEAM}


def test_the_2026_play_in_games_are_the_expected_six(games: pd.DataFrame) -> None:
    play_in = games[games["game_phase"] == PHASE_PLAY_IN].sort_values(
        ["game_date", "home_team_code"]
    )
    actual = {
        (str(r.game_date), r.visitor_team_code, r.home_team_code)
        for r in play_in.itertuples()
    }
    assert actual == {
        ("2026-04-14", "MIA", "CHA"),
        ("2026-04-14", "POR", "PHX"),
        ("2026-04-15", "ORL", "PHI"),
        ("2026-04-15", "GSW", "LAC"),
        ("2026-04-17", "CHA", "ORL"),
        ("2026-04-17", "GSW", "PHX"),
    }
    # All carry postseason=False -- which is exactly why the flag is unusable.
    assert not play_in["postseason"].any()


def test_final_regular_season_day_is_2026_04_12(games: pd.DataFrame) -> None:
    regular = games[games["game_phase"] == PHASE_REGULAR_SEASON]
    assert str(regular["game_date"].max()) == "2026-04-12"


def test_playoffs_open_2026_04_18(games: pd.DataFrame) -> None:
    playoffs = games[games["game_phase"] == PHASE_PLAYOFFS]
    assert str(playoffs["game_date"].min()) == "2026-04-18"


def test_nba_cup_final_is_identified_by_its_api_field(games: pd.DataFrame) -> None:
    final = games[games["game_phase"] == PHASE_NBA_CUP_CHAMPIONSHIP]
    assert len(final) == 1
    row = final.iloc[0]
    assert row["ist_stage"] == "Championship"
    assert str(row["game_date"]) == "2025-12-16"
    assert row["postseason"] is False or not row["postseason"]


# --- the primary Phase 2 dataset -------------------------------------------


def test_primary_dataset_has_exactly_1230_regular_season_games(quotes: pd.DataFrame) -> None:
    assert len(quotes) == EXPECTED_REGULAR_SEASON_GAMES == 1230
    assert quotes["nba_game_id"].is_unique
    assert not quotes["postseason"].any()


def test_primary_dataset_excludes_every_play_in_game(
    quotes: pd.DataFrame, games: pd.DataFrame
) -> None:
    play_in_ids = set(games[games["game_phase"] == PHASE_PLAY_IN]["source_game_id"])
    assert play_in_ids
    assert not set(quotes["nba_game_id"]) & play_in_ids


def test_primary_dataset_excludes_the_nba_cup_final(
    quotes: pd.DataFrame, games: pd.DataFrame
) -> None:
    final_ids = set(games[games["game_phase"] == PHASE_NBA_CUP_CHAMPIONSHIP]["source_game_id"])
    assert not set(quotes["nba_game_id"]) & final_ids


def test_primary_dataset_stops_at_the_regular_season_end(quotes: pd.DataFrame) -> None:
    assert str(quotes["game_date"].max()) == "2026-04-12"


def test_all_1230_games_have_both_side_usable_quotes(quotes: pd.DataFrame) -> None:
    assert int(quotes["both_sides_usable"].sum()) == 1230
    assert quotes["home_quote_issue"].isna().all()
    assert quotes["away_quote_issue"].isna().all()


def test_no_quote_is_taken_from_after_its_prediction_timestamp(quotes: pd.DataFrame) -> None:
    prediction = pd.to_datetime(quotes["prediction_ts_utc"], utc=True)
    for side in ("home", "away"):
        assert (pd.to_datetime(quotes[f"{side}_quote_ts_utc"], utc=True) <= prediction).all()
        assert (quotes[f"{side}_quote_age_seconds"] >= 0).all()


def test_anchor_is_exactly_thirty_minutes_before_tipoff(quotes: pd.DataFrame) -> None:
    delta = pd.to_datetime(quotes["game_datetime_utc"], utc=True) - pd.to_datetime(
        quotes["prediction_ts_utc"], utc=True
    )
    assert set(delta.unique()) == {pd.Timedelta(minutes=30)}


def test_play_in_games_are_preserved_in_the_source_tables(games: pd.DataFrame) -> None:
    """Excluded from modelling, not deleted -- they may be modelled later."""
    matches = _load(MATCHES)
    assert int((games["game_phase"] == PHASE_PLAY_IN).sum()) == 6
    assert int((matches["nba_game_phase"] == PHASE_PLAY_IN).sum()) == 6
