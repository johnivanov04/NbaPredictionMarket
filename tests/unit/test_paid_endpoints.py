"""Endpoint capability audit and the leakage prohibitions it encodes."""

from __future__ import annotations

import pytest

from nba_prediction_market.ingestion.paid_endpoints import (
    ENDPOINT_AUDIT,
    ENDPOINTS_BY_NAME,
    PROHIBITED_FOR_HISTORICAL_FEATURES,
    SAFETY_LAGGED_HISTORICAL,
    SAFETY_POST_TIP,
    SAFETY_PROSPECTIVE_ONLY,
    V2_METRIC_ERAS,
    is_prohibited,
)

SAFETY_CLASSES = {
    SAFETY_LAGGED_HISTORICAL, SAFETY_PROSPECTIVE_ONLY, SAFETY_POST_TIP, "D_unclear"
}


def test_every_audited_endpoint_has_a_safety_class_and_a_reason() -> None:
    assert ENDPOINT_AUDIT
    for endpoint in ENDPOINT_AUDIT:
        assert endpoint.safety_class in SAFETY_CLASSES
        assert endpoint.safety_reason.strip(), endpoint.name
        assert endpoint.path.startswith("/")


def test_endpoint_names_are_unique() -> None:
    names = [e.name for e in ENDPOINT_AUDIT]
    assert len(set(names)) == len(names)


# --- the three prohibitions this phase exists to establish ------------------


def test_lineups_are_prohibited_as_a_historical_feature() -> None:
    """Available only from 2025 and only once a game begins."""
    lineups = ENDPOINTS_BY_NAME["lineups"]
    assert lineups.safety_class == SAFETY_POST_TIP
    assert lineups.verified_first_season == 2025
    assert is_prohibited("lineups")
    assert "once a game begins" in lineups.safety_reason


def test_injuries_are_prospective_only() -> None:
    """No as-of timestamp, so past state cannot be reconstructed."""
    injuries = ENDPOINTS_BY_NAME["player_injuries"]
    assert injuries.safety_class == SAFETY_PROSPECTIVE_ONLY
    assert injuries.verified_first_season is None
    assert is_prohibited("player_injuries")
    assert "no as-of timestamp" in injuries.safety_reason


def test_season_averages_are_prohibited_retrospectively() -> None:
    """A full-season aggregate contains results that had not happened yet."""
    averages = ENDPOINTS_BY_NAME["season_averages"]
    assert averages.safety_class == SAFETY_POST_TIP
    assert is_prohibited("season_averages")
    assert "lookahead" in averages.safety_reason


@pytest.mark.parametrize(
    "name", ["lineups", "player_injuries", "season_averages"]
)
def test_the_three_prohibited_feeds_are_not_ingested(name: str) -> None:
    assert ENDPOINTS_BY_NAME[name].ingested is False
    assert name in PROHIBITED_FOR_HISTORICAL_FEATURES


# --- what is permitted ------------------------------------------------------


@pytest.mark.parametrize("name", ["game_player_stats", "game_advanced_stats_v1"])
def test_the_ingested_feeds_are_safe_when_lagged(name: str) -> None:
    endpoint = ENDPOINTS_BY_NAME[name]
    assert endpoint.safety_class == SAFETY_LAGGED_HISTORICAL
    assert endpoint.ingested is True
    assert not is_prohibited(name)
    assert endpoint.verified_first_season == 2006
    assert endpoint.verified_last_season == 2025


def test_safe_feeds_still_state_the_lagging_requirement() -> None:
    """"Safe" never means "usable for the game it describes"."""
    for name in ("game_player_stats", "game_advanced_stats_v1", "game_advanced_stats_v2"):
        assert "lagged" in ENDPOINTS_BY_NAME[name].safety_reason


def test_an_unknown_endpoint_is_not_silently_permitted() -> None:
    assert not is_prohibited("some_new_endpoint")
    assert "some_new_endpoint" not in ENDPOINTS_BY_NAME


# --- documented discrepancies ----------------------------------------------


def test_the_v2_documentation_discrepancy_is_recorded() -> None:
    """Docs say 2015; the feed actually returns data from 2012."""
    v2 = ENDPOINTS_BY_NAME["game_advanced_stats_v2"]
    assert v2.verified_first_season == 2012
    assert v2.doc_discrepancy is not None
    assert "2015" in v2.doc_discrepancy and "2012" in v2.doc_discrepancy


def test_v2_metric_eras_capture_both_discontinuities() -> None:
    assert V2_METRIC_ERAS["tracking_speed_distance_touches_passes"] == 2012
    assert V2_METRIC_ERAS["hustle_deflections_contested_boxouts"] == 2016
    assert min(V2_METRIC_ERAS.values()) == 2012


def test_v2_is_audited_but_not_ingested_in_this_phase() -> None:
    """14 seasons of coverage is recorded; ingestion is a Phase 3A3 decision."""
    assert ENDPOINTS_BY_NAME["game_advanced_stats_v2"].ingested is False


def test_pagination_is_recorded_and_matches_the_verified_limit() -> None:
    for endpoint in ENDPOINT_AUDIT:
        assert "100" in endpoint.pagination
