"""Franchise identity across relocations and rebrands.

BALLDONTLIE normalizes historical franchises to present-day identity (verified
against the live API). These tests pin that empirical behaviour and the
consequences for the 30 stable franchise ids.
"""

from __future__ import annotations

import pytest

from nba_prediction_market.matching.franchises import (
    FRANCHISE_IDS,
    FRANCHISES,
    canonical_abbreviation_for_source_id,
    is_nba_franchise,
)
from nba_prediction_market.matching.team_names import NBA_TEAMS, canonical_abbreviations


def test_exactly_thirty_franchises_with_ids_1_to_30() -> None:
    assert len(FRANCHISES) == 30
    assert frozenset(range(1, 31)) == FRANCHISE_IDS


def test_franchise_abbreviations_match_the_canonical_team_vocabulary() -> None:
    """The join vocabulary and the franchise table must not drift apart."""
    assert sorted(f.abbreviation for f in FRANCHISES.values()) == canonical_abbreviations()
    for franchise in FRANCHISES.values():
        assert franchise.abbreviation in NBA_TEAMS


def test_franchise_ids_and_abbreviations_are_both_unique() -> None:
    abbrs = [f.abbreviation for f in FRANCHISES.values()]
    assert len(set(abbrs)) == 30
    assert len({f.source_team_id for f in FRANCHISES.values()}) == 30


# --- the relocation / rebrand cases ----------------------------------------


@pytest.mark.parametrize(
    ("source_id", "abbreviation", "current_name", "historical_marker"),
    [
        (21, "OKC", "Oklahoma City Thunder", "Seattle SuperSonics"),
        (4, "CHA", "Charlotte Hornets", "Charlotte Bobcats"),
        (19, "NOP", "New Orleans Pelicans", "New Orleans Hornets"),
        (3, "BKN", "Brooklyn Nets", "New Jersey Nets"),
        (13, "LAC", "LA Clippers", "Los Angeles Clippers"),
    ],
)
def test_relocations_resolve_to_one_stable_franchise(
    source_id: int, abbreviation: str, current_name: str, historical_marker: str
) -> None:
    """Each historical identity is served under a single present-day id."""
    franchise = FRANCHISES[source_id]
    assert franchise.abbreviation == abbreviation
    assert franchise.current_full_name == current_name
    assert any(historical_marker in h for h in franchise.historical_identities), (
        f"{historical_marker} should be documented on franchise {source_id}"
    )
    assert canonical_abbreviation_for_source_id(source_id) == abbreviation


def test_seattle_and_oklahoma_city_are_the_same_franchise_not_two() -> None:
    """Elo and sequential features must carry across the 2008 relocation."""
    assert canonical_abbreviation_for_source_id(21) == "OKC"
    assert len([f for f in FRANCHISES.values() if f.abbreviation == "OKC"]) == 1


def test_charlotte_and_new_orleans_are_kept_separate() -> None:
    """The Hornets name moved between cities; the two franchises must not merge."""
    assert canonical_abbreviation_for_source_id(4) == "CHA"
    assert canonical_abbreviation_for_source_id(19) == "NOP"
    assert FRANCHISES[4].source_team_id != FRANCHISES[19].source_team_id


def test_every_documented_historical_identity_is_non_empty() -> None:
    for franchise in FRANCHISES.values():
        for identity in franchise.historical_identities:
            assert identity.strip()


# --- non-franchise ids -----------------------------------------------------


@pytest.mark.parametrize("source_id", [37, 42, 51, 2844, 5193, 216597])
def test_defunct_and_international_teams_are_not_franchises(source_id: int) -> None:
    """BALLDONTLIE also serves 1940s clubs and exhibition opponents."""
    assert not is_nba_franchise(source_id)
    assert canonical_abbreviation_for_source_id(source_id) is None


@pytest.mark.parametrize("value", [None, "21", 0, -1, 31, 3.5, True, False, [21]])
def test_non_integer_or_out_of_range_ids_are_not_franchises(value) -> None:
    assert not is_nba_franchise(value)
    assert canonical_abbreviation_for_source_id(value) is None


@pytest.mark.parametrize("source_id", sorted(FRANCHISE_IDS))
def test_every_franchise_id_round_trips(source_id: int) -> None:
    assert is_nba_franchise(source_id)
    abbreviation = canonical_abbreviation_for_source_id(source_id)
    assert abbreviation is not None
    assert FRANCHISES[source_id].abbreviation == abbreviation
