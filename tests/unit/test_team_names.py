"""Team normalization must be exact, complete, and never fuzzy."""

from __future__ import annotations

import pytest

from nba_prediction_market.matching.team_names import (
    AMBIGUOUS_TEAM_STRINGS,
    NBA_TEAMS,
    canonical_abbreviations,
    is_canonical,
    normalize_team,
    resolve_team,
)

EXPECTED_ABBREVIATIONS = [
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]


def test_exactly_thirty_canonical_teams() -> None:
    assert canonical_abbreviations() == EXPECTED_ABBREVIATIONS
    assert len(NBA_TEAMS) == 30


@pytest.mark.parametrize("abbreviation", EXPECTED_ABBREVIATIONS)
def test_abbreviation_resolves_to_itself(abbreviation: str) -> None:
    assert normalize_team(abbreviation) == abbreviation
    assert normalize_team(abbreviation.lower()) == abbreviation
    assert is_canonical(abbreviation)


@pytest.mark.parametrize("abbreviation", EXPECTED_ABBREVIATIONS)
def test_full_name_and_nickname_resolve(abbreviation: str) -> None:
    team = NBA_TEAMS[abbreviation]
    assert normalize_team(team.full_name) == abbreviation
    assert normalize_team(team.nickname) == abbreviation


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Oklahoma City Thunder", "OKC"),
        ("  golden state warriors  ", "GSW"),
        ("GS", "GSW"),
        ("Portland Trail Blazers", "POR"),
        ("Trailblazers", "POR"),
        ("76ers", "PHI"),
        ("Sixers", "PHI"),
        ("PHO", "PHX"),
        ("BRK", "BKN"),
        ("Los Angeles Lakers", "LAL"),
        ("LA Clippers", "LAC"),
        # Kalshi's disambiguated city labels.
        ("Los Angeles L", "LAL"),
        ("Los Angeles C", "LAC"),
        # Accents and typographic punctuation are stripped, not guessed at.
        ("Utah Jazz", "UTA"),
    ],
)
def test_known_variants(raw: str, expected: str) -> None:
    assert normalize_team(raw) == expected


@pytest.mark.parametrize("raw", ["Los Angeles", "LA", "los angeles", " la "])
def test_los_angeles_is_ambiguous_not_guessed(raw: str) -> None:
    """The two LA franchises share a city; picking one would be a coin flip."""
    resolution = resolve_team(raw)
    assert resolution.status == "ambiguous"
    assert resolution.abbreviation is None
    assert normalize_team(raw) is None
    assert resolution.reason is not None


def test_ambiguous_strings_are_absent_from_the_lookup() -> None:
    for raw in AMBIGUOUS_TEAM_STRINGS:
        assert normalize_team(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        "Lakerz",          # typo -- a fuzzy matcher would happily return LAL
        "Bost",            # prefix -- must not match "Boston"
        "Boston Celtics FC",
        "Seattle SuperSonics",
        "XYZ",
        "GUA",             # exhibition opponent seen in the Kalshi archive
        "",
        "   ",
    ],
)
def test_unknown_strings_never_fuzzy_match(raw: str) -> None:
    resolution = resolve_team(raw)
    assert resolution.status == "unknown"
    assert resolution.abbreviation is None
    assert normalize_team(raw) is None


def test_none_is_handled() -> None:
    resolution = resolve_team(None)
    assert resolution.status == "unknown"
    assert normalize_team(None) is None


def test_resolution_ok_property() -> None:
    assert resolve_team("BOS").ok
    assert not resolve_team("LA").ok
    assert not resolve_team("nope").ok


def test_no_alias_collisions_across_teams() -> None:
    """Two franchises must never claim the same alias (the map raises if they do)."""
    seen: dict[str, str] = {}
    for abbreviation in EXPECTED_ABBREVIATIONS:
        team = NBA_TEAMS[abbreviation]
        for alias in (abbreviation, team.full_name, team.nickname):
            resolved = normalize_team(alias)
            assert resolved is not None
            assert seen.setdefault(alias.casefold(), resolved) == resolved
