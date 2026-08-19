"""Canonical NBA team identity, and strict (never fuzzy) name resolution.

Design rule: a name either resolves to exactly one canonical abbreviation or it
does not resolve at all. There is no similarity scoring and no "closest match".
Strings that genuinely denote more than one franchise (notably ``"Los Angeles"``,
which Kalshi uses for the Lakers) resolve to ``ambiguous`` so callers fall back
to a stronger identifier instead of silently picking a side.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

__all__ = [
    "AMBIGUOUS_TEAM_STRINGS",
    "NBA_TEAMS",
    "TeamResolution",
    "canonical_abbreviations",
    "is_canonical",
    "normalize_team",
    "resolve_team",
]


@dataclass(frozen=True)
class NbaTeam:
    """One franchise's canonical identity."""

    abbreviation: str
    city: str
    nickname: str

    @property
    def full_name(self) -> str:
        return f"{self.city} {self.nickname}"


#: The 30 franchises, keyed by the canonical NBA abbreviation used everywhere
#: downstream. These keys are the join vocabulary for both sources.
NBA_TEAMS: Final[dict[str, NbaTeam]] = {
    t.abbreviation: t
    for t in (
        NbaTeam("ATL", "Atlanta", "Hawks"),
        NbaTeam("BOS", "Boston", "Celtics"),
        NbaTeam("BKN", "Brooklyn", "Nets"),
        NbaTeam("CHA", "Charlotte", "Hornets"),
        NbaTeam("CHI", "Chicago", "Bulls"),
        NbaTeam("CLE", "Cleveland", "Cavaliers"),
        NbaTeam("DAL", "Dallas", "Mavericks"),
        NbaTeam("DEN", "Denver", "Nuggets"),
        NbaTeam("DET", "Detroit", "Pistons"),
        NbaTeam("GSW", "Golden State", "Warriors"),
        NbaTeam("HOU", "Houston", "Rockets"),
        NbaTeam("IND", "Indiana", "Pacers"),
        NbaTeam("LAC", "Los Angeles", "Clippers"),
        NbaTeam("LAL", "Los Angeles", "Lakers"),
        NbaTeam("MEM", "Memphis", "Grizzlies"),
        NbaTeam("MIA", "Miami", "Heat"),
        NbaTeam("MIL", "Milwaukee", "Bucks"),
        NbaTeam("MIN", "Minnesota", "Timberwolves"),
        NbaTeam("NOP", "New Orleans", "Pelicans"),
        NbaTeam("NYK", "New York", "Knicks"),
        NbaTeam("OKC", "Oklahoma City", "Thunder"),
        NbaTeam("ORL", "Orlando", "Magic"),
        NbaTeam("PHI", "Philadelphia", "76ers"),
        NbaTeam("PHX", "Phoenix", "Suns"),
        NbaTeam("POR", "Portland", "Trail Blazers"),
        NbaTeam("SAC", "Sacramento", "Kings"),
        NbaTeam("SAS", "San Antonio", "Spurs"),
        NbaTeam("TOR", "Toronto", "Raptors"),
        NbaTeam("UTA", "Utah", "Jazz"),
        NbaTeam("WAS", "Washington", "Wizards"),
    )
}

#: Strings that map to more than one franchise. Resolving these would be a coin
#: flip, so they are refused. ``LA``/``Los Angeles`` both appear verbatim in
#: Kalshi ``yes_sub_title`` values for *both* LA teams.
AMBIGUOUS_TEAM_STRINGS: Final[frozenset[str]] = frozenset({"la", "los angeles", "losangeles"})

#: Extra spellings observed in, or plausibly emitted by, the two sources.
#: Every entry is an exact alias -- no prefix or fuzzy behaviour.
_EXTRA_ALIASES: Final[dict[str, str]] = {
    # Alternate abbreviations seen across NBA data providers.
    "brk": "BKN",
    "bro": "BKN",
    "njn": "BKN",
    "cho": "CHA",
    "gs": "GSW",
    "gsw": "GSW",
    "no": "NOP",
    "noh": "NOP",
    "nor": "NOP",
    "ny": "NYK",
    "nyk": "NYK",
    "phe": "PHX",
    "pho": "PHX",
    "sa": "SAS",
    "san": "SAS",
    "uta": "UTA",
    "utah": "UTA",
    "wsh": "WAS",
    # Kalshi disambiguated city labels ("Los Angeles L" = Lakers).
    "los angeles l": "LAL",
    "los angeles c": "LAC",
    "la l": "LAL",
    "la c": "LAC",
    "new york k": "NYK",
    "la lakers": "LAL",
    "la clippers": "LAC",
    # Common nickname-only and spelling variants.
    "sixers": "PHI",
    "76ers": "PHI",
    "philadelphia sixers": "PHI",
    "blazers": "POR",
    "trailblazers": "POR",
    "portland trailblazers": "POR",
    "wolves": "MIN",
    "timberwolves": "MIN",
    "cavs": "CLE",
    "mavs": "DAL",
    "nuggets": "DEN",
}

_WHITESPACE = re.compile(r"\s+")
# Strips both the ASCII and the typographic apostrophe; the curly form is
# intentional, since API payloads use it.
_PUNCTUATION = re.compile(r"[.’']")  # noqa: RUF001


def _normalize_key(value: str) -> str:
    """Casefold, strip accents/punctuation, and collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = _PUNCTUATION.sub("", stripped)
    return _WHITESPACE.sub(" ", cleaned).strip().casefold()


def _build_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}

    def add(raw: str, abbreviation: str) -> None:
        key = _normalize_key(raw)
        if not key or key in AMBIGUOUS_TEAM_STRINGS:
            return
        existing = lookup.get(key)
        if existing is not None and existing != abbreviation:
            raise ValueError(
                f"Alias {raw!r} is claimed by both {existing} and {abbreviation}; "
                "add it to AMBIGUOUS_TEAM_STRINGS instead of guessing."
            )
        lookup[key] = abbreviation

    for abbreviation, team in NBA_TEAMS.items():
        add(abbreviation, abbreviation)
        add(team.full_name, abbreviation)
        add(team.nickname, abbreviation)
        # City alone is only safe when it identifies one franchise (not LA).
        add(team.city, abbreviation)

    # The two LA clubs collide on city; drop the shared key rather than pick one.
    for ambiguous in AMBIGUOUS_TEAM_STRINGS:
        lookup.pop(ambiguous, None)

    for alias, abbreviation in _EXTRA_ALIASES.items():
        add(alias, abbreviation)
    return lookup


_LOOKUP: Final[dict[str, str]] = _build_lookup()


@dataclass(frozen=True)
class TeamResolution:
    """Outcome of resolving a raw team string.

    ``status`` is one of ``resolved``, ``ambiguous``, or ``unknown``; only
    ``resolved`` carries an ``abbreviation``.
    """

    raw: str
    status: str
    abbreviation: str | None = None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "resolved"


def resolve_team(value: str | None) -> TeamResolution:
    """Resolve ``value`` to a canonical abbreviation, explaining any failure."""
    if value is None:
        return TeamResolution(raw="", status="unknown", reason="value is None")
    key = _normalize_key(value)
    if not key:
        return TeamResolution(raw=value, status="unknown", reason="value is blank")
    if key in AMBIGUOUS_TEAM_STRINGS:
        return TeamResolution(
            raw=value,
            status="ambiguous",
            reason=f"{value!r} matches more than one franchise (Lakers and Clippers)",
        )
    abbreviation = _LOOKUP.get(key)
    if abbreviation is None:
        return TeamResolution(
            raw=value, status="unknown", reason=f"no exact alias registered for {value!r}"
        )
    return TeamResolution(raw=value, status="resolved", abbreviation=abbreviation)


def normalize_team(value: str | None) -> str | None:
    """Return the canonical abbreviation, or ``None`` if unresolved/ambiguous."""
    return resolve_team(value).abbreviation


def is_canonical(value: str) -> bool:
    """True when ``value`` is already one of the 30 canonical abbreviations."""
    return value in NBA_TEAMS


def canonical_abbreviations() -> list[str]:
    """Sorted canonical abbreviations, for schema validation and tests."""
    return sorted(NBA_TEAMS)
