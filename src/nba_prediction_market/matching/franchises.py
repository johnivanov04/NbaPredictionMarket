"""Canonical franchise identity across the historical range.

**Empirical finding (verified against the live API on 2026-08-19):**
BALLDONTLIE presents every historical franchise under its *present-day* id,
abbreviation, and full name. Games from eras with different branding come back
already normalized:

===============================  =======================================
historical reality               what ``/v1/games`` returns
===============================  =======================================
Seattle SuperSonics (to 2007-08) ``id=21 OKC "Oklahoma City Thunder"``
Charlotte Bobcats (2004-2014)    ``id=4 CHA "Charlotte Hornets"``
New Orleans Hornets (to 2012-13) ``id=19 NOP "New Orleans Pelicans"``
New Jersey Nets (to 2011-12)     ``id=3 BKN "Brooklyn Nets"``
===============================  =======================================

``/v1/teams`` contains **no** SuperSonics, Bobcats, or New Orleans Hornets
entry, confirming this is normalization rather than a per-era record.

Consequences, both of which matter for sequential features and Elo later:

* **No franchise-continuity mapping is needed.** The source team id is already
  a stable canonical franchise id across all 20 seasons. Building a relocation
  map would be redundant machinery around behaviour the API already provides.
* **Historical display names are not recoverable from this source.** A 2007-08
  Sonics game is labelled "Oklahoma City Thunder". That is correct for franchise
  continuity and wrong for historical presentation; if era-accurate names are
  ever needed, they must come from elsewhere. This is documented rather than
  patched, because inventing them would be fabricating data.

Ids outside this table (BALLDONTLIE also serves defunct 1940s clubs and
international exhibition opponents) are deliberately *not* franchises, which is
what lets exhibition games be identified and excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "FRANCHISES",
    "FRANCHISE_IDS",
    "Franchise",
    "canonical_abbreviation_for_source_id",
    "is_nba_franchise",
]


@dataclass(frozen=True)
class Franchise:
    """One of the 30 current NBA franchises, keyed by BALLDONTLIE's team id."""

    source_team_id: int
    abbreviation: str
    current_full_name: str
    #: Prior identities this id covers within 2006-07..2025-26, for the audit.
    historical_identities: tuple[str, ...] = ()


#: Source team id is the canonical franchise id. Verified stable across every
#: season in the historical range.
FRANCHISES: Final[dict[int, Franchise]] = {
    f.source_team_id: f
    for f in (
        Franchise(1, "ATL", "Atlanta Hawks"),
        Franchise(2, "BOS", "Boston Celtics"),
        Franchise(3, "BKN", "Brooklyn Nets", ("New Jersey Nets (through 2011-12)",)),
        Franchise(4, "CHA", "Charlotte Hornets", ("Charlotte Bobcats (2004-05..2013-14)",)),
        Franchise(5, "CHI", "Chicago Bulls"),
        Franchise(6, "CLE", "Cleveland Cavaliers"),
        Franchise(7, "DAL", "Dallas Mavericks"),
        Franchise(8, "DEN", "Denver Nuggets"),
        Franchise(9, "DET", "Detroit Pistons"),
        Franchise(10, "GSW", "Golden State Warriors"),
        Franchise(11, "HOU", "Houston Rockets"),
        Franchise(12, "IND", "Indiana Pacers"),
        Franchise(13, "LAC", "LA Clippers", ("Los Angeles Clippers (branding change 2015)",)),
        Franchise(14, "LAL", "Los Angeles Lakers"),
        Franchise(15, "MEM", "Memphis Grizzlies"),
        Franchise(16, "MIA", "Miami Heat"),
        Franchise(17, "MIL", "Milwaukee Bucks"),
        Franchise(18, "MIN", "Minnesota Timberwolves"),
        Franchise(
            19, "NOP", "New Orleans Pelicans",
            (
                "New Orleans Hornets (through 2012-13)",
                "New Orleans/Oklahoma City Hornets (2005-06, 2006-07)",
            ),
        ),
        Franchise(20, "NYK", "New York Knicks"),
        Franchise(21, "OKC", "Oklahoma City Thunder", ("Seattle SuperSonics (through 2007-08)",)),
        Franchise(22, "ORL", "Orlando Magic"),
        Franchise(23, "PHI", "Philadelphia 76ers"),
        Franchise(24, "PHX", "Phoenix Suns"),
        Franchise(25, "POR", "Portland Trail Blazers"),
        Franchise(26, "SAC", "Sacramento Kings"),
        Franchise(27, "SAS", "San Antonio Spurs"),
        Franchise(28, "TOR", "Toronto Raptors"),
        Franchise(29, "UTA", "Utah Jazz"),
        Franchise(30, "WAS", "Washington Wizards"),
    )
}

#: The 30 canonical franchise ids. Anything else is not an NBA franchise.
FRANCHISE_IDS: Final[frozenset[int]] = frozenset(FRANCHISES)


def is_nba_franchise(source_team_id: object) -> bool:
    """True when ``source_team_id`` is one of the 30 NBA franchises.

    BALLDONTLIE also serves defunct 1940s-50s clubs (ids 37-51) and
    international exhibition opponents (ids in the thousands); those are not
    franchises, and a game involving one is not an NBA game.
    """
    if isinstance(source_team_id, bool) or not isinstance(source_team_id, int):
        return False
    return source_team_id in FRANCHISE_IDS


def canonical_abbreviation_for_source_id(source_team_id: object) -> str | None:
    """Canonical abbreviation for a source team id, or ``None`` if not a franchise."""
    if not is_nba_franchise(source_team_id):
        return None
    return FRANCHISES[int(source_team_id)].abbreviation  # type: ignore[arg-type]
