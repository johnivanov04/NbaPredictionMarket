"""Verified player-identity corrections for official injury reports.

Resolution normally requires an exact name match *within the named team*, and
that rule stays untouched: names collide, so name-only matching is refused.
This module is the explicit escape hatch for cases that rule cannot reach, and
every entry is an individually verified fact rather than an inference.

**There is deliberately no fuzzy fallback.** Nothing here matches on string
similarity. An unlisted mismatch stays unresolved and is reported as such.

Three things go wrong, and they need different evidence:

* **Preferred name vs legal name.** The report prints "Bub Carrington"; the
  roster carries "Carlton Carrington". Verified by the legal-name player being
  the *only* holder of that surname on that exact team in the season registry.
* **Naming convention and legal name changes.** "Jones Garcia, David" is the
  Dominican double surname of the player BALLDONTLIE records as "David Jones";
  "Hayes-Davis, Nigel" is the current legal name of "Nigel Hayes". Verified
  against the canonical ``/players`` record, including its team.
* **Reported before the first box score.** A player traded or signed
  mid-season appears on his new team's injury report before he appears in any
  box score for it, so the season registry still has him elsewhere. Verified by
  the full name being unique league-wide, which makes the id unambiguous even
  though the team differs.

That last pattern is recurring rather than exceptional; the durable fix is a
date-aware roster instead of a season-level one. Until then each instance is
listed here with its evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

#: Alias reason codes, kept coarse enough to aggregate in reports.
PREFERRED_NAME: Final = "preferred_name_vs_legal_name"
NAMING_CONVENTION: Final = "naming_convention_or_legal_change"
PRE_FIRST_APPEARANCE: Final = "reported_before_first_box_score_for_team"


@dataclass(frozen=True)
class PlayerAlias:
    """One verified mapping from a reported name to a canonical player."""

    report_name: str
    report_team: str
    player_id: int
    canonical_name: str
    reason: str
    evidence: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.report_name, self.report_team)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_name": self.report_name,
            "report_team": self.report_team,
            "canonical_player_id": self.player_id,
            "canonical_name": self.canonical_name,
            "reason": self.reason,
            "evidence": self.evidence,
        }


_REGISTRY_UNIQUE_2024 = (
    "BALLDONTLIE returns exactly one player with this full name and its id "
    "matches the 2024-25 season registry, so the id is unambiguous even though "
    "the registry team differs"
)

_REGISTRY_UNIQUE = (
    "BALLDONTLIE /players returns exactly one active player with this full "
    "name, and its id matches the season registry, so the id is unambiguous "
    "even though the registry team differs"
)

#: 2024-25, surfaced by the Phase 3A3B2 recovery of that season. Same three
#: categories, verified the same way. The table is keyed by (name, team)
#: because a preferred name is corroborated by that team's roster, so a player
#: reported under two teams needs an entry for each.
ALIASES_2024_25: Final[tuple[PlayerAlias, ...]] = (
    PlayerAlias(
        "Williams, Nate", "HOU", 47738533, "Jeenathan Williams", PREFERRED_NAME,
        "only Williams on the 2024-25 Houston roster; BALLDONTLIE id 47738533",
    ),
    PlayerAlias(
        "Reddish, Cam", "LAL", 666860, "Cameron Reddish", PREFERRED_NAME,
        "only Reddish on the 2024-25 Lakers roster; BALLDONTLIE id 666860",
    ),
    PlayerAlias(
        "Reddish, Cam", "CHA", 666860, "Cameron Reddish", PREFERRED_NAME,
        "only Reddish on the 2024-25 Charlotte roster; BALLDONTLIE id 666860",
    ),
    PlayerAlias(
        "Hyland, Bones", "ATL", 17896031, "Nah'Shon Hyland", PREFERRED_NAME,
        "only Hyland on the 2024-25 Atlanta roster; BALLDONTLIE id 17896031",
    ),
    PlayerAlias(
        "Hyland, Bones", "LAC", 17896031, "Nah'Shon Hyland", PREFERRED_NAME,
        "only Hyland on the 2024-25 Clippers roster; BALLDONTLIE id 17896031",
    ),
    PlayerAlias(
        "Martin, KJ", "PHI", 3547294, "Kenyon Martin Jr.", PREFERRED_NAME,
        "BALLDONTLIE id 3547294 is Kenyon Martin Jr., who the 2024-25 registry "
        "places on Philadelphia; the only other Martin on that roster is Caleb",
    ),
    PlayerAlias(
        "Martin, KJ", "UTA", 3547294, "Kenyon Martin Jr.", PREFERRED_NAME,
        "BALLDONTLIE id 3547294, whom the 2024-25 registry also places on Utah",
    ),
    PlayerAlias(
        "Jones Garcia, David", "UTA", 1028245237, "David Jones", NAMING_CONVENTION,
        "BALLDONTLIE id 1028245237; the 2024-25 registry places David Jones on "
        "Utah, and the report prints the full Dominican double surname",
    ),
    PlayerAlias("Martin, KJ", "DET", 3547294, "Kenyon Martin Jr.",
                PRE_FIRST_APPEARANCE,
                _REGISTRY_UNIQUE_2024 + " (registry has Philadelphia and Utah)"),
    PlayerAlias("Wiseman, James", "TOR", 3547240, "James Wiseman",
                PRE_FIRST_APPEARANCE,
                _REGISTRY_UNIQUE_2024 + " (registry has Indiana)"),
    PlayerAlias("Bamba, Mo", "UTA", 28, "Mo Bamba", PRE_FIRST_APPEARANCE,
                _REGISTRY_UNIQUE_2024 + " (registry has the Clippers and New Orleans)"),
    PlayerAlias("Cissoko, Sidy", "WAS", 56677817, "Sidy Cissoko",
                PRE_FIRST_APPEARANCE,
                _REGISTRY_UNIQUE_2024 + " (registry has Portland, Sacramento, San Antonio)"),
    PlayerAlias("Jackson, Reggie", "WAS", 236, "Reggie Jackson",
                PRE_FIRST_APPEARANCE,
                _REGISTRY_UNIQUE_2024 + " (registry has Philadelphia)"),
)

ALIASES_2025_26: Final[tuple[PlayerAlias, ...]] = (
    # --- preferred name vs legal name -------------------------------------
    PlayerAlias(
        "Sarr, Alex", "WAS", 1028028405, "Alexandre Sarr", PREFERRED_NAME,
        "only Sarr on the 2025-26 Washington roster; BALLDONTLIE id 1028028405",
    ),
    PlayerAlias(
        "Claxton, Nic", "BKN", 666508, "Nicolas Claxton", PREFERRED_NAME,
        "only Claxton on the 2025-26 Brooklyn roster; BALLDONTLIE id 666508",
    ),
    PlayerAlias(
        "Bailey, Ace", "UTA", 1057260888, "Airious Bailey", PREFERRED_NAME,
        "only Bailey on the 2025-26 Utah roster; BALLDONTLIE id 1057260888",
    ),
    PlayerAlias(
        "Hyland, Bones", "MIN", 17896031, "Nah'Shon Hyland", PREFERRED_NAME,
        "only Hyland on the 2025-26 Minnesota roster; BALLDONTLIE id 17896031",
    ),
    PlayerAlias(
        "Williams, Nate", "GSW", 47738533, "Jeenathan Williams", PREFERRED_NAME,
        "only Williams on the 2025-26 Golden State roster; BALLDONTLIE id 47738533",
    ),
    PlayerAlias(
        "Carrington, Bub", "WAS", 1028025235, "Carlton Carrington", PREFERRED_NAME,
        "only Carrington on the 2025-26 Washington roster; BALLDONTLIE id 1028025235",
    ),
    # --- naming convention / legal name change ----------------------------
    PlayerAlias(
        "Jones Garcia, David", "SAS", 1028245237, "David Jones", NAMING_CONVENTION,
        "BALLDONTLIE id 1028245237 records David Jones on San Antonio, college "
        "Memphis, country Dominican Republic; the report prints the full "
        "Dominican double surname",
    ),
    PlayerAlias(
        "Hayes-Davis, Nigel", "PHX", 2221, "Nigel Hayes", NAMING_CONVENTION,
        "BALLDONTLIE id 2221 records Nigel Hayes on Phoenix, college Wisconsin; "
        "Hayes-Davis is the current legal name",
    ),
    PlayerAlias(
        "Hayes-Davis, Nigel", "MIL", 2221, "Nigel Hayes", NAMING_CONVENTION,
        "same player as the Phoenix entry (BALLDONTLIE id 2221), reported for "
        "Milwaukee before any box score for that team",
    ),
    # --- reported before first box score for the team ---------------------
    PlayerAlias("Gordon, Eric", "MEM", 178, "Eric Gordon", PRE_FIRST_APPEARANCE,
                _REGISTRY_UNIQUE + " (registry has Philadelphia)"),
    PlayerAlias("Landale, Jock", "UTA", 19465326, "Jock Landale", PRE_FIRST_APPEARANCE,
                _REGISTRY_UNIQUE + " (registry has Atlanta and Memphis)"),
    PlayerAlias("Conley, Mike", "CHI", 104, "Mike Conley", PRE_FIRST_APPEARANCE,
                _REGISTRY_UNIQUE + " (registry has Minnesota)"),
    PlayerAlias("Conley, Mike", "CHA", 104, "Mike Conley", PRE_FIRST_APPEARANCE,
                _REGISTRY_UNIQUE + " (registry has Minnesota)"),
    PlayerAlias("Ball, Lonzo", "UTA", 27, "Lonzo Ball", PRE_FIRST_APPEARANCE,
                _REGISTRY_UNIQUE + " (registry has Cleveland)"),
    PlayerAlias("Terry, Dalen", "NOP", 38017719, "Dalen Terry", PRE_FIRST_APPEARANCE,
                _REGISTRY_UNIQUE + " (registry has Chicago and Philadelphia)"),
    PlayerAlias("Boucher, Chris", "UTA", 58, "Chris Boucher", PRE_FIRST_APPEARANCE,
                _REGISTRY_UNIQUE + " (registry has Boston)"),
)

VERIFIED_ALIASES: Final[tuple[PlayerAlias, ...]] = (
    ALIASES_2024_25 + ALIASES_2025_26
)

ALIASES_BY_KEY: Final[dict[tuple[str, str], PlayerAlias]] = {
    alias.key: alias for alias in VERIFIED_ALIASES
}


@dataclass(frozen=True)
class KnownUnresolved:
    """A mismatch investigated and deliberately left unresolved."""

    report_name: str
    report_team: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_name": self.report_name,
            "report_team": self.report_team,
            "reason": self.reason,
        }


#: Investigated, no canonical identifier found. Listing them keeps the absence
#: deliberate and auditable rather than looking like an oversight.
KNOWN_UNRESOLVED: Final[tuple[KnownUnresolved, ...]] = (
    KnownUnresolved(
        "Djurisic, Nikola", "ATL",
        "no BALLDONTLIE player record under any spelling searched (Djurisic, "
        "Djuri, Nikola) and no 2025-26 box-score appearance; reports list him "
        "as G League - On Assignment. Left unresolved rather than guessed.",
    ),
)


def resolve_alias(report_name: str, report_team: str | None) -> PlayerAlias | None:
    """Look up a verified alias. Exact match on both name and team only."""
    if report_team is None:
        return None
    return ALIASES_BY_KEY.get((report_name, report_team))
