"""Explicit classification of an NBA game into its competition phase.

BALLDONTLIE exposes ``postseason``, which is **not** the same thing as "is this a
regular-season game". Several kinds of game carry ``postseason=False`` while not
counting toward a team's regular-season schedule:

* **Play-In Tournament** games. There is no field marking these at all -- the
  only signal available is the calendar window between the end of the regular
  season and the start of the playoffs, declared per season in
  :mod:`nba_prediction_market.ingestion.season_metadata`.
* The **NBA Cup Championship** game, which *is* identifiable from the API via
  ``ist_stage == "Championship"``. Every other NBA Cup game (group play,
  quarterfinals, semifinals) does count toward the regular season.
* **Exhibitions against non-NBA opponents**, identifiable because one side's
  team id is not one of the 30 franchises.

``postseason`` is also unreliable in the other direction: it is ``True`` for
play-in games in 2019-20 and 2021-22, ``False`` from 2022-23 onward, and *both*
within 2020-21. The declared play-in window therefore takes precedence over it.

Treating ``postseason == False`` as "regular season" silently admitted play-in
games into the 2025-26 modelling set (1,236 instead of 1,230).

**Era awareness matters.** Play-in did not exist before 2019-20 and the NBA Cup
did not exist before 2023-24, so those classifications are only reachable in
seasons whose declared metadata says they applied. Modern assumptions are never
projected onto older seasons.

Declared boundaries are audited, not trusted: :func:`verify_regular_season`
re-derives each season's expected structure from the data, so a wrong date fails
loudly instead of shifting the dataset.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Final

from nba_prediction_market.ingestion.season_metadata import (
    NBA_TEAM_COUNT,
    STANDARD_GAMES_PER_TEAM,
    STANDARD_REGULAR_SEASON_GAMES,
    SeasonInfo,
    season_info,
)
from nba_prediction_market.matching.franchises import is_nba_franchise

logger = logging.getLogger(__name__)

PHASE_REGULAR_SEASON: Final = "regular_season"
PHASE_PLAY_IN: Final = "play_in"
PHASE_PLAYOFFS: Final = "playoffs"
PHASE_NBA_CUP_CHAMPIONSHIP: Final = "nba_cup_championship"
#: A positively-identified non-regular-season game that is none of the above --
#: currently exhibitions against non-NBA opponents.
PHASE_OTHER_SPECIAL: Final = "other_special"
#: Used when a season's metadata is undeclared, a date is missing, or a game
#: falls in no known window. Never a silent fallback to ``regular_season``.
PHASE_UNCLASSIFIED: Final = "unclassified"

GAME_PHASES: Final[tuple[str, ...]] = (
    PHASE_REGULAR_SEASON,
    PHASE_PLAY_IN,
    PHASE_PLAYOFFS,
    PHASE_NBA_CUP_CHAMPIONSHIP,
    PHASE_OTHER_SPECIAL,
    PHASE_UNCLASSIFIED,
)

#: ``ist_stage`` value marking the one NBA Cup game excluded from the regular
#: season. Group/quarterfinal/semifinal games all count and are not listed.
IST_STAGE_CHAMPIONSHIP: Final = "Championship"

# Re-exported for callers that reason about a standard season.
REGULAR_SEASON_GAMES_PER_TEAM: Final = STANDARD_GAMES_PER_TEAM
EXPECTED_REGULAR_SEASON_GAMES: Final = STANDARD_REGULAR_SEASON_GAMES

__all__ = [
    "EXPECTED_REGULAR_SEASON_GAMES",
    "GAME_PHASES",
    "IST_STAGE_CHAMPIONSHIP",
    "NBA_TEAM_COUNT",
    "PHASE_NBA_CUP_CHAMPIONSHIP",
    "PHASE_OTHER_SPECIAL",
    "PHASE_PLAYOFFS",
    "PHASE_PLAY_IN",
    "PHASE_REGULAR_SEASON",
    "PHASE_UNCLASSIFIED",
    "REGULAR_SEASON_GAMES_PER_TEAM",
    "classify_game_phase",
    "verify_regular_season",
]


def classify_game_phase(
    *,
    game_date: date | None,
    postseason: bool | None,
    ist_stage: str | None,
    season: int | None,
    home_team_source_id: int | None = None,
    visitor_team_source_id: int | None = None,
) -> str:
    """Classify one game into a :data:`GAME_PHASES` value.

    Priority order, most authoritative signal first:

    1. A non-NBA opponent (team id outside the 30 franchises) -> exhibition.
    2. The season's declared play-in window, **only** for seasons that had one.
       This deliberately outranks ``postseason``, which is inconsistent for
       play-in games across (and even within) seasons.
    3. ``postseason`` -- the API's own playoff flag.
    4. ``ist_stage == "Championship"`` -- the NBA Cup final, explicitly; or the
       season's declared Cup final date, for the seasons where the API leaves
       ``ist_stage`` null.
    5. On or before the declared regular-season end -> regular season.

    Anything else -- an undeclared season, a missing date, or a game in a gap
    between declared windows -- returns ``unclassified``. It is never assumed to
    be a regular-season game.

    Team ids are optional so existing callers keep working; when omitted, the
    exhibition check is simply skipped.
    """
    for team_id in (home_team_source_id, visitor_team_source_id):
        if team_id is not None and not is_nba_franchise(team_id):
            return PHASE_OTHER_SPECIAL

    info = season_info(season) if season is not None else None

    # The declared play-in window outranks `postseason`, because that flag is
    # demonstrably unreliable for play-in games: True for all of 2019-20 and
    # 2021-22, False from 2022-23 onward, and *both* within 2020-21 (5 True,
    # 1 False). The window is exact and verified, so it wins. Playoffs cannot
    # fall inside it -- SeasonInfo enforces play_in_end < playoffs_start.
    if (
        info is not None
        and info.has_play_in
        and game_date is not None
        and info.play_in_start <= game_date <= info.play_in_end
    ):
        return PHASE_PLAY_IN

    if postseason:
        return PHASE_PLAYOFFS
    if ist_stage is not None and str(ist_stage).strip() == IST_STAGE_CHAMPIONSHIP:
        return PHASE_NBA_CUP_CHAMPIONSHIP
    if game_date is None or season is None or info is None:
        return PHASE_UNCLASSIFIED
    # ist_stage is only populated for 2025-26, so the Cup finals of 2023-24 and
    # 2024-25 carry no marker. Each is a standalone event -- the only game played
    # league-wide that day -- so a declared date identifies it unambiguously.
    if info.nba_cup_final_date is not None and game_date == info.nba_cup_final_date:
        return PHASE_NBA_CUP_CHAMPIONSHIP
    if info.regular_season_start <= game_date <= info.regular_season_end:
        return PHASE_REGULAR_SEASON
    return PHASE_UNCLASSIFIED


def verify_regular_season(games: list[dict[str, Any]], season: int) -> dict[str, Any]:
    """Audit games classified ``regular_season`` against the season's structure.

    Uses the season's *declared* expectations rather than assuming 30x82/2:
    a shortened season is checked against its own totals, and a season with no
    uniform structure (2019-20) is checked only on what can be asserted.

    Returns a diagnostic dict and never raises -- an in-progress or genuinely
    irregular season legitimately fails these counts, and the caller decides
    what to do about it.
    """
    info: SeasonInfo | None = season_info(season)
    regular = [g for g in games if g.get("game_phase") == PHASE_REGULAR_SEASON]

    per_team: dict[str, int] = {}
    for game in regular:
        for key in ("home_team_code", "visitor_team_code"):
            code = game.get(key)
            if code:
                per_team[code] = per_team.get(code, 0) + 1

    expected_total = info.expected_regular_season_games if info else None
    expected_per_team = info.expected_games_per_team if info else None
    exceptions = dict(info.known_team_game_count_exceptions) if info else {}

    wrong: dict[str, int] = {}
    if expected_per_team is not None:
        for code, count in per_team.items():
            allowed = exceptions.get(code, expected_per_team)
            if count != allowed:
                wrong[code] = count

    phase_counts: dict[str, int] = {}
    for game in games:
        phase = str(game.get("game_phase"))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    checks = {
        "metadata_declared": info is not None,
        "team_count_matches": len(per_team) == NBA_TEAM_COUNT,
        "total_matches_expected": (
            expected_total is None or len(regular) == expected_total
        ),
        "games_per_team_uniform": not wrong,
    }
    verified = all(checks.values())

    if not verified:
        logger.warning(
            "Regular-season invariant not satisfied for season %s: %d games "
            "(expected %s), %d teams, %d with unexpected game counts",
            season, len(regular), expected_total, len(per_team), len(wrong),
        )

    return {
        "season": season,
        "structure": info.structure if info else None,
        "unusual_reason": info.unusual_reason if info else None,
        "regular_season_games": len(regular),
        "expected_regular_season_games": expected_total,
        "teams": len(per_team),
        "games_per_team_expected": expected_per_team,
        "games_per_team_min": min(per_team.values()) if per_team else None,
        "games_per_team_max": max(per_team.values()) if per_team else None,
        "teams_with_unexpected_game_count": wrong,
        "phase_counts": phase_counts,
        # Retained key name for the Phase 2 report's schema.
        "boundaries_declared": info is not None,
        "checks": checks,
        "verified": verified,
    }
