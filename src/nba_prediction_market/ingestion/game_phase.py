"""Explicit classification of an NBA game into its competition phase.

BALLDONTLIE exposes ``postseason``, which is **not** the same thing as "is this a
regular-season game". Two kinds of game carry ``postseason=False`` while not
counting toward the 82-game regular season:

* **Play-In Tournament** games. There is no field marking these at all -- the
  only signal available is the calendar window between the end of the regular
  season and the start of the playoffs, so that window is declared per season in
  :data:`SEASON_PHASE_BOUNDARIES` rather than hard-coded at a call site.
* The **NBA Cup Championship** game. This one *is* identifiable from the API:
  ``ist_stage == "Championship"``. Every other NBA Cup game (group play,
  quarterfinals, semifinals) does count toward the regular season, so only the
  final is separated out.

Treating ``postseason == False`` as "regular season" silently admitted both into
the 2025-26 modelling set (1,236 games instead of 1,230).

The declared boundaries are audited, not trusted: :func:`verify_regular_season`
re-derives the invariant that 30 teams each play exactly 82 games, so a wrong
date in the table below fails loudly instead of shifting the dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

logger = logging.getLogger(__name__)

PHASE_REGULAR_SEASON: Final = "regular_season"
PHASE_PLAY_IN: Final = "play_in"
PHASE_PLAYOFFS: Final = "playoffs"
PHASE_NBA_CUP_CHAMPIONSHIP: Final = "nba_cup_championship"
#: Used when a season's boundaries are undeclared, or a game falls in no known
#: window. Never a silent fallback to ``regular_season``.
PHASE_UNCLASSIFIED: Final = "unclassified"

GAME_PHASES: Final[tuple[str, ...]] = (
    PHASE_REGULAR_SEASON,
    PHASE_PLAY_IN,
    PHASE_PLAYOFFS,
    PHASE_NBA_CUP_CHAMPIONSHIP,
    PHASE_UNCLASSIFIED,
)

#: ``ist_stage`` value marking the one NBA Cup game excluded from the regular
#: season. Group/quarterfinal/semifinal games all count and are not listed.
IST_STAGE_CHAMPIONSHIP: Final = "Championship"

#: League structure, used by the audit in :func:`verify_regular_season`.
NBA_TEAM_COUNT: Final = 30
REGULAR_SEASON_GAMES_PER_TEAM: Final = 82
EXPECTED_REGULAR_SEASON_GAMES: Final = NBA_TEAM_COUNT * REGULAR_SEASON_GAMES_PER_TEAM // 2  # 1230


@dataclass(frozen=True)
class SeasonPhaseBoundaries:
    """Calendar boundaries separating a season's phases.

    Dates are inclusive game dates in the league's local calendar (the same
    ``date`` both sources label games with -- no timezone shifting).
    """

    regular_season_end: date
    play_in_start: date
    play_in_end: date
    playoffs_start: date
    source: str

    def __post_init__(self) -> None:
        if not self.regular_season_end < self.play_in_start:
            raise ValueError("play-in must start after the regular season ends")
        if not self.play_in_start <= self.play_in_end:
            raise ValueError("play-in window is inverted")
        if not self.play_in_end < self.playoffs_start:
            raise ValueError("playoffs must start after the play-in ends")


#: Declared per season. Adding a season is a deliberate, reviewable edit; an
#: undeclared season classifies as ``unclassified`` rather than being guessed.
SEASON_PHASE_BOUNDARIES: Final[dict[int, SeasonPhaseBoundaries]] = {
    2025: SeasonPhaseBoundaries(
        regular_season_end=date(2026, 4, 12),
        play_in_start=date(2026, 4, 14),
        play_in_end=date(2026, 4, 17),
        playoffs_start=date(2026, 4, 18),
        source=(
            "2025-26 NBA calendar: regular season ended 2026-04-12, Play-In "
            "2026-04-14..17, playoffs opened 2026-04-18. Verified against the "
            "ingested schedule: 1230 regular-season games (82 per team), 6 "
            "play-in games, and the first postseason=True game on 2026-04-18."
        ),
    ),
}


def phase_boundaries(season: int) -> SeasonPhaseBoundaries | None:
    """Declared boundaries for ``season``, or ``None`` if undeclared."""
    return SEASON_PHASE_BOUNDARIES.get(season)


def classify_game_phase(
    *,
    game_date: date | None,
    postseason: bool | None,
    ist_stage: str | None,
    season: int | None,
) -> str:
    """Classify one game into a :data:`GAME_PHASES` value.

    Priority order, most authoritative signal first:

    1. ``postseason`` is the API's own playoff flag.
    2. ``ist_stage == "Championship"`` identifies the NBA Cup final explicitly.
    3. The declared play-in window for the season.
    4. On or before the declared regular-season end -> regular season.

    Anything else -- an undeclared season, a missing date, or a game in a gap
    between declared windows -- returns ``unclassified``. It is never assumed to
    be a regular-season game.
    """
    if postseason:
        return PHASE_PLAYOFFS
    if ist_stage is not None and str(ist_stage).strip() == IST_STAGE_CHAMPIONSHIP:
        return PHASE_NBA_CUP_CHAMPIONSHIP
    if game_date is None or season is None:
        return PHASE_UNCLASSIFIED

    boundaries = phase_boundaries(season)
    if boundaries is None:
        return PHASE_UNCLASSIFIED
    if boundaries.play_in_start <= game_date <= boundaries.play_in_end:
        return PHASE_PLAY_IN
    if game_date <= boundaries.regular_season_end:
        return PHASE_REGULAR_SEASON
    return PHASE_UNCLASSIFIED


def verify_regular_season(
    games: list[dict[str, Any]], season: int
) -> dict[str, Any]:
    """Audit games classified ``regular_season`` against the league's structure.

    Re-derives "30 teams x 82 games / 2 = 1230" from the data, so a wrong date in
    :data:`SEASON_PHASE_BOUNDARIES` surfaces as a failed invariant instead of a
    quietly larger or smaller dataset. Returns a diagnostic dict and never
    raises -- an in-progress season legitimately fails these counts, and the
    caller decides what to do about it.
    """
    regular = [g for g in games if g.get("game_phase") == PHASE_REGULAR_SEASON]
    per_team: dict[str, int] = {}
    for game in regular:
        for key in ("home_team_code", "visitor_team_code"):
            code = game.get(key)
            if code:
                per_team[code] = per_team.get(code, 0) + 1

    wrong = {code: n for code, n in per_team.items() if n != REGULAR_SEASON_GAMES_PER_TEAM}
    phase_counts: dict[str, int] = {}
    for game in games:
        phase = str(game.get("game_phase"))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    verified = (
        len(regular) == EXPECTED_REGULAR_SEASON_GAMES
        and len(per_team) == NBA_TEAM_COUNT
        and not wrong
    )
    if not verified:
        logger.warning(
            "Regular-season invariant not satisfied for season %s: %d games "
            "(expected %d), %d teams, %d with a game count other than %d",
            season, len(regular), EXPECTED_REGULAR_SEASON_GAMES,
            len(per_team), len(wrong), REGULAR_SEASON_GAMES_PER_TEAM,
        )
    return {
        "season": season,
        "regular_season_games": len(regular),
        "expected_regular_season_games": EXPECTED_REGULAR_SEASON_GAMES,
        "teams": len(per_team),
        "games_per_team_expected": REGULAR_SEASON_GAMES_PER_TEAM,
        "teams_with_unexpected_game_count": wrong,
        "phase_counts": phase_counts,
        "boundaries_declared": phase_boundaries(season) is not None,
        "verified": verified,
    }
