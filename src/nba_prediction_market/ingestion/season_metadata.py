"""Per-season structure: what each NBA season actually looked like.

The 30x82/2 = 1,230 invariant is only correct for a *standard* season. Across
2006-07..2025-26 several seasons are not standard, and forcing them to 1,230
would either drop real games or invent missing ones. Each season therefore
declares its own expected structure here, with the reason and provenance
attached, rather than having dates and counts scattered through pipeline code.

Two era facts also live here, because applying modern assumptions to older
seasons is the main way this dataset could silently go wrong:

* **Play-In** began in the 2019-20 bubble (a one-off format) and has run in its
  current form since 2020-21. Before that there is no play-in round at all, so
  no date window is declared and no game can be classified ``play_in``.
* **NBA Cup** (In-Season Tournament) began in 2023-24. Before that ``ist_stage``
  is always null, so no game can be classified ``nba_cup_championship``.
  **BALLDONTLIE only populates ``ist_stage`` for 2025-26**, so the Cup finals of
  2023-24 and 2024-25 carry no marker at all and were being counted toward the
  regular season (1,231 games, with the two finalists on 83). Each final is a
  standalone event -- the only game played league-wide that day -- so the date is
  declared per season and is self-validating.

Every declared expectation is *audited* against the ingested data rather than
trusted -- see ``verify_season_structure``. A wrong number here fails loudly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final

logger = logging.getLogger(__name__)

#: Seasons this phase covers (BALLDONTLIE season = starting year).
FIRST_HISTORICAL_SEASON: Final = 2006
LAST_HISTORICAL_SEASON: Final = 2025
HISTORICAL_SEASONS: Final[tuple[int, ...]] = tuple(
    range(FIRST_HISTORICAL_SEASON, LAST_HISTORICAL_SEASON + 1)
)

NBA_TEAM_COUNT: Final = 30
STANDARD_GAMES_PER_TEAM: Final = 82
STANDARD_REGULAR_SEASON_GAMES: Final = NBA_TEAM_COUNT * STANDARD_GAMES_PER_TEAM // 2  # 1230

# Season classification labels used in the audit report.
STRUCTURE_STANDARD: Final = "standard"
STRUCTURE_SHORTENED: Final = "shortened"
STRUCTURE_INTERRUPTED: Final = "interrupted"

#: First season in which a play-in round existed at all (2019-20 bubble format).
FIRST_PLAY_IN_SEASON: Final = 2019
#: First season of the NBA Cup / In-Season Tournament.
FIRST_NBA_CUP_SEASON: Final = 2023


@dataclass(frozen=True)
class SeasonInfo:
    """Declared structure of one season.

    ``expected_regular_season_games`` and ``expected_games_per_team`` may be
    ``None`` when a season's schedule was not uniform (teams played differing
    numbers of games), in which case the audit checks what it can and reports
    the rest rather than asserting a number nobody can justify.
    """

    season: int
    structure: str
    regular_season_start: date
    regular_season_end: date
    expected_regular_season_games: int | None = STANDARD_REGULAR_SEASON_GAMES
    expected_games_per_team: int | None = STANDARD_GAMES_PER_TEAM
    play_in_start: date | None = None
    play_in_end: date | None = None
    playoffs_start: date | None = None
    #: Date of the NBA Cup final. Declared because ``ist_stage`` is only
    #: populated for 2025-26 -- see the module docstring.
    nba_cup_final_date: date | None = None
    unusual_reason: str | None = None
    notes: str = ""
    #: Teams that legitimately played a non-uniform number of games.
    known_team_game_count_exceptions: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.regular_season_start > self.regular_season_end:
            raise ValueError(f"season {self.season}: regular season window is inverted")
        if (self.play_in_start is None) != (self.play_in_end is None):
            raise ValueError(f"season {self.season}: play-in window is half-declared")
        if self.play_in_start is not None:
            if self.play_in_start > self.play_in_end:
                raise ValueError(f"season {self.season}: play-in window is inverted")
            if self.regular_season_end >= self.play_in_start:
                raise ValueError(f"season {self.season}: play-in must follow the regular season")
            if self.playoffs_start is not None and self.play_in_end >= self.playoffs_start:
                raise ValueError(f"season {self.season}: playoffs must follow the play-in")
        if self.structure != STRUCTURE_STANDARD and not self.unusual_reason:
            raise ValueError(f"season {self.season}: non-standard seasons must state a reason")
        if self.nba_cup_final_date is not None:
            if self.season < FIRST_NBA_CUP_SEASON:
                raise ValueError(
                    f"season {self.season}: the NBA Cup did not exist before "
                    f"{FIRST_NBA_CUP_SEASON}"
                )
            if not (
                self.regular_season_start
                <= self.nba_cup_final_date
                <= self.regular_season_end
            ):
                raise ValueError(
                    f"season {self.season}: the Cup final falls outside the season window"
                )

    @property
    def has_play_in(self) -> bool:
        return self.play_in_start is not None

    @property
    def label(self) -> str:
        return f"{self.season}-{(self.season + 1) % 100:02d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "season": self.season,
            "season_label": self.label,
            "structure": self.structure,
            "regular_season_start": self.regular_season_start.isoformat(),
            "regular_season_end": self.regular_season_end.isoformat(),
            "expected_regular_season_games": self.expected_regular_season_games,
            "expected_games_per_team": self.expected_games_per_team,
            "play_in_start": self.play_in_start.isoformat() if self.play_in_start else None,
            "play_in_end": self.play_in_end.isoformat() if self.play_in_end else None,
            "playoffs_start": self.playoffs_start.isoformat() if self.playoffs_start else None,
            "unusual_reason": self.unusual_reason,
            "notes": self.notes,
            "known_team_game_count_exceptions": dict(self.known_team_game_count_exceptions),
        }


def _standard(
    season: int,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    *,
    play_in: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    playoffs_start: tuple[int, int, int] | None = None,
    nba_cup_final: tuple[int, int, int] | None = None,
    notes: str = "",
) -> SeasonInfo:
    return SeasonInfo(
        season=season,
        structure=STRUCTURE_STANDARD,
        regular_season_start=date(*start),
        regular_season_end=date(*end),
        play_in_start=date(*play_in[0]) if play_in else None,
        play_in_end=date(*play_in[1]) if play_in else None,
        playoffs_start=date(*playoffs_start) if playoffs_start else None,
        nba_cup_final_date=date(*nba_cup_final) if nba_cup_final else None,
        notes=notes,
    )


#: Declared per season. Dates are inclusive game dates in the league's local
#: calendar -- the same ``date`` both data sources label games with.
SEASON_METADATA: Final[dict[int, SeasonInfo]] = {
    2006: _standard(2006, (2006, 10, 31), (2007, 4, 18), playoffs_start=(2007, 4, 21)),
    2007: _standard(2007, (2007, 10, 30), (2008, 4, 16), playoffs_start=(2008, 4, 19)),
    2008: _standard(2008, (2008, 10, 28), (2009, 4, 15), playoffs_start=(2009, 4, 18)),
    2009: _standard(2009, (2009, 10, 27), (2010, 4, 14), playoffs_start=(2010, 4, 17)),
    2010: _standard(2010, (2010, 10, 26), (2011, 4, 13), playoffs_start=(2011, 4, 16)),
    2011: SeasonInfo(
        season=2011,
        structure=STRUCTURE_SHORTENED,
        regular_season_start=date(2011, 12, 25),
        regular_season_end=date(2012, 4, 26),
        expected_regular_season_games=NBA_TEAM_COUNT * 66 // 2,  # 990
        expected_games_per_team=66,
        playoffs_start=date(2012, 4, 28),
        unusual_reason="lockout",
        notes=(
            "2011 NBA lockout delayed the season to 25 Dec 2011 and cut it to 66 "
            "games per team (990 total)."
        ),
    ),
    2012: SeasonInfo(
        season=2012,
        structure=STRUCTURE_INTERRUPTED,
        regular_season_start=date(2012, 10, 30),
        regular_season_end=date(2013, 4, 17),
        # One scheduled game was cancelled and never made up, so the league
        # played 1229 games and two teams finished on 81.
        expected_regular_season_games=1229,
        expected_games_per_team=82,
        known_team_game_count_exceptions={"BOS": 81, "IND": 81},
        playoffs_start=date(2013, 4, 20),
        unusual_reason="cancelled_game_boston_marathon_bombing",
        notes=(
            "Boston Celtics v Indiana Pacers, scheduled 2013-04-16 at TD Garden, "
            "was cancelled after the Boston Marathon bombing and never rescheduled. "
            "Verified in the ingested data: BOS and IND each played 81 games, they "
            "met only twice head-to-head, and only 2 games league-wide were played "
            "on 2013-04-16 against 11 and 15 on the adjacent days."
        ),
    ),
    2013: _standard(2013, (2013, 10, 29), (2014, 4, 16), playoffs_start=(2014, 4, 19)),
    2014: _standard(2014, (2014, 10, 28), (2015, 4, 15), playoffs_start=(2015, 4, 18)),
    2015: _standard(2015, (2015, 10, 27), (2016, 4, 13), playoffs_start=(2016, 4, 16)),
    2016: _standard(2016, (2016, 10, 25), (2017, 4, 12), playoffs_start=(2017, 4, 15)),
    2017: _standard(2017, (2017, 10, 17), (2018, 4, 11), playoffs_start=(2018, 4, 14)),
    2018: _standard(2018, (2018, 10, 16), (2019, 4, 10), playoffs_start=(2019, 4, 13)),
    2019: SeasonInfo(
        season=2019,
        structure=STRUCTURE_INTERRUPTED,
        regular_season_start=date(2019, 10, 22),
        regular_season_end=date(2020, 8, 14),
        # COVID-19 suspended play on 11 Mar 2020; only 22 teams resumed in the
        # Orlando bubble, so teams finished on differing game counts and the
        # season total is below 1230. No uniform expectation can be asserted.
        expected_regular_season_games=None,
        expected_games_per_team=None,
        play_in_start=date(2020, 8, 15),
        play_in_end=date(2020, 8, 16),
        playoffs_start=date(2020, 8, 17),
        unusual_reason="covid_suspension_and_bubble_restart",
        notes=(
            "Suspended 2020-03-11; resumed 2020-07-30 in the Orlando bubble with "
            "only 22 teams playing 8 seeding games each. Teams finished on 64-75 "
            "games, so neither a uniform per-team count nor a 1230 total applies. "
            "A one-off play-in (West 8/9 seed) was held 2020-08-15..16."
        ),
    ),
    2020: SeasonInfo(
        season=2020,
        structure=STRUCTURE_SHORTENED,
        regular_season_start=date(2020, 12, 22),
        regular_season_end=date(2021, 5, 16),
        expected_regular_season_games=NBA_TEAM_COUNT * 72 // 2,  # 1080
        expected_games_per_team=72,
        play_in_start=date(2021, 5, 18),
        play_in_end=date(2021, 5, 21),
        playoffs_start=date(2021, 5, 22),
        unusual_reason="covid_shortened_schedule",
        notes=(
            "COVID-shortened 72-game schedule (1080 total) starting 22 Dec 2020. "
            "First season of the modern 4-team-per-conference play-in."
        ),
    ),
    2021: _standard(
        2021, (2021, 10, 19), (2022, 4, 10),
        play_in=((2022, 4, 12), (2022, 4, 15)), playoffs_start=(2022, 4, 16),
    ),
    2022: _standard(
        2022, (2022, 10, 18), (2023, 4, 9),
        play_in=((2023, 4, 11), (2023, 4, 14)), playoffs_start=(2023, 4, 15),
    ),
    2023: _standard(
        2023, (2023, 10, 24), (2024, 4, 14),
        play_in=((2024, 4, 16), (2024, 4, 19)), playoffs_start=(2024, 4, 20),
        nba_cup_final=(2023, 12, 9),
        notes=(
            "First NBA Cup season; the Cup final (LAL 123 - IND 109) does not count "
            "toward the 82. ist_stage is null for this season, so the final is "
            "identified by date -- verified as the only game played league-wide "
            "on 2023-12-09."
        ),
    ),
    2024: _standard(
        2024, (2024, 10, 22), (2025, 4, 13),
        play_in=((2025, 4, 15), (2025, 4, 18)), playoffs_start=(2025, 4, 19),
        nba_cup_final=(2024, 12, 17),
        notes=(
            "NBA Cup season; the Cup final (MIL 97 - OKC 81) does not count toward "
            "the 82. ist_stage is null for this season, so the final is identified "
            "by date -- verified as the only game played league-wide on 2024-12-17."
        ),
    ),
    2025: _standard(
        2025, (2025, 10, 21), (2026, 4, 12),
        play_in=((2026, 4, 14), (2026, 4, 17)), playoffs_start=(2026, 4, 18),
        nba_cup_final=(2025, 12, 16),
        notes=(
            "NBA Cup season. The only season for which BALLDONTLIE populates "
            "ist_stage, so the declared Cup final date and the field agree here -- "
            "a cross-check for the two seasons where only the date is available. "
            "Verified: 1230 regular-season games (82 per team), 6 play-in, 1 Cup final."
        ),
    ),
}


class UnknownSeasonError(RuntimeError):
    """Raised when a season has no declared metadata and one is required."""


def season_info(season: int) -> SeasonInfo | None:
    """Declared metadata for ``season``, or ``None`` if undeclared."""
    return SEASON_METADATA.get(season)


def require_season_info(season: int) -> SeasonInfo:
    """Declared metadata for ``season``, raising an actionable error if absent."""
    info = season_info(season)
    if info is None:
        raise UnknownSeasonError(
            f"No declared metadata for season {season}. Add a SeasonInfo entry to "
            "SEASON_METADATA in ingestion/season_metadata.py -- a season is never "
            "classified by guessing."
        )
    return info


def season_has_play_in(season: int) -> bool:
    """Whether a play-in round existed in ``season``, per declared metadata."""
    info = season_info(season)
    return bool(info and info.has_play_in)


def season_has_nba_cup(season: int) -> bool:
    """Whether the NBA Cup existed in ``season``."""
    return season >= FIRST_NBA_CUP_SEASON
