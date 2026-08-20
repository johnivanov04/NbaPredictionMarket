"""Normalize BALLDONTLIE game payloads into a clean, typed table."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from nba_prediction_market.config import season_label
from nba_prediction_market.ingestion.game_phase import (
    PHASE_UNCLASSIFIED,
    classify_game_phase,
    verify_regular_season,
)
from nba_prediction_market.matching.team_names import resolve_team

logger = logging.getLogger(__name__)

#: Status values that mean "this game is over and the score is final".
FINAL_STATUS_STATES = frozenset({"final"})

NBA_GAME_COLUMNS: list[str] = [
    "source",
    "source_game_id",
    "season",
    "season_label",
    "game_date",
    "tipoff_utc",
    "status",
    "status_state",
    "is_final",
    "period",
    "postseason",
    "ist_stage",
    "game_phase",
    "postponed",
    "home_team_id",
    "home_team_abbreviation",
    "home_team_full_name",
    "home_team_code",
    "home_score",
    "visitor_team_id",
    "visitor_team_abbreviation",
    "visitor_team_full_name",
    "visitor_team_code",
    "visitor_score",
    "home_win",
    "matchup_key",
]


class SeasonVerificationError(RuntimeError):
    """Raised when returned games do not look like the requested season."""


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_utc_datetime(value: Any) -> datetime | None:
    """Parse an ISO-8601 instant into an aware UTC datetime, or ``None``."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _team_code(team: dict[str, Any]) -> str | None:
    """Resolve a BALLDONTLIE team blob to a canonical abbreviation.

    Tries the abbreviation first, then the full name; never guesses.
    """
    for candidate in (team.get("abbreviation"), team.get("full_name")):
        resolution = resolve_team(candidate)
        if resolution.ok:
            return resolution.abbreviation
    return None


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_game(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten one raw game into the normalized schema.

    ``home_win`` is only populated for completed games with two integer scores
    that are not level; anything else stays ``None`` rather than being inferred.
    """
    home = raw.get("home_team") or {}
    visitor = raw.get("visitor_team") or {}
    home_code = _team_code(home)
    visitor_code = _team_code(visitor)
    home_score = _to_int(raw.get("home_team_score"))
    visitor_score = _to_int(raw.get("visitor_team_score"))

    status = raw.get("status")
    status_state = raw.get("status_state")
    is_final = str(status_state or "").lower() in FINAL_STATUS_STATES or (
        str(status or "").strip().lower() == "final"
    )

    home_win: bool | None = None
    have_scores = home_score is not None and visitor_score is not None
    if is_final and have_scores and home_score != visitor_score:
        home_win = home_score > visitor_score

    game_date = _parse_date(raw.get("date"))
    season = _to_int(raw.get("season"))
    postseason = bool(raw.get("postseason")) if raw.get("postseason") is not None else None
    ist_stage = raw.get("ist_stage")
    matchup_key: str | None = None
    if game_date and home_code and visitor_code:
        matchup_key = f"{game_date.isoformat()}|{'|'.join(sorted((home_code, visitor_code)))}"

    return {
        "source": "balldontlie",
        "source_game_id": _to_int(raw.get("id")),
        "season": season,
        "season_label": season_label(season) if season is not None else None,
        "game_date": game_date,
        "tipoff_utc": parse_utc_datetime(raw.get("datetime")),
        "status": status,
        "status_state": status_state,
        "is_final": is_final,
        "period": _to_int(raw.get("period")),
        "postseason": postseason,
        # postseason=False does NOT mean regular season -- play-in and the NBA
        # Cup final also carry it. See ingestion/game_phase.py.
        "ist_stage": ist_stage,
        "game_phase": classify_game_phase(
            game_date=game_date,
            postseason=postseason,
            ist_stage=ist_stage,
            season=season,
            home_team_source_id=_to_int(home.get("id")),
            visitor_team_source_id=_to_int(visitor.get("id")),
        ),
        "postponed": bool(raw.get("postponed")) if raw.get("postponed") is not None else None,
        "home_team_id": _to_int(home.get("id")),
        "home_team_abbreviation": home.get("abbreviation"),
        "home_team_full_name": home.get("full_name"),
        "home_team_code": home_code,
        "home_score": home_score,
        "visitor_team_id": _to_int(visitor.get("id")),
        "visitor_team_abbreviation": visitor.get("abbreviation"),
        "visitor_team_full_name": visitor.get("full_name"),
        "visitor_team_code": visitor_code,
        "visitor_score": visitor_score,
        "home_win": home_win,
        "matchup_key": matchup_key,
    }


def verify_season(games: list[dict[str, Any]], season: int) -> dict[str, Any]:
    """Check that the returned games really are the requested season.

    BALLDONTLIE labels a season by its starting year, so season ``2025`` must be
    the 2025-26 campaign: dates must fall in the Jul ``season`` .. Jun
    ``season+1`` window, and the schedule must actually start in the autumn of
    ``season``. Raises rather than letting a mislabelled pull through.
    """
    dates = [g["game_date"] for g in games if g.get("game_date")]
    if not dates:
        raise SeasonVerificationError(
            f"No parseable game dates returned for season {season}; cannot verify the season."
        )
    first, last = min(dates), max(dates)
    seasons = {g["season"] for g in games if g.get("season") is not None}

    problems: list[str] = []
    if seasons != {season}:
        problems.append(f"expected every game to carry season={season}, saw {sorted(seasons)}")
    if not (date(season, 7, 1) <= first <= date(season + 1, 6, 30)):
        problems.append(f"first game date {first} falls outside the {season_label(season)} window")
    if not (date(season, 7, 1) <= last <= date(season + 1, 6, 30)):
        problems.append(f"last game date {last} falls outside the {season_label(season)} window")
    if first.year != season or first.month < 9:
        problems.append(
            f"first game date {first} does not look like the start of {season_label(season)} "
            "(expected autumn of the season's first year)"
        )
    if problems:
        raise SeasonVerificationError(
            f"Season verification failed for {season_label(season)}: " + "; ".join(problems)
        )

    return {
        "season": season,
        "season_label": season_label(season),
        "game_count": len(games),
        "first_game_date": first.isoformat(),
        "last_game_date": last.isoformat(),
        "verified": True,
    }


def build_games_frame(raw_games: list[dict[str, Any]], season: int) -> pd.DataFrame:
    """Normalize, validate, and de-duplicate raw games into a DataFrame."""
    normalized = [normalize_game(game) for game in raw_games]

    missing_id = [g for g in normalized if g["source_game_id"] is None]
    if missing_id:
        raise ValueError(f"{len(missing_id)} BALLDONTLIE games came back without an id")

    unresolved = [
        g for g in normalized if g["home_team_code"] is None or g["visitor_team_code"] is None
    ]
    if unresolved:
        sample = sorted(
            {
                str(g[key])
                for g in unresolved[:20]
                for key in ("home_team_full_name", "visitor_team_full_name")
            }
        )[:10]
        raise ValueError(
            f"{len(unresolved)} games reference teams that are not in the canonical map; "
            f"examples: {sample}. Add explicit aliases to team_names.py -- do not fuzzy match."
        )

    verify_season(normalized, season)

    audit = verify_regular_season(normalized, season)
    logger.info("Game phases for season %s: %s", season, audit["phase_counts"])
    if not audit["boundaries_declared"]:
        logger.warning(
            "No phase boundaries declared for season %s; play-in games cannot be "
            "distinguished and are marked %r. Add an entry to "
            "SEASON_PHASE_BOUNDARIES in ingestion/game_phase.py.",
            season, PHASE_UNCLASSIFIED,
        )
    elif not audit["verified"]:
        logger.warning(
            "Regular-season invariant failed for season %s: %d games across %d teams "
            "(expected %d / %d). Check SEASON_PHASE_BOUNDARIES.",
            season, audit["regular_season_games"], audit["teams"],
            audit["expected_regular_season_games"], audit["games_per_team_expected"],
        )

    frame = pd.DataFrame(normalized, columns=NBA_GAME_COLUMNS)
    before = len(frame)
    frame = frame.drop_duplicates(subset=["source_game_id"], keep="first")
    if len(frame) != before:
        logger.warning("Dropped %d duplicate BALLDONTLIE game ids", before - len(frame))
    return frame.sort_values(["game_date", "source_game_id"], kind="stable").reset_index(drop=True)
