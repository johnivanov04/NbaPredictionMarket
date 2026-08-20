"""Deterministic, auditable matching of NBA games to Kalshi game-winner events.

The join key is ``(scheduled game date, unordered pair of canonical team
codes)``. Both sources label a game by its *local* scheduled date -- BALLDONTLIE
in ``date``, Kalshi in the settlement rules -- so no timezone shifting is needed
or applied. The pair is deliberately unordered: home/away orientation is
recorded and compared afterwards rather than being required up front, so a
disagreement about orientation surfaces as a flag instead of silently dropping
an otherwise obvious match.

Rules:

* A match is only produced when exactly one game and exactly one event share a
  key. Any many-to-one or one-to-many group is reported ``ambiguous`` in full.
* A single relaxed tier (+/- 1 calendar day, same team pair) catches genuine
  date-labelling edge cases, and only fires when the pairing is mutually unique
  among the records still unmatched.
* Nothing is ever chosen arbitrarily between remaining candidates.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

STATUS_MATCHED = "matched"
STATUS_UNMATCHED_NBA = "unmatched_nba"
STATUS_UNMATCHED_KALSHI = "unmatched_kalshi"
STATUS_AMBIGUOUS = "ambiguous"

TIER_EXACT = "exact_date_and_teams"
TIER_DATE_OFFSET = "teams_and_date_within_1_day"

#: Maximum |days| the relaxed tier will bridge.
MAX_DATE_OFFSET_DAYS = 1

MATCH_COLUMNS: list[str] = [
    "match_status",
    "match_tier",
    "date_offset_days",
    "matchup_key",
    "nba_game_id",
    "nba_game_date",
    "nba_season",
    "nba_postseason",
    "nba_game_phase",
    "nba_status",
    "nba_is_final",
    "nba_tipoff_utc",
    "nba_home_team_code",
    "nba_away_team_code",
    "nba_home_score",
    "nba_away_score",
    "nba_home_win",
    "kalshi_event_ticker",
    "kalshi_event_title",
    "kalshi_event_sub_title",
    "kalshi_scheduled_game_date",
    "kalshi_home_team_code",
    "kalshi_away_team_code",
    "kalshi_home_market_ticker",
    "kalshi_away_market_ticker",
    "kalshi_market_count",
    "kalshi_home_result",
    "kalshi_away_result",
    "kalshi_open_time_utc",
    "kalshi_close_time_utc",
    "kalshi_total_volume_fp",
    "orientation_agrees",
    "settlement_agrees_with_score",
    "candidate_event_tickers",
    "candidate_nba_game_ids",
    "ambiguity_reason",
]

KALSHI_EVENT_COLUMNS: list[str] = [
    "event_ticker",
    "matchup_key",
    "scheduled_game_date",
    "home_team_code",
    "away_team_code",
    "title",
    "event_sub_title",
    "home_market_ticker",
    "away_market_ticker",
    "market_count",
    "home_result",
    "away_result",
    "open_time_utc",
    "close_time_utc",
    "total_volume_fp",
    "is_nba_matchup",
]


@dataclass(frozen=True)
class MatchSummary:
    """Counts by classification, for the report and for quick assertions."""

    matched: int
    unmatched_nba: int
    unmatched_kalshi: int
    ambiguous: int

    def to_dict(self) -> dict[str, int]:
        return {
            STATUS_MATCHED: self.matched,
            STATUS_UNMATCHED_NBA: self.unmatched_nba,
            STATUS_UNMATCHED_KALSHI: self.unmatched_kalshi,
            STATUS_AMBIGUOUS: self.ambiguous,
        }


def _first_not_null(values: pd.Series) -> Any:
    non_null = values.dropna()
    return non_null.iloc[0] if len(non_null) else None


def build_kalshi_events_frame(markets: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-team market rows into one row per Kalshi event.

    Each NBA event carries exactly two markets (one per team). The home and away
    market tickers are kept side by side so a pregame price can later be
    attached to a game without re-deriving orientation.
    """
    if markets.empty:
        return pd.DataFrame(columns=KALSHI_EVENT_COLUMNS)

    rows: list[dict[str, Any]] = []
    for event_ticker, group in markets.groupby("event_ticker", sort=True):
        home_rows = group[group["market_team_is_home"] == True]  # noqa: E712
        away_rows = group[group["market_team_is_home"] == False]  # noqa: E712
        rows.append(
            {
                "event_ticker": event_ticker,
                "matchup_key": _first_not_null(group["matchup_key"]),
                "scheduled_game_date": _first_not_null(group["scheduled_game_date"]),
                "home_team_code": _first_not_null(group["home_team_code"]),
                "away_team_code": _first_not_null(group["away_team_code"]),
                "title": _first_not_null(group["title"]),
                "event_sub_title": _first_not_null(group["event_sub_title"]),
                "home_market_ticker": (
                    home_rows["ticker"].iloc[0] if len(home_rows) == 1 else None
                ),
                "away_market_ticker": (
                    away_rows["ticker"].iloc[0] if len(away_rows) == 1 else None
                ),
                "market_count": len(group),
                "home_result": _first_not_null(home_rows["result"]) if len(home_rows) else None,
                "away_result": _first_not_null(away_rows["result"]) if len(away_rows) else None,
                "open_time_utc": _first_not_null(group["open_time_utc"]),
                "close_time_utc": _first_not_null(group["close_time_utc"]),
                "total_volume_fp": float(group["volume_fp"].fillna(0).sum()),
                "is_nba_matchup": bool(group["is_nba_matchup"].all()),
            }
        )
    frame = pd.DataFrame(rows, columns=KALSHI_EVENT_COLUMNS)
    return frame.sort_values(["scheduled_game_date", "event_ticker"], kind="stable").reset_index(
        drop=True
    )


def _nba_fields(game: dict[str, Any]) -> dict[str, Any]:
    return {
        "nba_game_id": game.get("source_game_id"),
        "nba_game_date": game.get("game_date"),
        "nba_season": game.get("season"),
        "nba_postseason": game.get("postseason"),
        "nba_game_phase": game.get("game_phase"),
        "nba_status": game.get("status"),
        "nba_is_final": game.get("is_final"),
        "nba_tipoff_utc": game.get("tipoff_utc"),
        "nba_home_team_code": game.get("home_team_code"),
        "nba_away_team_code": game.get("visitor_team_code"),
        "nba_home_score": game.get("home_score"),
        "nba_away_score": game.get("visitor_score"),
        "nba_home_win": game.get("home_win"),
    }


def _kalshi_fields(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "kalshi_event_ticker": event.get("event_ticker"),
        "kalshi_event_title": event.get("title"),
        "kalshi_event_sub_title": event.get("event_sub_title"),
        "kalshi_scheduled_game_date": event.get("scheduled_game_date"),
        "kalshi_home_team_code": event.get("home_team_code"),
        "kalshi_away_team_code": event.get("away_team_code"),
        "kalshi_home_market_ticker": event.get("home_market_ticker"),
        "kalshi_away_market_ticker": event.get("away_market_ticker"),
        "kalshi_market_count": event.get("market_count"),
        "kalshi_home_result": event.get("home_result"),
        "kalshi_away_result": event.get("away_result"),
        "kalshi_open_time_utc": event.get("open_time_utc"),
        "kalshi_close_time_utc": event.get("close_time_utc"),
        "kalshi_total_volume_fp": event.get("total_volume_fp"),
    }


def _orientation_agrees(game: dict[str, Any], event: dict[str, Any]) -> bool | None:
    if not all(
        (
            game.get("home_team_code"),
            game.get("visitor_team_code"),
            event.get("home_team_code"),
            event.get("away_team_code"),
        )
    ):
        return None
    return (
        game["home_team_code"] == event["home_team_code"]
        and game["visitor_team_code"] == event["away_team_code"]
    )


def _settlement_agrees(game: dict[str, Any], event: dict[str, Any]) -> bool | None:
    """Compare the Kalshi settlement to the final score, when both are known."""
    home_win = game.get("home_win")
    home_result = event.get("home_result")
    if home_win is None or home_result not in {"yes", "no"}:
        return None
    if _orientation_agrees(game, event) is not True:
        return None
    return bool(home_win) == (home_result == "yes")


def _norm_key(value: Any) -> str | None:
    """Coerce a matchup key to ``str`` or ``None``.

    pandas surfaces missing object-dtype values as ``NaN`` (a float), which would
    otherwise make the key set unsortable and, worse, let two distinct ``NaN``
    values compare unequal while grouping.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _pair_key(a: Any, b: Any) -> tuple[str, str] | None:
    """Unordered team-pair key, or ``None`` if either side is missing."""
    left, right = _norm_key(a), _norm_key(b)
    if not left or not right:
        return None
    return (left, right) if left <= right else (right, left)


def _as_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, date):
        return value
    try:
        converted = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    return None if pd.isna(converted) else converted.date()


def match_games_to_markets(
    games: pd.DataFrame,
    kalshi_events: pd.DataFrame,
) -> tuple[pd.DataFrame, MatchSummary]:
    """Match NBA games to Kalshi events and classify every record on both sides."""
    game_records: list[dict[str, Any]] = games.to_dict("records")
    event_records: list[dict[str, Any]] = kalshi_events.to_dict("records")

    games_by_key: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for game in game_records:
        games_by_key[_norm_key(game.get("matchup_key"))].append(game)
    events_by_key: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for event in event_records:
        events_by_key[_norm_key(event.get("matchup_key"))].append(event)

    rows: list[dict[str, Any]] = []
    pending_games: list[dict[str, Any]] = []
    pending_events: list[dict[str, Any]] = []

    # --- Tier 1: exact (date, unordered team pair).
    for key in sorted(set(games_by_key) | set(events_by_key), key=lambda k: (k is None, k)):
        key_games = games_by_key.get(key, [])
        key_events = events_by_key.get(key, [])

        if key is None:
            # Records whose key could not be derived can never join on it.
            pending_games.extend(key_games)
            pending_events.extend(key_events)
            continue

        if len(key_games) == 1 and len(key_events) == 1:
            rows.append(_matched_row(key_games[0], key_events[0], TIER_EXACT, 0))
        elif len(key_games) == 1 and not key_events:
            pending_games.append(key_games[0])
        elif len(key_events) == 1 and not key_games:
            pending_events.append(key_events[0])
        elif not key_games:
            pending_events.extend(key_events)
        elif not key_events:
            pending_games.extend(key_games)
        else:
            reason = (
                f"{len(key_games)} NBA games and {len(key_events)} Kalshi events share "
                f"matchup key {key}"
            )
            candidates_e = sorted(str(e.get("event_ticker")) for e in key_events)
            candidates_g = sorted(str(g.get("source_game_id")) for g in key_games)
            for game in key_games:
                rows.append(
                    _ambiguous_row(game=game, event=None, key=key, reason=reason,
                                   candidate_events=candidates_e, candidate_games=candidates_g)
                )
            for event in key_events:
                rows.append(
                    _ambiguous_row(game=None, event=event, key=key, reason=reason,
                                   candidate_events=candidates_e, candidate_games=candidates_g)
                )

    # --- Tier 2: same team pair, date within +/-1 day, mutually unique.
    events_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in pending_events:
        pair = _pair_key(event.get("home_team_code"), event.get("away_team_code"))
        if pair is not None:
            events_by_pair[pair].append(event)

    games_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for game in pending_games:
        pair = _pair_key(game.get("home_team_code"), game.get("visitor_team_code"))
        if pair is not None:
            games_by_pair[pair].append(game)

    def near_candidates(
        pair: tuple[str, str] | None,
        anchor: date | None,
        pool: dict[tuple[str, str], list[dict[str, Any]]],
        date_field: str,
    ) -> list[dict[str, Any]]:
        if pair is None or anchor is None:
            return []
        out = []
        for other in pool.get(pair, []):
            other_date = _as_date(other.get(date_field))
            if other_date is None:
                continue
            if abs((other_date - anchor).days) <= MAX_DATE_OFFSET_DAYS:
                out.append(other)
        return out

    consumed_events: set[int] = set()
    consumed_games: set[int] = set()

    for game in pending_games:
        pair = _pair_key(game.get("home_team_code"), game.get("visitor_team_code"))
        game_date = _as_date(game.get("game_date"))
        candidates = [
            e
            for e in near_candidates(pair, game_date, events_by_pair, "scheduled_game_date")
            if id(e) not in consumed_events
        ]
        if len(candidates) != 1:
            continue
        event = candidates[0]
        # Require mutual uniqueness: the event must not also be near another game.
        back = [
            g
            for g in near_candidates(
                pair, _as_date(event.get("scheduled_game_date")), games_by_pair, "game_date"
            )
            if id(g) not in consumed_games
        ]
        if len(back) != 1 or id(back[0]) != id(game):
            continue
        offset = (_as_date(event.get("scheduled_game_date")) - game_date).days
        rows.append(_matched_row(game, event, TIER_DATE_OFFSET, offset))
        consumed_events.add(id(event))
        consumed_games.add(id(game))

    # --- Whatever is left is genuinely unmatched (or ambiguous within tier 2).
    for game in pending_games:
        if id(game) in consumed_games:
            continue
        pair = _pair_key(game.get("home_team_code"), game.get("visitor_team_code"))
        leftovers = [
            e
            for e in near_candidates(
                pair, _as_date(game.get("game_date")), events_by_pair, "scheduled_game_date"
            )
            if id(e) not in consumed_events
        ]
        if len(leftovers) > 1:
            rows.append(
                _ambiguous_row(
                    game=game, event=None, key=game.get("matchup_key"),
                    reason=(
                        f"{len(leftovers)} Kalshi events for {pair} within "
                        f"{MAX_DATE_OFFSET_DAYS} day(s) of {game.get('game_date')}"
                    ),
                    candidate_events=sorted(str(e.get("event_ticker")) for e in leftovers),
                    candidate_games=[str(game.get("source_game_id"))],
                )
            )
        else:
            rows.append(_unmatched_nba_row(game))

    for event in pending_events:
        if id(event) in consumed_events:
            continue
        rows.append(_unmatched_kalshi_row(event))

    frame = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    frame = frame.sort_values(
        ["match_status", "nba_game_date", "kalshi_scheduled_game_date", "nba_game_id"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    counts = frame["match_status"].value_counts().to_dict()
    summary = MatchSummary(
        matched=int(counts.get(STATUS_MATCHED, 0)),
        unmatched_nba=int(counts.get(STATUS_UNMATCHED_NBA, 0)),
        unmatched_kalshi=int(counts.get(STATUS_UNMATCHED_KALSHI, 0)),
        ambiguous=int(counts.get(STATUS_AMBIGUOUS, 0)),
    )
    return frame, summary


def _base_row() -> dict[str, Any]:
    return dict.fromkeys(MATCH_COLUMNS)


def _matched_row(
    game: dict[str, Any], event: dict[str, Any], tier: str, offset: int
) -> dict[str, Any]:
    row = _base_row()
    row.update(_nba_fields(game))
    row.update(_kalshi_fields(event))
    row.update(
        {
            "match_status": STATUS_MATCHED,
            "match_tier": tier,
            "date_offset_days": offset,
            "matchup_key": game.get("matchup_key"),
            "orientation_agrees": _orientation_agrees(game, event),
            "settlement_agrees_with_score": _settlement_agrees(game, event),
        }
    )
    return row


def _unmatched_nba_row(game: dict[str, Any]) -> dict[str, Any]:
    row = _base_row()
    row.update(_nba_fields(game))
    row.update({"match_status": STATUS_UNMATCHED_NBA, "matchup_key": game.get("matchup_key")})
    return row


def _unmatched_kalshi_row(event: dict[str, Any]) -> dict[str, Any]:
    row = _base_row()
    row.update(_kalshi_fields(event))
    row.update({"match_status": STATUS_UNMATCHED_KALSHI, "matchup_key": event.get("matchup_key")})
    return row


def _ambiguous_row(
    *,
    game: dict[str, Any] | None,
    event: dict[str, Any] | None,
    key: Any,
    reason: str,
    candidate_events: list[str],
    candidate_games: list[str],
) -> dict[str, Any]:
    row = _base_row()
    if game is not None:
        row.update(_nba_fields(game))
    if event is not None:
        row.update(_kalshi_fields(event))
    row.update(
        {
            "match_status": STATUS_AMBIGUOUS,
            "matchup_key": key,
            "ambiguity_reason": reason,
            "candidate_event_tickers": ",".join(candidate_events),
            "candidate_nba_game_ids": ",".join(candidate_games),
        }
    )
    return row
