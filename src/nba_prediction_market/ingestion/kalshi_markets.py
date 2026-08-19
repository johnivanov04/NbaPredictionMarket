"""Normalize Kalshi ``KXNBAGAME`` market metadata into a clean table.

Observed quirks this module is built around (all verified against live data on
2026-08-19; see README "Verified API behaviour"):

* ``no_sub_title`` **equals** ``yes_sub_title`` on every market -- it is *not*
  the opposing team, so it cannot be used to derive an opponent.
* ``occurrence_datetime`` is only populated for postseason markets (~6% of the
  archive), so it cannot be the primary date field.
* ``title`` uses two different formats: ``"A at B Winner?"`` (home/away implied)
  for later markets and ``"A vs B Winner?"`` (no orientation) for earlier ones.
* ``yes_sub_title`` carries city labels that are ambiguous between the two Los
  Angeles franchises (``"Los Angeles"``, ``"LA"``).

Team identity and orientation therefore come from the event's ``sub_title``
(``"NYK at SAS (Jun 13)"``, abbreviations, present for every event) with the
structured event ticker as fallback, and each derivation is cross-checked
against the other fields so disagreements surface in the report instead of
being silently absorbed.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any

import pandas as pd

from nba_prediction_market.ingestion.nba_games import parse_utc_datetime
from nba_prediction_market.matching.team_names import resolve_team

logger = logging.getLogger(__name__)

#: ``KXNBAGAME-26JUN13NYKSAS`` -> date part + away code + home code.
#: Team codes are exactly three letters: all 2,898 archived markets had a
#: six-character team segment and a three-character suffix, across 31 codes
#: (the 30 NBA clubs plus ``GUA`` for a Guangzhou exhibition opponent). The
#: width is pinned rather than left flexible because ``{2,4}`` would greedily
#: split ``NYKSAS`` into ``NYKS`` + ``AS``.
EVENT_TICKER_RE = re.compile(
    r"^(?P<series>[A-Z0-9]+)-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})"
    r"(?P<away>[A-Z]{3})(?P<home>[A-Z]{3})$"
)
#: ``KXNBAGAME-26JUN13NYKSAS-SAS`` -> event ticker + the team this market is on.
MARKET_TICKER_RE = re.compile(
    r"^(?P<event>[A-Z0-9]+-\d{2}[A-Z]{3}\d{2}[A-Z]{6})-(?P<team>[A-Z]{3})$"
)
#: Both observed rules phrasings: "scheduled for" and "originally scheduled for".
RULES_DATE_RE = re.compile(r"(?:originally )?scheduled for ([A-Z][a-z]{2} \d{1,2}, \d{4})")
TITLE_AT_RE = re.compile(r"^(?:Game \d+: )?(?P<away>.+?) at (?P<home>.+?) Winner\?$")
TITLE_VS_RE = re.compile(r"^(?:Game \d+: )?(?P<first>.+?) vs (?P<second>.+?) Winner\?$")
#: ``"NYK at SAS (Jun 13)"`` -- abbreviations, and no year.
EVENT_SUBTITLE_RE = re.compile(r"^(?P<away>[A-Z]{2,4}) at (?P<home>[A-Z]{2,4})\b")

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

KALSHI_MARKET_COLUMNS: list[str] = [
    "source",
    "ticker",
    "event_ticker",
    "series_ticker",
    "title",
    "event_sub_title",
    "yes_sub_title",
    "no_sub_title",
    "market_team_code",
    "market_team_agrees_with_yes_sub_title",
    "market_team_is_home",
    "away_team_code",
    "home_team_code",
    "orientation_source",
    "scheduled_game_date",
    "scheduled_date_source",
    "ticker_date_agrees_with_rules_date",
    "team_codes_agree_with_event_subtitle",
    "is_nba_matchup",
    "occurrence_datetime_utc",
    "open_time_utc",
    "close_time_utc",
    "expected_expiration_time_utc",
    "expiration_time_utc",
    "settlement_ts_utc",
    "status",
    "result",
    "expiration_value",
    "settlement_value_dollars",
    "last_price_dollars",
    "previous_price_dollars",
    "yes_bid_dollars",
    "yes_ask_dollars",
    "no_bid_dollars",
    "no_ask_dollars",
    "volume_fp",
    "volume_24h_fp",
    "open_interest_fp",
    "liquidity_dollars",
    "market_type",
    "strike_type",
    "custom_strike_team_uuid",
    "rules_primary",
    "source_endpoints",
    "matchup_key",
]


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_event_ticker(event_ticker: str | None) -> dict[str, Any]:
    """Split a Kalshi NBA event ticker into date and the two team codes.

    Returns ``{}`` when the ticker does not match the expected structure -- a
    caller must treat that as "unknown", never as a partial guess.
    """
    if not event_ticker:
        return {}
    match = EVENT_TICKER_RE.match(event_ticker.strip().upper())
    if not match:
        return {}
    month = MONTHS.get(match.group("mon"))
    if month is None:
        return {}
    try:
        ticker_date = date(2000 + int(match.group("yy")), month, int(match.group("dd")))
    except ValueError:
        return {}
    return {
        "series_ticker": match.group("series"),
        "ticker_date": ticker_date,
        # Verified away-then-home on all 1,570 markets whose title uses the
        # "A at B" form, with zero counter-examples.
        "away_raw": match.group("away"),
        "home_raw": match.group("home"),
    }


def parse_market_ticker(ticker: str | None) -> dict[str, Any]:
    """Split a market ticker into its event ticker and team suffix."""
    if not ticker:
        return {}
    match = MARKET_TICKER_RE.match(ticker.strip().upper())
    if not match:
        return {}
    return {"event_ticker": match.group("event"), "team_raw": match.group("team")}


def parse_rules_date(rules_primary: str | None) -> date | None:
    """Extract the scheduled game date (with year) from the settlement rules."""
    if not rules_primary:
        return None
    match = RULES_DATE_RE.search(rules_primary)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%b %d, %Y").date()
    except ValueError:
        return None


def parse_event_subtitle(sub_title: str | None) -> tuple[str | None, str | None]:
    """Return ``(away_code, home_code)`` from an event sub_title, or ``(None, None)``."""
    if not sub_title:
        return None, None
    match = EVENT_SUBTITLE_RE.match(sub_title.strip())
    if not match:
        return None, None
    return match.group("away"), match.group("home")


def parse_title_orientation(title: str | None) -> tuple[str | None, str | None]:
    """Return ``(away_city, home_city)`` from an ``"A at B Winner?"`` title.

    ``"A vs B Winner?"`` titles carry no orientation, so they yield ``(None, None)``.
    """
    if not title:
        return None, None
    match = TITLE_AT_RE.match(title.strip())
    if not match:
        return None, None
    return match.group("away"), match.group("home")


def _resolve_code(raw: str | None) -> str | None:
    resolution = resolve_team(raw)
    return resolution.abbreviation if resolution.ok else None


def normalize_market(
    raw: dict[str, Any],
    *,
    events_by_ticker: dict[str, dict[str, Any]] | None = None,
    source_endpoints: list[str] | None = None,
) -> dict[str, Any]:
    """Flatten one raw Kalshi market into the normalized schema."""
    events_by_ticker = events_by_ticker or {}
    ticker = raw.get("ticker")
    event_ticker = raw.get("event_ticker")

    market_parts = parse_market_ticker(ticker)
    event_parts = parse_event_ticker(event_ticker)
    event = events_by_ticker.get(event_ticker or "", {})
    event_sub_title = event.get("sub_title")

    # --- Orientation: event sub_title first (abbreviations, complete coverage),
    # --- then the structured event ticker.
    sub_away, sub_home = parse_event_subtitle(event_sub_title)
    away_code = _resolve_code(sub_away)
    home_code = _resolve_code(sub_home)
    orientation_source = "event_sub_title" if away_code and home_code else None

    ticker_away = _resolve_code(event_parts.get("away_raw"))
    ticker_home = _resolve_code(event_parts.get("home_raw"))
    if orientation_source is None and ticker_away and ticker_home:
        away_code, home_code = ticker_away, ticker_home
        orientation_source = "event_ticker"

    # Only a real cross-check when the two derivations are independent. If the
    # orientation itself came from the ticker, comparing it to the ticker would
    # trivially report agreement and hide the absence of a second opinion.
    codes_agree: bool | None = None
    if orientation_source == "event_sub_title" and ticker_away and ticker_home:
        codes_agree = (away_code, home_code) == (ticker_away, ticker_home)
        if not codes_agree:
            logger.warning(
                "Kalshi %s: event sub_title says %s at %s but ticker says %s at %s",
                ticker, away_code, home_code, ticker_away, ticker_home,
            )

    # --- Scheduled date: settlement rules carry an explicit year; the ticker
    # --- date is the cross-check.
    rules_date = parse_rules_date(raw.get("rules_primary"))
    ticker_date = event_parts.get("ticker_date")
    if rules_date is not None:
        scheduled_date, scheduled_date_source = rules_date, "rules_primary"
    else:
        scheduled_date, scheduled_date_source = ticker_date, "event_ticker" if ticker_date else None
    date_agrees = None if (rules_date is None or ticker_date is None) else rules_date == ticker_date

    # --- Which team does this market's YES side pay on?
    market_team_code = _resolve_code(market_parts.get("team_raw"))
    # Cross-check only: yes_sub_title is a city label, so it is unresolvable for
    # the two Los Angeles clubs and stays None there rather than guessing.
    subtitle_code = _resolve_code(raw.get("yes_sub_title"))
    team_matches_subtitle: bool | None = None
    if market_team_code and subtitle_code:
        team_matches_subtitle = market_team_code == subtitle_code

    market_team_is_home: bool | None = None
    if market_team_code and home_code and away_code:
        if market_team_code == home_code:
            market_team_is_home = True
        elif market_team_code == away_code:
            market_team_is_home = False

    is_nba_matchup = bool(
        away_code and home_code and market_team_code and away_code != home_code
    )
    matchup_key: str | None = None
    if is_nba_matchup and scheduled_date and away_code and home_code:
        matchup_key = f"{scheduled_date.isoformat()}|{'|'.join(sorted((home_code, away_code)))}"

    custom_strike = raw.get("custom_strike") or {}

    return {
        "source": "kalshi",
        "ticker": ticker,
        "event_ticker": event_ticker,
        "series_ticker": event.get("series_ticker") or event_parts.get("series_ticker"),
        "title": raw.get("title"),
        "event_sub_title": event_sub_title,
        "yes_sub_title": raw.get("yes_sub_title"),
        "no_sub_title": raw.get("no_sub_title"),
        "market_team_code": market_team_code,
        "market_team_agrees_with_yes_sub_title": team_matches_subtitle,
        "market_team_is_home": market_team_is_home,
        "away_team_code": away_code,
        "home_team_code": home_code,
        "orientation_source": orientation_source,
        "scheduled_game_date": scheduled_date,
        "scheduled_date_source": scheduled_date_source,
        "ticker_date_agrees_with_rules_date": date_agrees,
        "team_codes_agree_with_event_subtitle": codes_agree,
        "is_nba_matchup": is_nba_matchup,
        "occurrence_datetime_utc": parse_utc_datetime(raw.get("occurrence_datetime")),
        "open_time_utc": parse_utc_datetime(raw.get("open_time")),
        "close_time_utc": parse_utc_datetime(raw.get("close_time")),
        "expected_expiration_time_utc": parse_utc_datetime(raw.get("expected_expiration_time")),
        "expiration_time_utc": parse_utc_datetime(raw.get("expiration_time")),
        "settlement_ts_utc": parse_utc_datetime(raw.get("settlement_ts")),
        "status": raw.get("status"),
        "result": raw.get("result"),
        "expiration_value": raw.get("expiration_value"),
        "settlement_value_dollars": _to_float(raw.get("settlement_value_dollars")),
        "last_price_dollars": _to_float(raw.get("last_price_dollars")),
        "previous_price_dollars": _to_float(raw.get("previous_price_dollars")),
        "yes_bid_dollars": _to_float(raw.get("yes_bid_dollars")),
        "yes_ask_dollars": _to_float(raw.get("yes_ask_dollars")),
        "no_bid_dollars": _to_float(raw.get("no_bid_dollars")),
        "no_ask_dollars": _to_float(raw.get("no_ask_dollars")),
        "volume_fp": _to_float(raw.get("volume_fp")),
        "volume_24h_fp": _to_float(raw.get("volume_24h_fp")),
        "open_interest_fp": _to_float(raw.get("open_interest_fp")),
        "liquidity_dollars": _to_float(raw.get("liquidity_dollars")),
        "market_type": raw.get("market_type"),
        "strike_type": raw.get("strike_type"),
        "custom_strike_team_uuid": custom_strike.get("basketball_team"),
        "rules_primary": raw.get("rules_primary"),
        "source_endpoints": ",".join(sorted(source_endpoints or [])),
        "matchup_key": matchup_key,
    }


def build_markets_frame(
    raw_markets: list[dict[str, Any]],
    *,
    events: list[dict[str, Any]] | None = None,
    sources_by_ticker: dict[str, list[str]] | None = None,
    season_window: tuple[date, date] | None = None,
) -> pd.DataFrame:
    """Normalize raw markets, de-duplicate by ticker, and optionally scope to a season.

    ``season_window`` filters on the derived ``scheduled_game_date``. Markets
    whose date could not be derived are always kept, so nothing disappears
    without appearing in the report.
    """
    events_by_ticker = {
        e["event_ticker"]: e for e in (events or []) if e.get("event_ticker")
    }
    sources_by_ticker = sources_by_ticker or {}

    normalized = [
        normalize_market(
            market,
            events_by_ticker=events_by_ticker,
            source_endpoints=sources_by_ticker.get(market.get("ticker", ""), []),
        )
        for market in raw_markets
    ]

    missing_ticker = [m for m in normalized if not m["ticker"]]
    if missing_ticker:
        raise ValueError(f"{len(missing_ticker)} Kalshi markets came back without a ticker")

    frame = pd.DataFrame(normalized, columns=KALSHI_MARKET_COLUMNS)
    before = len(frame)
    frame = frame.drop_duplicates(subset=["ticker"], keep="first")
    if len(frame) != before:
        logger.warning("Dropped %d duplicate Kalshi market tickers", before - len(frame))

    if season_window is not None:
        start, end = season_window
        dates = frame["scheduled_game_date"]
        keep = dates.isna() | ((dates >= start) & (dates <= end))
        dropped = int((~keep).sum())
        if dropped:
            logger.info(
                "Filtered out %d Kalshi markets outside %s..%s (other seasons)",
                dropped, start, end,
            )
        frame = frame[keep]

    return frame.sort_values(
        ["scheduled_game_date", "event_ticker", "ticker"], kind="stable"
    ).reset_index(drop=True)
