"""Runtime configuration and season-window conventions.

Every timestamp handled by this package is UTC. Dates that represent an NBA
"game date" (the local calendar day the league schedules the game on) are kept
as naive ``datetime.date`` values and never shifted into UTC, because both
sources label games by that local date -- see :func:`season_window`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

BALLDONTLIE_BASE_URL: Final = "https://api.balldontlie.io/v1"
KALSHI_BASE_URL: Final = "https://external-api.kalshi.com/trade-api/v2"
KALSHI_NBA_SERIES_TICKER: Final = "KXNBAGAME"

#: BALLDONTLIE's free tier allows 5 requests/minute. 12.5s spacing keeps us just
#: under that ceiling; 429s are still retried with the server's ``Retry-After``.
DEFAULT_BALLDONTLIE_MIN_INTERVAL: Final = 12.5
DEFAULT_KALSHI_MIN_INTERVAL: Final = 0.2

#: Page sizes verified against the live APIs (see README "Verified API behaviour").
BALLDONTLIE_PER_PAGE: Final = 100
KALSHI_MARKETS_LIMIT: Final = 1000
KALSHI_EVENTS_LIMIT: Final = 200

# --- Phase 2: pregame quote extraction ------------------------------------

#: Minutes before scheduled tipoff that the quote snapshot is anchored to.
DEFAULT_MINUTES_BEFORE_TIP: Final = 30
#: How far back the candlestick window reaches from the prediction timestamp.
DEFAULT_QUOTE_LOOKBACK_MINUTES: Final = 60
#: A selected quote older than this is preserved but flagged unusable.
DEFAULT_MAX_QUOTE_AGE_MINUTES: Final = 10
#: Candle granularity. Kalshi rejects anything outside its allowed set with a 400.
DEFAULT_CANDLE_PERIOD_INTERVAL: Final = 1
#: Values the candlestick endpoints accept for ``period_interval`` (minutes).
KALSHI_CANDLE_PERIOD_INTERVALS: Final = frozenset({1, 60, 1440})


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def season_label(season: int) -> str:
    """Return the human label for a BALLDONTLIE season year (2025 -> ``2025-26``)."""
    return f"{season}-{(season + 1) % 100:02d}"


def season_slug(season: int) -> str:
    """Return the filename-safe season label (2025 -> ``2025_26``)."""
    return f"{season}_{(season + 1) % 100:02d}"


def season_window(season: int) -> tuple[date, date]:
    """Return the inclusive game-date window used to scope a season.

    ASSUMPTION (tested in ``tests/unit/test_config.py``): an NBA season labelled
    ``season`` runs entirely within 1 July of ``season`` through 30 June of
    ``season + 1``. The 2025-26 season ran 2025-10-21 .. 2026-06-13, and the
    prior season's Finals ended 2025-06-22, so this window separates the two
    cleanly. Kalshi's ``KXNBAGAME`` archive is a single undated stream covering
    multiple seasons, so *some* explicit window is unavoidable; this one is
    stated here rather than buried in the matcher.
    """
    return date(season, 7, 1), date(season + 1, 6, 30)


@dataclass(frozen=True)
class Paths:
    """Filesystem layout for raw, processed, and report artefacts."""

    root: Path

    @property
    def raw_nba(self) -> Path:
        return self.root / "raw" / "nba"

    @property
    def raw_kalshi(self) -> Path:
        return self.root / "raw" / "kalshi"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> None:
        for path in (self.raw_nba, self.raw_kalshi, self.processed, self.reports):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    """Resolved settings for a pipeline run."""

    paths: Paths
    balldontlie_api_key: str | None = None
    balldontlie_base_url: str = BALLDONTLIE_BASE_URL
    kalshi_base_url: str = KALSHI_BASE_URL
    balldontlie_min_interval: float = DEFAULT_BALLDONTLIE_MIN_INTERVAL
    kalshi_min_interval: float = DEFAULT_KALSHI_MIN_INTERVAL
    request_timeout: float = 60.0
    max_retries: int = 5
    extra: dict[str, str] = field(default_factory=dict)

    def require_balldontlie_key(self) -> str:
        if not self.balldontlie_api_key:
            raise ConfigError(
                "BALLDONTLIE_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or export BALLDONTLIE_API_KEY in the environment."
            )
        return self.balldontlie_api_key


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def load_settings(
    data_dir: str | Path | None = None,
    *,
    env_file: str | Path | None = None,
    load_env: bool = True,
) -> Settings:
    """Build :class:`Settings` from the environment and an optional ``.env`` file."""
    if load_env:
        load_dotenv(dotenv_path=env_file, override=False)

    resolved_dir = Path(data_dir or os.environ.get("NBA_PM_DATA_DIR") or "data")
    return Settings(
        paths=Paths(resolved_dir.resolve()),
        balldontlie_api_key=(os.environ.get("BALLDONTLIE_API_KEY") or "").strip() or None,
        balldontlie_base_url=os.environ.get("BALLDONTLIE_BASE_URL", BALLDONTLIE_BASE_URL),
        kalshi_base_url=os.environ.get("KALSHI_BASE_URL", KALSHI_BASE_URL),
        balldontlie_min_interval=_float_env(
            "BALLDONTLIE_MIN_INTERVAL_SECONDS", DEFAULT_BALLDONTLIE_MIN_INTERVAL
        ),
        kalshi_min_interval=_float_env("KALSHI_MIN_INTERVAL_SECONDS", DEFAULT_KALSHI_MIN_INTERVAL),
    )
