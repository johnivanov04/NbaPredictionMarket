"""BALLDONTLIE API client (games endpoint only, for Phase 1)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from nba_prediction_market.clients.base import ApiError, BaseApiClient
from nba_prediction_market.config import (
    BALLDONTLIE_BASE_URL,
    BALLDONTLIE_PER_PAGE,
    DEFAULT_BALLDONTLIE_MIN_INTERVAL,
)

logger = logging.getLogger(__name__)


def _next_cursor(payload: dict[str, Any]) -> Any:
    meta = payload.get("meta") or {}
    if not isinstance(meta, dict):
        raise ApiError(f"Expected 'meta' object in BALLDONTLIE response, got {type(meta).__name__}")
    return meta.get("next_cursor")


class BallDontLieClient(BaseApiClient):
    """Cursor-paginated reader for ``GET /v1/games``.

    Auth is a bare API key in the ``Authorization`` header (no ``Bearer`` prefix)
    -- that is what the service accepts.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BALLDONTLIE_BASE_URL,
        timeout: float = 60.0,
        min_interval: float = DEFAULT_BALLDONTLIE_MIN_INTERVAL,
        max_retries: int = 5,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for BallDontLieClient")
        super().__init__(
            base_url,
            headers={"Authorization": api_key, "Accept": "application/json"},
            timeout=timeout,
            min_interval=min_interval,
            max_retries=max_retries,
            client=client,
        )

    def iter_games(
        self,
        season: int,
        *,
        per_page: int = BALLDONTLIE_PER_PAGE,
        postseason: bool | None = None,
        on_page: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield pages of games for ``season``.

        ``postseason`` is left unset by default so regular season *and* playoff
        games arrive in one stream; the flag is preserved per game downstream.
        """
        if not 1 <= per_page <= 100:
            raise ValueError(f"per_page must be within 1..100, got {per_page}")
        params: dict[str, Any] = {"seasons[]": season, "per_page": per_page}
        if postseason is not None:
            params["postseason"] = str(postseason).lower()
        yield from self.paginate(
            "/games",
            params=params,
            items_key="data",
            next_cursor=_next_cursor,
            on_page=on_page,
        )

    def fetch_games(
        self,
        season: int,
        *,
        per_page: int = BALLDONTLIE_PER_PAGE,
        on_page: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Return every game for ``season`` as raw API dicts."""
        games: list[dict[str, Any]] = []
        for page in self.iter_games(season, per_page=per_page, on_page=on_page):
            games.extend(page)
        logger.info("Fetched %d BALLDONTLIE games for season %s", len(games), season)
        return games
