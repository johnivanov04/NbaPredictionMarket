"""Shared HTTP plumbing: timeouts, rate limiting, retries, and error mapping."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import httpx
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt

logger = logging.getLogger(__name__)

#: Server-side failures worth retrying. 429 is handled separately so we can
#: honour the ``Retry-After`` header instead of guessing a backoff.
RETRYABLE_STATUS = frozenset({500, 502, 503, 504})
MAX_RETRY_AFTER_SECONDS = 120.0


class ApiError(RuntimeError):
    """A non-retryable API failure, carrying enough context to debug it."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class TransientApiError(ApiError):
    """A retryable API failure (5xx, network error)."""


class RateLimitedError(TransientApiError):
    """HTTP 429. ``retry_after`` is the server's requested cool-off, if supplied."""

    def __init__(self, message: str, *, retry_after: float | None = None, body: str | None = None):
        super().__init__(message, status_code=429, body=body)
        self.retry_after = retry_after


class RateLimiter:
    """Blocking minimum-interval throttle.

    Deliberately simple: upstream limits here are per-minute request counts, so
    spacing requests evenly is both sufficient and easy to reason about. Uses a
    monotonic clock so wall-clock adjustments cannot stall a run.
    """

    def __init__(self, min_interval: float, *, sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.min_interval = max(0.0, min_interval)
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_call: float | None = None

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        now = self._monotonic()
        if self._last_call is not None:
            wait = self.min_interval - (now - self._last_call)
            if wait > 0:
                self._sleep(wait)
                now = self._monotonic()
        self._last_call = now


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _wait_strategy(retry_state: RetryCallState) -> float:
    """Honour ``Retry-After`` when present, else exponential backoff (1,2,4,8...s)."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, RateLimitedError) and exc.retry_after is not None:
        return min(exc.retry_after + 1.0, MAX_RETRY_AFTER_SECONDS)
    return min(2.0 ** max(0, retry_state.attempt_number - 1), 30.0)


class BaseApiClient:
    """Minimal JSON-over-HTTP client with retries and a request throttle."""

    def __init__(
        self,
        base_url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 60.0,
        min_interval: float = 0.0,
        max_retries: int = 5,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(min_interval)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout, headers=dict(headers or {}), follow_redirects=True
        )
        if client is not None and headers:
            self._client.headers.update(dict(headers))

    def __enter__(self) -> BaseApiClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _request_once(self, path: str, params: Mapping[str, Any] | None) -> dict[str, Any]:
        self.rate_limiter.acquire()
        url = f"{self.base_url}{path}"
        try:
            response = self._client.get(url, params=dict(params or {}))
        except httpx.TimeoutException as exc:
            raise TransientApiError(f"Timeout requesting {url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TransientApiError(f"Network error requesting {url}: {exc}") from exc

        if response.status_code == 429:
            raise RateLimitedError(
                f"Rate limited by {url}",
                retry_after=_parse_retry_after(response.headers.get("retry-after")),
                body=response.text[:500],
            )
        if response.status_code in RETRYABLE_STATUS:
            raise TransientApiError(
                f"Server error {response.status_code} from {url}",
                status_code=response.status_code,
                body=response.text[:500],
            )
        if response.status_code >= 400:
            raise ApiError(
                f"Request to {url} failed with HTTP {response.status_code}: "
                f"{response.text[:500]}",
                status_code=response.status_code,
                body=response.text[:500],
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiError(f"Non-JSON response from {url}: {response.text[:200]!r}") from exc
        if not isinstance(payload, dict):
            raise ApiError(f"Expected a JSON object from {url}, got {type(payload).__name__}")
        return payload

    def get_json(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """GET a JSON object, retrying transient failures."""
        retryer = Retrying(
            retry=retry_if_exception_type(TransientApiError),
            wait=_wait_strategy,
            stop=stop_after_attempt(self.max_retries),
            reraise=True,
        )
        for attempt in retryer:
            with attempt:
                return self._request_once(path, params)
        raise AssertionError("unreachable: Retrying always returns or raises")

    def paginate(
        self,
        path: str,
        *,
        params: Mapping[str, Any],
        items_key: str,
        next_cursor: Callable[[dict[str, Any]], Any],
        cursor_param: str = "cursor",
        max_pages: int = 1000,
        on_page: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield each page's items, following cursors until the source is exhausted.

        Stops on a falsy cursor, an empty page, or a repeated cursor (which would
        otherwise loop forever). ``max_pages`` is a hard safety valve that raises
        rather than silently truncating -- a truncated season must never look
        like a complete one.
        """
        cursor: Any = None
        seen_cursors: set[str] = set()
        for page_number in range(1, max_pages + 1):
            page_params = dict(params)
            if cursor:
                page_params[cursor_param] = cursor
            payload = self._request_once_with_retries(path, page_params)

            items = payload.get(items_key)
            if items is None:
                raise ApiError(
                    f"Response from {path} has no {items_key!r} key; got keys "
                    f"{sorted(payload)}"
                )
            if not isinstance(items, list):
                raise ApiError(f"Expected {items_key!r} to be a list from {path}")

            if on_page is not None:
                on_page(page_number, payload)
            if items:
                yield items

            cursor = next_cursor(payload)
            if not cursor or not items:
                return
            token = str(cursor)
            if token in seen_cursors:
                logger.warning("Repeated cursor %s from %s; stopping pagination", token, path)
                return
            seen_cursors.add(token)

        raise ApiError(
            f"Pagination of {path} exceeded max_pages={max_pages}; refusing to "
            "return a partial dataset."
        )

    def _request_once_with_retries(
        self, path: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self.get_json(path, params)
