"""Season conventions, settings resolution, and the raw-payload store."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from nba_prediction_market.config import (
    ConfigError,
    Paths,
    Settings,
    load_settings,
    season_label,
    season_slug,
    season_window,
)
from nba_prediction_market.ingestion.raw_store import RawStore, utc_now, utc_stamp


@pytest.mark.parametrize(
    ("season", "label", "slug"),
    [(2025, "2025-26", "2025_26"), (1999, "1999-00", "1999_00"), (2009, "2009-10", "2009_10")],
)
def test_season_labels_roll_the_century_over(season: int, label: str, slug: str) -> None:
    assert season_label(season) == label
    assert season_slug(season) == slug


def test_season_window_brackets_the_2025_26_campaign() -> None:
    """The documented assumption: a season lives inside 1 Jul .. 30 Jun."""
    start, end = season_window(2025)
    assert (start, end) == (date(2025, 7, 1), date(2026, 6, 30))

    # The 2025-26 season actually ran 2025-10-21 .. 2026-06-13.
    assert start <= date(2025, 10, 21) <= end
    assert start <= date(2026, 6, 13) <= end
    # ...and the previous season's Finals (2025-06-22) fall outside it.
    assert not (start <= date(2025, 6, 22) <= end)


def test_season_windows_do_not_overlap() -> None:
    assert season_window(2024)[1] < season_window(2025)[0]


def test_missing_api_key_raises_an_actionable_error() -> None:
    settings = Settings(paths=Paths(Path("/tmp/x")), balldontlie_api_key=None)
    with pytest.raises(ConfigError, match="BALLDONTLIE_API_KEY is not set"):
        settings.require_balldontlie_key()
    with_key = Settings(paths=Paths(Path("/tmp/x")), balldontlie_api_key="k")
    assert with_key.require_balldontlie_key() == "k"


def test_load_settings_reads_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BALLDONTLIE_API_KEY", "  env-key  ")
    monkeypatch.setenv("BALLDONTLIE_MIN_INTERVAL_SECONDS", "0.5")
    settings = load_settings(tmp_path, load_env=False)

    assert settings.balldontlie_api_key == "env-key"
    assert settings.balldontlie_min_interval == 0.5
    assert settings.paths.root == tmp_path.resolve()


def test_blank_api_key_is_treated_as_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BALLDONTLIE_API_KEY", "   ")
    assert load_settings(tmp_path, load_env=False).balldontlie_api_key is None


def test_non_numeric_interval_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("KALSHI_MIN_INTERVAL_SECONDS", "fast")
    with pytest.raises(ConfigError, match="must be a number"):
        load_settings(tmp_path, load_env=False)


def test_paths_ensure_creates_the_full_layout(tmp_path: Path) -> None:
    paths = Paths(tmp_path / "data")
    paths.ensure()
    for directory in (paths.raw_nba, paths.raw_kalshi, paths.processed, paths.reports):
        assert directory.is_dir()
    paths.ensure()  # idempotent


# --- raw store -------------------------------------------------------------


def test_raw_store_preserves_pages_verbatim(tmp_path: Path) -> None:
    store = RawStore(tmp_path, run_stamp="20260819T000000Z")
    pages = [{"data": [{"id": 1}], "meta": {"next_cursor": 1}}, {"data": [{"id": 2}]}]

    snapshot = store.write("games", pages, record_count=2, params={"seasons[]": 2025})

    assert snapshot.path.name == "games_20260819T000000Z.json"
    document = json.loads(snapshot.path.read_text())
    assert document["pages"] == pages
    assert document["record_count"] == 2
    assert document["page_count"] == 2
    assert document["params"] == {"seasons[]": 2025}
    assert document["fetched_at_utc"].endswith("+00:00")


def test_raw_store_roundtrips_records(tmp_path: Path) -> None:
    store = RawStore(tmp_path, run_stamp="20260819T000000Z")
    pages = [{"markets": [{"ticker": "A"}]}, {"markets": [{"ticker": "B"}]}]
    snapshot = store.write("markets", pages, record_count=2)

    document = RawStore.load(snapshot.path)
    assert [m["ticker"] for m in RawStore.records(document, "markets")] == ["A", "B"]


def test_raw_store_latest_picks_the_newest_stamp(tmp_path: Path) -> None:
    RawStore(tmp_path, run_stamp="20260101T000000Z").write("games", [], record_count=0)
    newest = RawStore(tmp_path, run_stamp="20260819T000000Z").write("games", [], record_count=0)

    assert RawStore(tmp_path).latest("games") == newest.path
    assert RawStore(tmp_path).latest("nothing") is None


def test_timestamps_are_utc() -> None:
    now = utc_now()
    assert now.tzinfo is not None
    assert now.utcoffset().total_seconds() == 0
    assert utc_stamp(now).endswith("Z")
    assert len(utc_stamp(now)) == 16
