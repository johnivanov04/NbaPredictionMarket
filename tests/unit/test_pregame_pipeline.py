"""Phase 2 pipeline: side assignment, eligibility, row shape, and the report.

Covers requirements 16-20 (home/away assignment, contradictions, one row per
game, postseason exclusion). Network access is served by MockTransport.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from nba_prediction_market.clients.kalshi import KalshiClient
from nba_prediction_market.config import Paths, Settings
from nba_prediction_market.ingestion.candlesticks import (
    ISSUE_FETCH_FAILED,
    ISSUE_NO_MARKET_TICKER,
    ISSUE_STALE,
    ISSUE_TEAM_MISMATCH,
)
from nba_prediction_market.ingestion.game_phase import (
    PHASE_NBA_CUP_CHAMPIONSHIP,
    PHASE_PLAY_IN,
    PHASE_PLAYOFFS,
    PHASE_REGULAR_SEASON,
    PHASE_UNCLASSIFIED,
)
from nba_prediction_market.pipelines.build_pregame_quotes import (
    PREGAME_COLUMNS,
    build_arg_parser,
    main,
    run_pipeline,
    select_eligible_games,
    verify_side_assignment,
)

TIP = datetime(2026, 1, 15, 23, 30, tzinfo=UTC)
_UNSET = object()
PRED_TS = int(TIP.timestamp()) - 30 * 60


def match_row(
    game_id: int,
    *,
    home: str = "OKC",
    away: str = "HOU",
    postseason: bool = False,
    phase: str = PHASE_REGULAR_SEASON,
    status: str = "matched",
    season: int = 2025,
    tipoff: datetime | None = TIP,
    event: str = "KXNBAGAME-26JAN15HOUOKC",
    home_ticker: Any = _UNSET,
    away_ticker: Any = _UNSET,
    game_date: str = "2026-01-15",
) -> dict[str, Any]:
    return {
        "match_status": status,
        "match_tier": "exact_date_and_teams",
        "nba_game_id": game_id,
        "nba_game_date": game_date,
        "nba_season": season,
        "nba_postseason": postseason,
        "nba_game_phase": phase,
        "nba_status": "Final",
        "nba_is_final": True,
        "nba_tipoff_utc": tipoff,
        "nba_home_team_code": home,
        "nba_away_team_code": away,
        "nba_home_score": 110,
        "nba_away_score": 100,
        "nba_home_win": True,
        "kalshi_event_ticker": event,
        "kalshi_home_market_ticker": f"{event}-{home}" if home_ticker is _UNSET else home_ticker,
        "kalshi_away_market_ticker": f"{event}-{away}" if away_ticker is _UNSET else away_ticker,
    }


def market_row(ticker: str, team: str) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "market_team_code": team,
        "settlement_ts_utc": datetime(2026, 1, 16, 3, 0, tzinfo=UTC),
        "close_time_utc": datetime(2026, 1, 16, 2, 30, tzinfo=UTC),
    }


def candle_payload(*offsets_and_prices) -> dict[str, Any]:
    """Build a payload; each entry is ``(seconds_before_pred, bid, ask)``."""
    return {
        "candlesticks": [
            {
                "end_period_ts": PRED_TS - offset,
                "open_interest": "1000.00",
                "price": {"close": "0.6000", "previous": "0.5900"},
                "volume": "25.00",
                "yes_bid": {"close": f"{bid:.4f}"} if bid is not None else {},
                "yes_ask": {"close": f"{ask:.4f}"} if ask is not None else {},
            }
            for offset, bid, ask in offsets_and_prices
        ]
    }


# --- 16, 17, 18: home/away assignment --------------------------------------

MARKETS = {
    "KXNBAGAME-26JAN15HOUOKC-OKC": market_row("KXNBAGAME-26JAN15HOUOKC-OKC", "OKC"),
    "KXNBAGAME-26JAN15HOUOKC-HOU": market_row("KXNBAGAME-26JAN15HOUOKC-HOU", "HOU"),
}


def test_home_and_away_markets_are_assigned_from_market_team_not_order() -> None:
    home_issue, away_issue = verify_side_assignment(match_row(1), MARKETS)
    assert home_issue is None
    assert away_issue is None


def test_assignment_survives_the_tickers_being_listed_in_either_order() -> None:
    """Orientation comes from market_team_code, so array order is irrelevant."""
    swapped = match_row(1)
    swapped["kalshi_home_market_ticker"], swapped["kalshi_away_market_ticker"] = (
        swapped["kalshi_away_market_ticker"],
        swapped["kalshi_home_market_ticker"],
    )
    home_issue, away_issue = verify_side_assignment(swapped, MARKETS)
    # Now home_ticker points at HOU's market while the NBA home team is OKC.
    assert home_issue == ISSUE_TEAM_MISMATCH
    assert away_issue == ISSUE_TEAM_MISMATCH


def test_market_naming_a_third_team_is_contradictory() -> None:
    markets = {**MARKETS}
    markets["KXNBAGAME-26JAN15HOUOKC-OKC"] = market_row("KXNBAGAME-26JAN15HOUOKC-OKC", "LAL")
    home_issue, away_issue = verify_side_assignment(match_row(1), markets)
    assert home_issue == ISSUE_TEAM_MISMATCH
    assert away_issue is None


def test_both_sides_pointing_at_one_market_is_contradictory() -> None:
    same = match_row(1, home_ticker="KXNBAGAME-26JAN15HOUOKC-OKC",
                     away_ticker="KXNBAGAME-26JAN15HOUOKC-OKC")
    assert verify_side_assignment(same, MARKETS) == (ISSUE_TEAM_MISMATCH, ISSUE_TEAM_MISMATCH)


@pytest.mark.parametrize("missing", [None, "", float("nan")])
def test_missing_market_ticker_is_flagged(missing) -> None:
    home_issue, away_issue = verify_side_assignment(match_row(1, home_ticker=missing), MARKETS)
    assert home_issue == ISSUE_NO_MARKET_TICKER
    assert away_issue is None


def test_ticker_absent_from_the_market_table_is_flagged() -> None:
    home_issue, _ = verify_side_assignment(match_row(1, home_ticker="UNKNOWN-TICKER"), MARKETS)
    assert home_issue == ISSUE_TEAM_MISMATCH


# --- 20: eligibility -------------------------------------------------------


def test_postseason_games_are_excluded_from_the_primary_dataset() -> None:
    matches = pd.DataFrame([
        match_row(1),
        match_row(2, postseason=True, phase=PHASE_PLAYOFFS),
        match_row(3),
    ])
    eligible = select_eligible_games(matches, 2025)
    assert list(eligible["nba_game_id"]) == [1, 3]
    assert not eligible["nba_postseason"].any()


@pytest.mark.parametrize(
    "phase", [PHASE_PLAY_IN, PHASE_PLAYOFFS, PHASE_NBA_CUP_CHAMPIONSHIP, PHASE_UNCLASSIFIED]
)
def test_only_regular_season_games_are_eligible(phase: str) -> None:
    """postseason=False is not sufficient -- the phase label decides."""
    matches = pd.DataFrame([match_row(1), match_row(2, postseason=False, phase=phase)])
    eligible = select_eligible_games(matches, 2025)

    assert list(eligible["nba_game_id"]) == [1]
    assert set(eligible["nba_game_phase"]) == {PHASE_REGULAR_SEASON}


def test_play_in_games_are_excluded_even_though_postseason_is_false() -> None:
    """The exact bug: six play-in games entered the 2025-26 modelling set."""
    matches = pd.DataFrame([
        match_row(i, game_date="2026-04-10") for i in range(1, 4)
    ] + [
        match_row(100 + i, game_date="2026-04-14", postseason=False, phase=PHASE_PLAY_IN)
        for i in range(6)
    ])
    eligible = select_eligible_games(matches, 2025)

    assert len(eligible) == 3
    assert not eligible["nba_game_id"].isin(range(100, 106)).any()
    # ...and the play-in rows are still present in the source table.
    assert int((matches["nba_game_phase"] == PHASE_PLAY_IN).sum()) == 6


def test_nba_cup_final_is_excluded_even_though_postseason_is_false() -> None:
    matches = pd.DataFrame([
        match_row(1),
        match_row(2, game_date="2025-12-16", phase=PHASE_NBA_CUP_CHAMPIONSHIP),
    ])
    assert list(select_eligible_games(matches, 2025)["nba_game_id"]) == [1]


def test_a_match_table_without_the_phase_column_is_rejected() -> None:
    """Guards against silently reusing a pre-classification Phase 1 output."""
    from nba_prediction_market.config import ConfigError

    stale = pd.DataFrame([match_row(1)]).drop(columns=["nba_game_phase"])
    with pytest.raises(ConfigError, match="predates game-phase classification"):
        select_eligible_games(stale, 2025)


@pytest.mark.parametrize("status", ["unmatched_nba", "unmatched_kalshi", "ambiguous"])
def test_unmatched_and_ambiguous_records_are_excluded(status: str) -> None:
    matches = pd.DataFrame([match_row(1), match_row(2, status=status)])
    assert list(select_eligible_games(matches, 2025)["nba_game_id"]) == [1]


def test_other_seasons_are_excluded() -> None:
    matches = pd.DataFrame([match_row(1, season=2025), match_row(2, season=2024)])
    assert list(select_eligible_games(matches, 2025)["nba_game_id"]) == [1]


def test_eligible_games_are_deterministically_ordered() -> None:
    matches = pd.DataFrame([
        match_row(3, game_date="2026-01-16"),
        match_row(1, game_date="2026-01-15"),
        match_row(2, game_date="2026-01-15"),
    ])
    assert list(select_eligible_games(matches, 2025)["nba_game_id"]) == [1, 2, 3]


# --- end-to-end ------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(paths=Paths(tmp_path / "data"))


def write_phase1(settings: Settings, matches: list[dict], markets: list[dict]) -> None:
    settings.paths.ensure()
    pd.DataFrame(matches).to_parquet(
        settings.paths.processed / "nba_kalshi_matches_2025_26.parquet", index=False
    )
    pd.DataFrame(markets).to_parquet(
        settings.paths.processed / "kalshi_nba_markets_2025_26.parquet", index=False
    )


def kalshi_stub(responses: dict[str, dict] | None = None, default: dict | None = None):
    """MockTransport serving cutoff plus per-ticker candle payloads."""
    responses = responses or {}
    default = default if default is not None else candle_payload((0, 0.60, 0.61))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/historical/cutoff"):
            return httpx.Response(200, json={"market_settled_ts": "2026-06-20T00:00:00Z"})
        ticker = request.url.path.split("/")[-2]
        payload = responses.get(ticker, default)
        if payload == "404":
            return httpx.Response(404, json={"error": "not found"})
        if payload == "500":
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=payload)

    return KalshiClient(
        base_url="https://api.test", min_interval=0.0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)


def test_one_row_per_eligible_game(settings: Settings) -> None:
    """Requirement 19: the primary table is per game, not per market."""
    matches = [match_row(i, event=f"KXNBAGAME-26JAN15HOUOKC{i}") for i in (1, 2, 3)]
    markets = [
        market_row(f"KXNBAGAME-26JAN15HOUOKC{i}-{team}", team)
        for i in (1, 2, 3) for team in ("OKC", "HOU")
    ]
    write_phase1(settings, matches, markets)

    result = run_pipeline(2025, settings=settings, client=kalshi_stub(), write_csv=False)

    assert len(result.quotes) == 3
    assert result.quotes["nba_game_id"].is_unique
    assert list(result.quotes.columns) == PREGAME_COLUMNS


def test_postseason_games_never_reach_the_output(settings: Settings) -> None:
    write_phase1(
        settings,
        [match_row(1), match_row(2, postseason=True, phase=PHASE_PLAYOFFS)],
        list(MARKETS.values()),
    )
    result = run_pipeline(2025, settings=settings, client=kalshi_stub(), write_csv=False)

    assert list(result.quotes["nba_game_id"]) == [1]
    assert not result.quotes["postseason"].any()


def test_row_carries_the_anchor_and_a_two_sided_quote(settings: Settings) -> None:
    write_phase1(settings, [match_row(1)], list(MARKETS.values()))
    responses = {
        "KXNBAGAME-26JAN15HOUOKC-OKC": candle_payload((0, 0.60, 0.61)),
        "KXNBAGAME-26JAN15HOUOKC-HOU": candle_payload((0, 0.38, 0.39)),
    }
    result = run_pipeline(
        2025, settings=settings, client=kalshi_stub(responses), write_csv=False
    )
    row = result.quotes.iloc[0]

    assert row["prediction_ts_utc"] == pd.Timestamp(TIP) - pd.Timedelta(minutes=30)
    assert row["home_team"] == "OKC" and row["away_team"] == "HOU"
    assert row["home_yes_bid"] == pytest.approx(0.60)
    assert row["home_yes_ask"] == pytest.approx(0.61)
    assert row["home_market_midpoint"] == pytest.approx(0.605)
    assert row["home_spread"] == pytest.approx(0.01)
    assert row["away_market_midpoint"] == pytest.approx(0.385)
    assert bool(row["both_sides_usable"]) is True
    assert row["market_midpoint_sum"] == pytest.approx(0.99)
    assert row["midpoint_sum_deviation_from_one"] == pytest.approx(0.01)
    assert row["home_quote_age_seconds"] == 0.0
    assert row["match_method"] == "exact_date_and_teams"


def test_midpoints_are_never_normalised_to_sum_to_one(settings: Settings) -> None:
    """We want to observe the real deviation, not a rescaled one."""
    write_phase1(settings, [match_row(1)], list(MARKETS.values()))
    responses = {
        "KXNBAGAME-26JAN15HOUOKC-OKC": candle_payload((0, 0.70, 0.71)),
        "KXNBAGAME-26JAN15HOUOKC-HOU": candle_payload((0, 0.40, 0.41)),
    }
    row = run_pipeline(
        2025, settings=settings, client=kalshi_stub(responses), write_csv=False
    ).quotes.iloc[0]

    assert row["market_midpoint_sum"] == pytest.approx(1.11)
    assert row["midpoint_sum_deviation_from_one"] == pytest.approx(0.11)
    assert row["home_market_midpoint"] == pytest.approx(0.705)


def test_no_selected_quote_is_ever_after_the_prediction_ts(settings: Settings) -> None:
    write_phase1(settings, [match_row(1)], list(MARKETS.values()))
    # Newest candle sits AFTER the anchor and must be ignored.
    payload = candle_payload((-60, 0.90, 0.91), (120, 0.60, 0.61))
    result = run_pipeline(
        2025, settings=settings, client=kalshi_stub(default=payload), write_csv=False
    )
    row = result.quotes.iloc[0]

    assert row["home_yes_bid"] == pytest.approx(0.60)   # not 0.90
    assert row["home_quote_age_seconds"] == 120.0
    assert row["home_quote_ts_utc"] <= row["prediction_ts_utc"]
    assert result.report["lookahead_guard"]["selected_quotes_after_prediction_ts"] == 0


def test_stale_quotes_are_preserved_but_unusable(settings: Settings) -> None:
    write_phase1(settings, [match_row(1)], list(MARKETS.values()))
    result = run_pipeline(
        2025, settings=settings, client=kalshi_stub(default=candle_payload((900, 0.6, 0.61))),
        max_quote_age_minutes=10, write_csv=False,
    )
    row = result.quotes.iloc[0]

    assert row["home_quote_issue"] == ISSUE_STALE
    assert bool(row["home_quote_usable"]) is False
    assert bool(row["both_sides_usable"]) is False
    assert row["home_yes_bid"] == pytest.approx(0.60)   # kept for diagnostics
    assert row["home_quote_age_seconds"] == 900.0


def test_fetch_failure_is_recorded_per_side(settings: Settings) -> None:
    write_phase1(settings, [match_row(1)], list(MARKETS.values()))
    result = run_pipeline(
        2025, settings=settings,
        client=kalshi_stub({"KXNBAGAME-26JAN15HOUOKC-OKC": "500"}), write_csv=False,
    )
    row = result.quotes.iloc[0]

    assert row["home_quote_issue"] == ISSUE_FETCH_FAILED
    assert bool(row["home_quote_usable"]) is False
    assert bool(row["away_quote_usable"]) is True     # the other side still works
    assert bool(row["both_sides_usable"]) is False


def test_contradictory_assignment_marks_the_row_unusable(settings: Settings) -> None:
    bad = market_row("KXNBAGAME-26JAN15HOUOKC-OKC", "LAL")
    write_phase1(settings, [match_row(1)], [bad, MARKETS["KXNBAGAME-26JAN15HOUOKC-HOU"]])
    result = run_pipeline(2025, settings=settings, client=kalshi_stub(), write_csv=False)
    row = result.quotes.iloc[0]

    assert row["home_quote_issue"] == ISSUE_TEAM_MISMATCH
    assert bool(row["home_quote_usable"]) is False
    assert bool(row["both_sides_usable"]) is False


def test_missing_tipoff_yields_no_anchor_and_no_quote(settings: Settings) -> None:
    write_phase1(settings, [match_row(1, tipoff=None)], list(MARKETS.values()))
    result = run_pipeline(2025, settings=settings, client=kalshi_stub(), write_csv=False)
    row = result.quotes.iloc[0]

    assert pd.isna(row["prediction_ts_utc"])
    assert bool(row["both_sides_usable"]) is False


def test_rerunning_uses_the_cache_and_reproduces_the_dataset(settings: Settings) -> None:
    write_phase1(settings, [match_row(1)], list(MARKETS.values()))
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path.endswith("/historical/cutoff"):
            return httpx.Response(200, json={"market_settled_ts": "2026-06-20T00:00:00Z"})
        calls += 1
        return httpx.Response(200, json=candle_payload((0, 0.60, 0.61)))

    def client():
        return KalshiClient(base_url="https://api.test", min_interval=0.0,
                            client=httpx.Client(transport=httpx.MockTransport(handler)))

    first = run_pipeline(2025, settings=settings, client=client(), write_csv=False)
    assert calls == 2
    second = run_pipeline(2025, settings=settings, client=client(), write_csv=False)
    assert calls == 2                      # fully served from cache
    assert second.report["cache"]["hits"] == 2
    pd.testing.assert_frame_equal(first.quotes, second.quotes)


def test_outputs_and_report_are_written(settings: Settings) -> None:
    write_phase1(settings, [match_row(1)], list(MARKETS.values()))
    result = run_pipeline(2025, settings=settings, client=kalshi_stub(), write_csv=True)

    parquet = settings.paths.processed / "nba_kalshi_pregame_t30_2025_26.parquet"
    assert parquet.is_file()
    assert (settings.paths.processed / "nba_kalshi_pregame_t30_2025_26.csv").is_file()
    report_path = settings.paths.reports / "pregame_t30_report.json"
    assert report_path.is_file()
    assert all(p.exists() for p in result.written_files)

    reloaded = pd.read_parquet(parquet)
    assert list(reloaded.columns) == PREGAME_COLUMNS


def test_report_covers_every_required_section(settings: Settings) -> None:
    write_phase1(
        settings,
        [match_row(1), match_row(2, event="KXNBAGAME-26JAN15HOUOKC2")],
        [*MARKETS.values(),
         market_row("KXNBAGAME-26JAN15HOUOKC2-OKC", "OKC"),
         market_row("KXNBAGAME-26JAN15HOUOKC2-HOU", "HOU")],
    )
    responses = {"KXNBAGAME-26JAN15HOUOKC2-OKC": candle_payload((900, 0.6, 0.61))}
    result = run_pipeline(
        2025, settings=settings, client=kalshi_stub(responses), write_csv=False
    )
    report = json.loads(
        (settings.paths.reports / "pregame_t30_report.json").read_text()
    )

    coverage = report["coverage"]
    assert coverage["eligible_regular_season_matched_games"] == 2
    assert coverage["both_sides_usable"] == 1
    assert coverage["neither_usable"] + coverage["away_only_usable"] == 1
    assert report["quote_issues"][ISSUE_STALE] == 1
    assert ISSUE_STALE in report["quote_issue_examples"]
    for key in ("quote_age_seconds", "spread_dollars", "market_midpoint_sum",
                "midpoint_sum_deviation_from_one"):
        assert "p50" in report["distributions"][key]
    for threshold in ("gt_0.01", "gt_0.02", "gt_0.05"):
        assert threshold in report["midpoint_sum_deviation_counts"]
    assert report["configuration"]["minutes_before_tip"] == 30
    assert report["lookahead_guard"]["selected_quotes_after_prediction_ts"] == 0
    assert coverage["eligible_regular_season_matched_games"] == len(result.quotes)


def test_missing_phase1_output_is_an_actionable_error(settings: Settings) -> None:
    settings.paths.ensure()
    from nba_prediction_market.config import ConfigError

    with pytest.raises(ConfigError, match="Run the Phase 1 pipeline first"):
        run_pipeline(2025, settings=settings, client=kalshi_stub())


# --- CLI -------------------------------------------------------------------


def test_cli_defaults_match_the_documented_command() -> None:
    args = build_arg_parser().parse_args([])
    assert args.season == 2025
    assert args.minutes_before_tip == 30
    assert args.max_quote_age_minutes == 10
    assert args.lookback_minutes == 60
    assert args.period_interval == 1
    assert args.refresh is False


def test_cli_accepts_the_documented_flags() -> None:
    args = build_arg_parser().parse_args(
        ["--season", "2025", "--minutes-before-tip", "30", "--max-quote-age-minutes", "10"]
    )
    assert (args.season, args.minutes_before_tip, args.max_quote_age_minutes) == (2025, 30, 10.0)


def test_cli_reports_missing_phase1_inputs_without_a_traceback(
    tmp_path: Path, capsys
) -> None:
    exit_code = main(["--data-dir", str(tmp_path / "data"), "--log-level", "ERROR"])
    assert exit_code == 2
    assert "Phase 1" in capsys.readouterr().err


def test_deviation_thresholds_are_not_inflated_by_float_noise(settings: Settings) -> None:
    """A true deviation of exactly 0.01 must not be counted as greater than 0.01."""
    write_phase1(settings, [match_row(1)], list(MARKETS.values()))
    responses = {
        "KXNBAGAME-26JAN15HOUOKC-OKC": candle_payload((0, 0.60, 0.61)),   # mid 0.605
        "KXNBAGAME-26JAN15HOUOKC-HOU": candle_payload((0, 0.38, 0.39)),   # mid 0.385
    }
    result = run_pipeline(
        2025, settings=settings, client=kalshi_stub(responses), write_csv=False
    )
    row = result.quotes.iloc[0]

    assert row["market_midpoint_sum"] == 0.99
    assert row["midpoint_sum_deviation_from_one"] == 0.01
    assert result.report["midpoint_sum_deviation_counts"]["gt_0.01"] == 0
    assert result.report["midpoint_sum_deviation_counts"]["gt_0.02"] == 0
