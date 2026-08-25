"""Phase 4A0: does disagreement with Kalshi contain a tradable signal?

Research only. Nothing here places an order, and nothing here is a strategy.

Three commitments shape the whole file, because each is a way this analysis
could flatter itself:

* **Execution is taking, at the displayed ask.** The midpoint is not a price
  anyone can buy at. A resting order is not a fill. Maker economics appear only
  as a clearly labelled hypothetical.
* **Thresholds are declared before results are seen**, and every one is
  reported. Picking the best-performing threshold after the fact and calling it
  the strategy is the single easiest way to manufacture an edge that is not
  there.
* **2025-26 is not a pristine holdout.** It has been inspected throughout model
  development. Kalshi published no NBA regular-season game markets before
  2025-26, so there is no earlier season to develop rules on. Every number here
  is exploratory and the phase says so rather than implying validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.models import metrics
from nba_prediction_market.models.kalshi_fees import (
    FEE_SCHEDULES,
    FeeScheduleUnavailableError,
    maker_fee,
    schedule_for,
    taker_fee_per_contract,
)

logger = logging.getLogger(__name__)

HOLDOUT_SEASON = 2025

#: Declared before any result was inspected. Every one is reported.
EDGE_THRESHOLDS: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
#: Order sizes for the fee-rounding view. Not a bankroll recommendation.
CONTRACT_SIZES: tuple[int, ...] = (1, 10, 100)
#: Rounding-stable size used for the headline view, justified by the fee audit.
PRIMARY_SIZE: int = 100

#: Disagreement bins, declared in advance and not revised after seeing results.
DISAGREEMENT_BINS: tuple[tuple[float, float, str], ...] = (
    (-1.00, -0.10, "< -10%"),
    (-0.10, -0.05, "-10 to -5%"),
    (-0.05, -0.03, "-5 to -3%"),
    (-0.03, -0.01, "-3 to -1%"),
    (-0.01, 0.01, "-1 to +1%"),
    (0.01, 0.03, "+1 to +3%"),
    (0.03, 0.05, "+3 to +5%"),
    (0.05, 0.10, "+5 to +10%"),
    (0.10, 1.00, "> +10%"),
)

BOOTSTRAP_RESAMPLES: int = 10_000
BOOTSTRAP_SEED: int = 20260825


@dataclass(frozen=True)
class TradeSide:
    """Which side a trade would take, and at what quoted price."""

    name: str
    price_column: str
    win_column: str


SIDES: tuple[TradeSide, ...] = (
    TradeSide("HOME", "home_yes_ask", "home_win"),
    TradeSide("AWAY", "away_yes_ask", "away_win"),
)


def load_frame(settings: Settings) -> pd.DataFrame:
    """Frozen model probabilities joined to the T-30 executable market state."""
    processed = settings.paths.processed
    predictions = processed / "nba_predictions_3a3c_2025_26.parquet"
    quotes = processed / "nba_kalshi_pregame_t30_2025_26.parquet"
    for path in (predictions, quotes):
        if not path.is_file():
            raise ConfigError(f"Missing {path}. Run the earlier phases first.")

    model = pd.read_parquet(predictions)[
        ["nba_game_id", "home_win", "probability_3a3c_native"]
    ].rename(columns={"probability_3a3c_native": "p_home"})
    market = pd.read_parquet(quotes)
    merged = model.merge(market, on="nba_game_id", how="inner", suffixes=("", "_mkt"))
    if len(merged) != len(model):
        raise ConfigError(
            f"expected every model game to have a quote: {len(merged)} of {len(model)}"
        )
    # The normalized midpoint lives with the earlier predictions. It is a
    # *forecasting benchmark* only -- never an executable price, and never a
    # model input.
    benchmark = processed / "nba_predictions_3a3_2025_26.parquet"
    if benchmark.is_file():
        prior = pd.read_parquet(benchmark)
        column = "kalshi_home_probability_normalized"
        if column in prior.columns:
            merged = merged.merge(
                prior[["nba_game_id", column]], on="nba_game_id", how="left"
            )
    if "kalshi_home_probability_normalized" not in merged.columns:
        raise ConfigError(
            "kalshi_home_probability_normalized not found; it is needed as the "
            "midpoint benchmark"
        )

    merged["p_away"] = 1.0 - merged["p_home"]
    merged["away_win"] = 1 - merged["home_win"].astype(int)
    merged["trade_date"] = pd.to_datetime(
        merged["game_datetime_utc"], utc=True
    ).dt.tz_convert("America/New_York").dt.date
    return merged


def verify_frozen_model(frame: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """The frozen model must be used exactly, not approximately."""
    published = settings.paths.processed / "nba_predictions_3a4_2025_26.parquet"
    result: dict[str, Any] = {"checked": False}
    if not published.is_file():
        return result
    prior = pd.read_parquet(published)[["nba_game_id", "probability_phase_3a4"]]
    merged = frame[["nba_game_id", "p_home"]].merge(prior, on="nba_game_id")
    difference = (merged["p_home"] - merged["probability_phase_3a4"]).abs()
    return {
        "checked": True,
        "n_games": len(merged),
        "max_abs_difference": float(difference.max()),
        "identical": bool(difference.max() == 0.0),
        "note": "Phase 3A4 froze the Phase 3A3C logistic, so the two must agree exactly",
    }


def market_pair_sanity(frame: pd.DataFrame) -> dict[str, Any]:
    """Audit the two mutually exclusive contracts against each other.

    Nothing is repaired. A crossed or wide pair is an observation about the
    market, and quietly fixing it would erase the very thing worth reporting.
    """
    bid_sum = frame["home_yes_bid"] + frame["away_yes_bid"]
    ask_sum = frame["home_yes_ask"] + frame["away_yes_ask"]
    mid_sum = frame["home_market_midpoint"] + frame["away_market_midpoint"]

    def describe(series: pd.Series) -> dict[str, float]:
        return {
            "min": float(series.min()), "p05": float(series.quantile(0.05)),
            "median": float(series.median()), "p95": float(series.quantile(0.95)),
            "max": float(series.max()), "mean": float(series.mean()),
        }

    return {
        "bid_sum": describe(bid_sum),
        "ask_sum": describe(ask_sum),
        "midpoint_sum": describe(mid_sum),
        "crossed_bid_sum_above_one": int((bid_sum > 1.0).sum()),
        "ask_sum_below_one": int((ask_sum < 1.0).sum()),
        "combined_spread_over_3c": int(((ask_sum - bid_sum) > 0.03).sum()),
        "home_spread_distribution": (
            frame["home_spread"].round(4).value_counts().sort_index().to_dict()
        ),
        "away_spread_distribution": (
            frame["away_spread"].round(4).value_counts().sort_index().to_dict()
        ),
        "quote_age_seconds": describe(frame["max_quote_age_seconds"]),
        "both_sides_usable": int(frame["both_sides_usable"].sum()),
        "games": len(frame),
        "interpretation": (
            "ask_sum above 1.0 is the round-trip cost of the spread and is "
            "normal; a bid_sum above 1.0 would be a genuine arbitrage and is "
            "counted separately"
        ),
    }


def add_edges(frame: pd.DataFrame, contracts: int) -> pd.DataFrame:
    """Gross and net executable edge for each side, at one order size."""
    out = frame.copy()
    out["delta_mid_home"] = out["p_home"] - out["kalshi_home_probability_normalized"]

    for side in SIDES:
        probability = out["p_home"] if side.name == "HOME" else out["p_away"]
        ask = out[side.price_column]
        out[f"gross_edge_{side.name.lower()}"] = probability - ask
        fees = [
            taker_fee_per_contract(float(price), contracts, when)
            for price, when in zip(ask, out["trade_date"], strict=True)
        ]
        out[f"fee_per_contract_{side.name.lower()}"] = fees
        out[f"net_edge_{side.name.lower()}"] = (
            out[f"gross_edge_{side.name.lower()}"] - np.asarray(fees)
        )

    home_better = out["gross_edge_home"] > out["gross_edge_away"]
    out["best_side"] = np.where(home_better, "HOME", "AWAY")
    out["best_gross_edge"] = np.where(
        home_better, out["gross_edge_home"], out["gross_edge_away"]
    )
    out["best_net_edge"] = np.where(
        home_better, out["net_edge_home"], out["net_edge_away"]
    )
    out["best_ask"] = np.where(
        home_better, out["home_yes_ask"], out["away_yes_ask"]
    )
    out["best_side_won"] = np.where(
        home_better, out["home_win"], out["away_win"]
    ).astype(int)
    out["best_fee_per_contract"] = np.where(
        home_better,
        out["fee_per_contract_home"],
        out["fee_per_contract_away"],
    )
    return out


def disagreement_calibration(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Who is more right, and where, across predeclared disagreement bins.

    Reported symmetrically: bins where the model likes the home side *less*
    than the market matter as much as bins where it likes it more. Looking only
    at the side the model prefers would answer a different, easier question.
    """
    truth = frame["home_win"].astype(int).to_numpy()
    model = frame["p_home"].to_numpy()
    market = frame["kalshi_home_probability_normalized"].to_numpy()
    delta = model - market

    rows: list[dict[str, Any]] = []
    for low, high, label in DISAGREEMENT_BINS:
        mask = (delta >= low) & (delta < high)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": label, "games": 0})
            continue
        model_brier = float(np.mean((model[mask] - truth[mask]) ** 2))
        market_brier = float(np.mean((market[mask] - truth[mask]) ** 2))
        rows.append({
            "bin": label,
            "games": n,
            "mean_model_probability": float(model[mask].mean()),
            "mean_market_probability": float(market[mask].mean()),
            "actual_home_win_rate": float(truth[mask].mean()),
            "model_brier": model_brier,
            "market_brier": market_brier,
            "model_minus_market_brier": model_brier - market_brier,
        })
    return rows


def disagreement_information(frame: pd.DataFrame) -> dict[str, Any]:
    """Does larger disagreement mean the model is more right, or less?"""
    truth = frame["home_win"].astype(int).to_numpy()
    model = frame["p_home"].to_numpy()
    market = frame["kalshi_home_probability_normalized"].to_numpy()
    delta = model - market
    magnitude = np.abs(delta)

    model_losses = metrics.brier_losses(truth, model)
    market_losses = metrics.brier_losses(truth, market)
    paired = model_losses - market_losses

    quartiles = pd.Series(pd.qcut(magnitude, 4, labels=["Q1", "Q2", "Q3", "Q4"]))
    by_quartile = []
    for label in ["Q1", "Q2", "Q3", "Q4"]:
        mask = (quartiles == label).to_numpy()
        by_quartile.append({
            "quartile": label,
            "games": int(mask.sum()),
            "mean_abs_disagreement": float(magnitude[mask].mean()),
            "model_brier": float(model_losses[mask].mean()),
            "market_brier": float(market_losses[mask].mean()),
            "model_minus_market": float(paired[mask].mean()),
            "bootstrap": metrics.paired_bootstrap(
                model_losses[mask], market_losses[mask],
                n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
            ),
        })

    residual = truth - market
    return {
        "correlation_disagreement_vs_market_residual": float(
            np.corrcoef(delta, residual)[0, 1]
        ),
        "correlation_note": (
            "positive means that when the model likes the home side more than "
            "the market does, the home side does tend to win more often than "
            "the market implied"
        ),
        "by_disagreement_quartile": by_quartile,
        "overall_model_minus_market_brier": float(paired.mean()),
        "overall_bootstrap": metrics.paired_bootstrap(
            model_losses, market_losses,
            n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
        ),
    }


def block_bootstrap_mean(
    values: np.ndarray, blocks: np.ndarray, *, n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED, confidence: float = 0.95,
) -> dict[str, Any]:
    """Cluster-aware bootstrap of a mean, resampling whole days.

    Games on the same night share weather, news cycles and scheduling, so
    resampling individual games would understate uncertainty. Days are drawn
    with replacement instead.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"n": 0}
    unique = pd.unique(blocks)
    grouped = [values[blocks == block] for block in unique]
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        picked = rng.integers(0, len(grouped), size=len(grouped))
        pooled = np.concatenate([grouped[j] for j in picked])
        means[i] = pooled.mean()
    tail = (1.0 - confidence) / 2.0
    return {
        "n": int(values.size),
        "n_blocks": len(unique),
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, tail)),
        "ci_high": float(np.quantile(means, 1.0 - tail)),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "seed": seed,
        "method": "block bootstrap over game dates",
    }


def simulate_threshold(
    frame: pd.DataFrame, threshold: float, contracts: int
) -> dict[str, Any]:
    """Settlement simulation for one predeclared threshold.

    A trade is taken only where the net executable edge clears the threshold.
    Buying YES at the ask pays $1 on the outcome and $0 otherwise, less the
    historically applicable fee. No slippage is modelled -- we have no depth
    data, and inventing a number would be worse than naming the omission.
    """
    traded = frame[frame["best_net_edge"] >= threshold].copy()
    eligible = len(frame)
    if traded.empty:
        return {
            "threshold": threshold, "contracts": contracts, "trades": 0,
            "coverage": 0.0, "eligible_games": eligible,
        }

    cost = traded["best_ask"].to_numpy() * contracts
    fees = traded["best_fee_per_contract"].to_numpy() * contracts
    payout = traded["best_side_won"].to_numpy() * contracts
    net = payout - cost - fees
    spent = cost.sum()

    return {
        "threshold": threshold,
        "contracts": contracts,
        "eligible_games": eligible,
        "trades": len(traded),
        "coverage": float(len(traded) / eligible),
        "home_trades": int((traded["best_side"] == "HOME").sum()),
        "away_trades": int((traded["best_side"] == "AWAY").sum()),
        "wins": int(traded["best_side_won"].sum()),
        "losses": int(len(traded) - traded["best_side_won"].sum()),
        "hit_rate": float(traded["best_side_won"].mean()),
        "mean_ask": float(traded["best_ask"].mean()),
        "mean_net_edge": float(traded["best_net_edge"].mean()),
        "dollars_spent": float(spent),
        "gross_settlement_pnl": float((payout - cost).sum()),
        "fees_paid": float(fees.sum()),
        "net_pnl": float(net.sum()),
        "roi_on_dollars_spent": float(net.sum() / spent) if spent else None,
        "mean_pnl_per_trade": float(net.mean()),
        "median_pnl_per_trade": float(np.median(net)),
        "pnl_per_trade_block_bootstrap": block_bootstrap_mean(
            net, traded["trade_date"].to_numpy()
        ),
    }


def segment_diagnostics(
    frame: pd.DataFrame, settings: Settings, threshold: float, contracts: int
) -> dict[str, Any]:
    """Descriptive breakdowns. None of these choose anything."""
    traded = frame[frame["best_net_edge"] >= threshold].copy()

    def summarise(subset: pd.DataFrame, label: str) -> dict[str, Any]:
        if len(subset) < 10:
            return {"segment": label, "trades": len(subset),
                    "note": "too few to report"}
        cost = subset["best_ask"].to_numpy() * contracts
        fees = subset["best_fee_per_contract"].to_numpy() * contracts
        net = subset["best_side_won"].to_numpy() * contracts - cost - fees
        return {
            "segment": label,
            "trades": len(subset),
            "hit_rate": float(subset["best_side_won"].mean()),
            "net_pnl": float(net.sum()),
            "mean_pnl_per_trade": float(net.mean()),
            "roi": float(net.sum() / cost.sum()) if cost.sum() else None,
        }

    out: dict[str, Any] = {}

    # Spread: the round-trip cost of the pair.
    combined = (traded["home_yes_ask"] + traded["away_yes_ask"]
                - traded["home_yes_bid"] - traded["away_yes_bid"]).round(2)
    out["by_combined_spread"] = [
        summarise(traded[combined == value], f"combined spread {value:.2f}")
        for value in sorted(combined.unique())
    ]

    # Liquidity, where the data exists.
    for column, label in (("home_candle_volume", "volume"),
                          ("home_open_interest", "open_interest")):
        if column in traded and traded[column].notna().any():
            quartile = pd.qcut(traded[column], 4, labels=["Q1", "Q2", "Q3", "Q4"],
                               duplicates="drop")
            out[f"by_{label}_quartile"] = [
                summarise(traded[quartile == q], f"{label} {q}")
                for q in quartile.cat.categories
            ]

    # Quote age bands.
    age = traded["max_quote_age_seconds"]
    bands = [(0, 60, "0-60s"), (60, 300, "60-300s"), (300, 1e9, ">300s")]
    out["by_quote_age"] = [
        summarise(traded[(age >= lo) & (age < hi)], label) for lo, hi, label in bands
    ]

    # Availability burden, from the Phase 3A3C features.
    features_path = (
        settings.paths.processed / "nba_game_availability_features_2019_26.parquet"
    )
    if features_path.is_file():
        availability = pd.read_parquet(features_path)
        availability = availability[availability["season"] == HOLDOUT_SEASON]
        columns = [c for c in (
            "nba_game_id", "avail_home_t30_out_expected_minutes",
            "avail_away_t30_out_expected_minutes",
            "avail_home_t30_questionable_expected_minutes",
            "avail_away_t30_questionable_expected_minutes",
        ) if c in availability.columns]
        merged = traded.merge(availability[columns], on="nba_game_id", how="left")
        burden = (merged["avail_home_t30_out_expected_minutes"]
                  - merged["avail_away_t30_out_expected_minutes"]).abs()
        questionable = merged[[
            "avail_home_t30_questionable_expected_minutes",
            "avail_away_t30_questionable_expected_minutes",
        ]].max(axis=1)
        cuts = [
            (burden <= burden.quantile(0.33), "low burden"),
            ((burden > burden.quantile(0.33)) & (burden <= burden.quantile(0.66)),
             "medium burden"),
            (burden > burden.quantile(0.66), "high burden"),
            (burden >= burden.quantile(0.75), "top quartile OUT burden"),
            (questionable >= 20.0, "high-minute questionable"),
        ]
        out["by_availability_burden"] = [
            summarise(merged[mask.fillna(False)], label) for mask, label in cuts
        ]

    # Market favourite / underdog, relative to the market's own view.
    market = traded["kalshi_home_probability_normalized"]
    model = traded["p_home"]
    favourite_is_home = market >= 0.5
    out["by_market_view"] = [
        summarise(
            traded[favourite_is_home & (model > market)],
            "model likes the market favourite more"),
        summarise(
            traded[favourite_is_home & (model < market)],
            "model thinks the favourite is overpriced"),
        summarise(
            traded[(~favourite_is_home) & (model < market)],
            "model prefers the market underdog"),
        summarise(
            traded[(model - market).abs() <= 0.01],
            "model and market essentially agree"),
    ]
    return out


def maker_sensitivity(frame: pd.DataFrame, contracts: int) -> dict[str, Any]:
    """Hypothetical maker economics. Explicitly not realised P&L.

    A resting bid at the current best bid would only trade if someone crossed
    to it. We hold no fill data, so this cannot count as a backtest -- it says
    what the arithmetic *would* be, conditional on a fill we never observed.
    """
    rows = []
    for side in SIDES:
        bid_column = side.price_column.replace("_ask", "_bid")
        probability = frame["p_home"] if side.name == "HOME" else frame["p_away"]
        bid = frame[bid_column]
        fees = [
            (maker_fee(float(price), contracts, when) or 0.0) / contracts
            for price, when in zip(bid, frame["trade_date"], strict=True)
        ]
        edge = probability - bid - np.asarray(fees)
        rows.append({
            "side": side.name,
            "mean_hypothetical_net_edge_at_bid": float(edge.mean()),
            "games_with_positive_hypothetical_edge": int((edge > 0).sum()),
        })
    return {
        "status": "HYPOTHETICAL - not realised P&L",
        "reason": (
            "a resting order is not a fill; no fill evidence exists in the "
            "candlestick data, so this cannot count as a backtest"
        ),
        "by_side": rows,
    }


def leakage_audit(frame: pd.DataFrame, verification: dict[str, Any]) -> dict[str, Any]:
    """Structural checks on the research itself."""
    quote_ts = pd.to_datetime(frame["home_quote_ts_utc"], utc=True)
    anchor = pd.to_datetime(frame["prediction_ts_utc"], utc=True)
    away_ts = pd.to_datetime(frame["away_quote_ts_utc"], utc=True)

    fee_dates_resolvable = True
    try:
        for when in pd.unique(frame["trade_date"]):
            schedule_for(when)
    except FeeScheduleUnavailableError:
        fee_dates_resolvable = False

    return {
        "frozen_model_identical_to_phase_3a4": verification.get("identical"),
        "all_quotes_at_or_before_anchor": bool(
            (quote_ts <= anchor).all() and (away_ts <= anchor).all()
        ),
        "no_quote_after_anchor": int(
            ((quote_ts > anchor) | (away_ts > anchor)).sum()
        ),
        "executable_price_is_ask_not_midpoint": True,
        "midpoint_used_only_as_forecast_benchmark": True,
        "maker_fills_not_counted_as_realised": True,
        "thresholds_predetermined": list(EDGE_THRESHOLDS),
        "disagreement_bins_predetermined": [b[2] for b in DISAGREEMENT_BINS],
        "fee_schedule_resolvable_for_every_trade_date": fee_dates_resolvable,
        "no_outcome_in_trade_decision": (
            "trade selection uses p_home, the quoted ask and the fee schedule "
            "only; the outcome enters settlement alone"
        ),
        "kalshi_never_a_model_input": (
            "the Phase 3A3C allowlist contains no market-derived column; "
            "verified structurally in the Phase 3A4 audits"
        ),
        "2025_26_status": (
            "EXPLORATORY, not a pristine strategy holdout: it was inspected "
            "repeatedly during model development, and no earlier NBA "
            "regular-season market history exists to develop rules on"
        ),
    }


def kalshi_history_audit(settings: Settings) -> dict[str, Any]:
    """What NBA market history exists, established empirically not from docs."""
    events_path = settings.paths.processed / "kalshi_nba_events_2025_26.parquet"
    ingested: dict[str, Any] = {}
    if events_path.is_file():
        events = pd.read_parquet(events_path)
        dates = pd.to_datetime(events["scheduled_game_date"], errors="coerce").dropna()
        ingested = {
            "events": len(events),
            "date_range": [str(dates.min().date()), str(dates.max().date())],
        }
    return {
        "series_probed": "KXNBAGAME",
        "ingested": ingested,
        "empirical_findings": {
            "earliest_kxnbagame_event": "2025-04-15",
            "events_before_2024_25_regular_season_end_2025_04_13": 0,
            "pre_2025_26_events": (
                "86 events across 2025-04, 2025-05 and 2025-06, which are the "
                "2024-25 play-in and playoffs, not regular-season games"
            ),
            "legacy_series_probed": [
                "NBAGAME", "NBA", "NBAWIN", "NBAWINNER", "PRONBAGAME", "NBAG",
                "NBAMONEYLINE", "NBAML",
            ],
            "legacy_series_with_events": "none returned any events",
            "nba_series_in_catalogue": (
                "KXNBAGAME is the only NBA game-winner series; KXNBA is "
                "championship futures, not game level"
            ),
        },
        "conclusion": (
            "no NBA regular-season market history exists before 2025-26, so "
            "market rules cannot be developed on an earlier season and "
            "evaluated on this one. 2025-26 is exploratory."
        ),
        "seasons": {
            "2022-23": "no KXNBAGAME markets",
            "2023-24": "no KXNBAGAME markets",
            "2024-25": "play-in and playoffs only, no regular season",
            "2025-26": "full regular-season coverage, 1,230 games with T-30 quotes",
        },
    }


def run_pipeline(*, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()

    frame = load_frame(settings)
    verification = verify_frozen_model(frame, settings)
    if verification.get("checked") and not verification.get("identical"):
        raise ConfigError(
            "frozen model mismatch: Phase 4A0 must use the Phase 3A3C "
            f"probabilities exactly (max diff {verification['max_abs_difference']})"
        )

    enriched = add_edges(frame, PRIMARY_SIZE)

    thresholds: dict[str, list[dict[str, Any]]] = {}
    for contracts in CONTRACT_SIZES:
        sized = add_edges(frame, contracts)
        thresholds[str(contracts)] = [
            simulate_threshold(sized, threshold, contracts)
            for threshold in EDGE_THRESHOLDS
        ]

    report = {
        "generated_at_utc": utc_now().isoformat(),
        "status": "RESEARCH ONLY - exploratory, not a validated strategy",
        "kalshi_history_audit": kalshi_history_audit(settings),
        "frozen_model_verification": verification,
        "fee_schedules": [s.to_dict() for s in FEE_SCHEDULES],
        "fee_size_dependence": {
            "note": (
                "fees round up to the next cent on the whole order, so the "
                "per-contract cost falls with size; 100 contracts is "
                "rounding-stable, which is why it is the headline view"
            ),
            "example_at_50c": {
                str(c): taker_fee_per_contract(
                    0.50, c, next(iter(FEE_SCHEDULES)).effective_from
                )
                for c in CONTRACT_SIZES
            },
        },
        "market_pair_sanity": market_pair_sanity(frame),
        "disagreement_distribution": {
            "delta_mid_home": {
                "mean": float(enriched["delta_mid_home"].mean()),
                "std": float(enriched["delta_mid_home"].std()),
                "p05": float(enriched["delta_mid_home"].quantile(0.05)),
                "median": float(enriched["delta_mid_home"].median()),
                "p95": float(enriched["delta_mid_home"].quantile(0.95)),
                "mean_abs": float(enriched["delta_mid_home"].abs().mean()),
            },
            "best_gross_edge": {
                "mean": float(enriched["best_gross_edge"].mean()),
                "p95": float(enriched["best_gross_edge"].quantile(0.95)),
                "max": float(enriched["best_gross_edge"].max()),
                "games_with_positive_gross_edge": int(
                    (enriched["best_gross_edge"] > 0).sum()
                ),
            },
            "best_net_edge": {
                "mean": float(enriched["best_net_edge"].mean()),
                "p95": float(enriched["best_net_edge"].quantile(0.95)),
                "max": float(enriched["best_net_edge"].max()),
                "games_with_positive_net_edge": int(
                    (enriched["best_net_edge"] > 0).sum()
                ),
            },
        },
        "disagreement_calibration": disagreement_calibration(frame),
        "disagreement_information": disagreement_information(frame),
        "thresholds_by_size": thresholds,
        "primary_size": PRIMARY_SIZE,
        "segments_at_threshold_0": segment_diagnostics(
            enriched, settings, 0.0, PRIMARY_SIZE
        ),
        "maker_sensitivity": maker_sensitivity(frame, PRIMARY_SIZE),
        "leakage_audit": leakage_audit(frame, verification),
        "limitations": [
            "2025-26 was inspected throughout model development, so it is not a "
            "pristine strategy holdout and no result here is validation",
            "no depth or order-book data, so slippage beyond the quoted ask is "
            "not modelled; real fills at size could be worse",
            "quotes are one-minute candlestick aggregates, not a live book",
            "maker economics are hypothetical because no fill evidence exists",
            "1,230 games is a small sample for a strategy with low coverage",
        ],
    }

    columns = [
        "nba_game_id", "game_datetime_utc", "trade_date", "home_team", "away_team",
        "home_win", "p_home", "p_away", "kalshi_home_probability_normalized",
        "home_yes_bid", "home_yes_ask", "away_yes_bid", "away_yes_ask",
        "home_market_midpoint", "away_market_midpoint", "home_spread", "away_spread",
        "home_candle_volume", "home_open_interest", "max_quote_age_seconds",
        "delta_mid_home", "gross_edge_home", "gross_edge_away",
        "net_edge_home", "net_edge_away", "best_side", "best_ask",
        "best_gross_edge", "best_net_edge", "best_side_won", "best_fee_per_contract",
    ]
    out_path = settings.paths.processed / "nba_market_edge_t30_2025_26.parquet"
    enriched[[c for c in columns if c in enriched.columns]].to_parquet(
        out_path, index=False
    )

    rows = []
    for size, results in thresholds.items():
        for r in results:
            rows.append({"contracts": size, **r})
    csv_path = settings.paths.reports / "market_edge_thresholds_2025_26.csv"
    pd.DataFrame(rows).drop(columns=["pnl_per_trade_block_bootstrap"],
                            errors="ignore").to_csv(csv_path, index=False)

    report_path = settings.paths.reports / "market_edge_audit_2025_26.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"] = [str(out_path), str(csv_path), str(report_path)]
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Phase 4A0 Kalshi market edge audit.")


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    try:
        report = run_pipeline()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"\n{'threshold':>10s} {'trades':>7s} {'cover':>7s} {'hit':>7s} "
          f"{'spent':>11s} {'fees':>9s} {'net P&L':>10s} {'ROI':>8s}")
    for r in report["thresholds_by_size"][str(PRIMARY_SIZE)]:
        if r["trades"] == 0:
            print(f"{r['threshold']:>10.3f} {0:>7d}       -       -           -"
                  f"         -          -        -")
            continue
        print(f"{r['threshold']:>10.3f} {r['trades']:>7d} {r['coverage']:>6.1%} "
              f"{r['hit_rate']:>6.1%} {r['dollars_spent']:>11,.0f} "
              f"{r['fees_paid']:>9,.0f} {r['net_pnl']:>10,.0f} "
              f"{r['roi_on_dollars_spent']:>7.2%}")
    for path in report["written_files"]:
        print(f"\nWrote {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
