"""Anchor-specific forecasts, and how the market compares at each one.

Phase 4A0 asked whether our disagreement with Kalshi is tradable at T-30 and
found it is not. That is a statement about one instant. Earlier in the day the
injury picture is incomplete and the market has had less time to price it, so
the same question has to be asked separately at each anchor.

Reusing the T-30 prediction at T-6h would answer nothing: it embeds availability
information that did not exist six hours before tip. So the *same frozen Phase
3A3C architecture* is retrained at each anchor on information available then --
same feature families, same history policy, same small C grid, chosen on
development folds only.

Nothing about 2025-26 selects anything. All four anchor models are frozen before
the market is looked at.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.features.availability_features import STATUS_ORDER
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.models import metrics
from nba_prediction_market.models.availability_bundles import (
    PHASE_3A3_FEATURES,
    ROLE_MINUTE_FAMILY,
)
from nba_prediction_market.models.logistic import LogisticConfig, fit_logistic
from nba_prediction_market.models.selection import assert_no_holdout
from nba_prediction_market.models.status_calibration import calibrate
from nba_prediction_market.pipelines.build_availability_model import (
    AVAILABILITY_SEASONS,
    C_GRID,
    DEVELOPMENT_SEASONS,
    HOLDOUT_SEASON,
    MAX_TRAINING_SEASONS,
    TARGET,
    evaluate_predictions,
    load_observations,
    training_seasons_for,
)

logger = logging.getLogger(__name__)

#: Anchors studied, ordered from earliest to the decision anchor.
ANCHORS: tuple[tuple[str, str, int], ...] = (
    ("t6h", "T-6h", 360),
    ("t3h", "T-3h", 180),
    ("t1h", "T-1h", 60),
    ("t30", "T-30m", 30),
)

#: The frozen Phase 3A3C bundle C allowlist, unchanged. Only the anchor the
#: availability half is measured at varies.
ANCHOR_BUNDLE: tuple[str, ...] = PHASE_3A3_FEATURES + ROLE_MINUTE_FAMILY


def load_anchor_features(settings: Settings) -> pd.DataFrame:
    """Model features joined to availability measured at every anchor."""
    processed = settings.paths.processed
    path = processed / "nba_game_availability_features_2019_26_anchors.parquet"
    if not path.is_file():
        raise ConfigError(
            f"Missing {path}. Build availability features with MULTI_ANCHORS first."
        )
    availability = pd.read_parquet(path)
    model = pd.read_parquet(processed / "nba_model_features_3a3_2006_26.parquet")
    merged = model.merge(
        availability.drop(
            columns=["season", "home_team", "away_team", "game_datetime_utc"],
            errors="ignore",
        ),
        on="nba_game_id", how="inner",
    )
    return merged.sort_values("game_datetime_utc", kind="stable")


def derive_anchor_features(
    frame: pd.DataFrame, calibration: Any, anchor: str
) -> pd.DataFrame:
    """Build the bundle's availability columns from one anchor's state.

    The column *names* match the frozen Phase 3A3C allowlist so the same model
    specification can be fitted; only the anchor they are measured at changes.
    A game with no report at this anchor gets nulls, which the train-only
    imputer handles -- never a zero, which would assert nobody was designated.
    """
    out = frame.copy()
    covered = out.get(f"avail_{anchor}_covered")
    mask = covered.fillna(False).astype(bool) if covered is not None else None

    for status in STATUS_ORDER:
        home = out.get(f"avail_home_{anchor}_{status}_expected_minutes")
        away = out.get(f"avail_away_{anchor}_{status}_expected_minutes")
        column = f"avail_{status}_expected_minutes_diff"
        if home is None or away is None:
            out[column] = np.nan
            continue
        difference = home - away
        out[column] = difference.where(mask) if mask is not None else difference
    return out


def fit_anchor_model(
    frame: pd.DataFrame, anchor: str, observations: list[Any]
) -> dict[str, Any]:
    """Select C on development folds, then freeze. 2025-26 selects nothing."""
    assert_no_holdout(list(DEVELOPMENT_SEASONS), where=f"anchor {anchor}")

    fold_rows: list[dict[str, Any]] = []
    for season in DEVELOPMENT_SEASONS:
        training = training_seasons_for(season, AVAILABILITY_SEASONS)
        assert_no_holdout(training, where=f"anchor {anchor} fold {season}")
        scoped = derive_anchor_features(
            frame, calibrate(observations, training), anchor
        )
        train = scoped[scoped["season"].isin(training)]
        valid = scoped[scoped["season"] == season]
        for c_value in C_GRID:
            config = LogisticConfig(
                training_history=MAX_TRAINING_SEASONS, c_value=c_value
            )
            fitted = fit_logistic(train, config, features=ANCHOR_BUNDLE)
            probability = fitted.predict_proba(valid)
            truth = valid[TARGET].astype(int).to_numpy()
            fold_rows.append({
                "anchor": anchor, "season": season, "C": c_value,
                "brier": metrics.brier_score(truth, probability),
                "log_loss": metrics.log_loss(truth, probability),
                "auc": metrics.roc_auc(truth, probability),
            })

    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby("C")
        .agg(mean_brier=("brier", "mean"), mean_log_loss=("log_loss", "mean"),
             mean_auc=("auc", "mean"))
        .reset_index()
        .sort_values(["mean_brier", "mean_log_loss"])
    )
    chosen = float(summary.iloc[0]["C"])
    return {
        "anchor": anchor,
        "chosen_C": chosen,
        "development_summary": summary.to_dict("records"),
        "development_folds": fold_rows,
        "features": list(ANCHOR_BUNDLE),
    }


def score_holdout(
    frame: pd.DataFrame, anchor: str, c_value: float, observations: list[Any]
) -> pd.DataFrame:
    """Frozen anchor model applied to 2025-26. Nothing is selected here."""
    training = training_seasons_for(HOLDOUT_SEASON, AVAILABILITY_SEASONS)
    scoped = derive_anchor_features(
        frame, calibrate(observations, training), anchor
    )
    train = scoped[scoped["season"].isin(training)]
    holdout = scoped[scoped["season"] == HOLDOUT_SEASON]
    config = LogisticConfig(training_history=MAX_TRAINING_SEASONS, c_value=c_value)
    fitted = fit_logistic(train, config, features=ANCHOR_BUNDLE)
    return pd.DataFrame({
        "nba_game_id": holdout["nba_game_id"].to_numpy(),
        "home_win": holdout[TARGET].astype(int).to_numpy(),
        f"p_home_{anchor}": fitted.predict_proba(holdout),
        f"avail_covered_{anchor}": (
            holdout.get(f"avail_{anchor}_covered", pd.Series(True, index=holdout.index))
            .fillna(False).astype(bool).to_numpy()
        ),
    })


def market_at_anchors(settings: Settings) -> pd.DataFrame:
    """Home/away quotes pivoted to one row per game and anchor."""
    path = (
        settings.paths.processed / "nba_market_multi_anchor_2025_26.parquet"
    )
    if not path.is_file():
        raise ConfigError(f"Missing {path}. Run build_multi_anchor_market first.")
    raw = pd.read_parquet(path)
    home = raw[raw["side"] == "home"].set_index(["nba_game_id", "anchor_minutes"])
    away = raw[raw["side"] == "away"].set_index(["nba_game_id", "anchor_minutes"])
    joined = home.join(away, lsuffix="_home", rsuffix="_away").reset_index()

    midpoint_sum = joined["midpoint_home"] + joined["midpoint_away"]
    # Normalised midpoint is the forecasting benchmark. It is never an
    # executable price -- the asks are.
    joined["market_p_home"] = np.where(
        midpoint_sum > 0, joined["midpoint_home"] / midpoint_sum, np.nan
    )
    joined["both_usable"] = joined["usable_home"] & joined["usable_away"]
    return joined


def compare_at_anchor(
    predictions: pd.DataFrame, market: pd.DataFrame, anchor: str, minutes: int
) -> dict[str, Any]:
    """Model and market forecasting quality on the same games at one anchor."""
    scoped = market[market["anchor_minutes"] == minutes]
    merged = predictions.merge(
        scoped[["nba_game_id", "market_p_home", "both_usable",
                "yes_ask_home", "yes_ask_away", "yes_bid_home", "yes_bid_away",
                "spread_home", "spread_away", "quote_age_seconds_home"]],
        on="nba_game_id", how="inner",
    )
    usable = merged[
        merged["both_usable"].fillna(False)
        & merged["market_p_home"].notna()
        & merged[f"p_home_{anchor}"].notna()
    ]
    if usable.empty:
        return {"anchor": anchor, "games": 0}

    truth = usable["home_win"].to_numpy()
    model = usable[f"p_home_{anchor}"].to_numpy()
    market_p = usable["market_p_home"].to_numpy()
    model_losses = metrics.brier_losses(truth, model)
    market_losses = metrics.brier_losses(truth, market_p)

    return {
        "anchor": anchor,
        "anchor_minutes": minutes,
        "games": len(usable),
        "model": evaluate_predictions(truth, model),
        "market": evaluate_predictions(truth, market_p),
        "model_minus_market_brier": float(
            model_losses.mean() - market_losses.mean()
        ),
        "bootstrap": metrics.paired_bootstrap(
            model_losses, market_losses, n_resamples=10_000, seed=20260825
        ),
        "mean_abs_disagreement": float(np.abs(model - market_p).mean()),
        "median_spread_home": float(usable["spread_home"].median()),
    }


def convergence_path(
    predictions: pd.DataFrame, market: pd.DataFrame
) -> dict[str, Any]:
    """How the model-market gap evolves from T-6h to T-30.

    Movement is decomposed: the model can move because a new injury report
    arrived, and the market can move because it repriced. Attributing all
    convergence to one side would be the easy mistake.
    """
    frames = []
    for anchor, _, minutes in ANCHORS:
        scoped = market[market["anchor_minutes"] == minutes][
            ["nba_game_id", "market_p_home", "both_usable"]
        ]
        merged = predictions[["nba_game_id", f"p_home_{anchor}"]].merge(
            scoped, on="nba_game_id", how="inner"
        )
        merged["anchor"] = anchor
        merged["minutes"] = minutes
        merged = merged.rename(columns={f"p_home_{anchor}": "p_model"})
        frames.append(merged[["nba_game_id", "anchor", "minutes", "p_model",
                              "market_p_home", "both_usable"]])
    stacked = pd.concat(frames, ignore_index=True)
    stacked["gap"] = stacked["p_model"] - stacked["market_p_home"]

    by_anchor = []
    for anchor, label, _minutes in ANCHORS:
        subset = stacked[
            (stacked["anchor"] == anchor) & stacked["both_usable"].fillna(False)
        ]
        by_anchor.append({
            "anchor": label,
            "games": len(subset),
            "mean_abs_gap": float(subset["gap"].abs().mean()),
            "median_abs_gap": float(subset["gap"].abs().median()),
            "mean_gap": float(subset["gap"].mean()),
        })

    wide = stacked.pivot_table(
        index="nba_game_id", columns="anchor",
        values=["p_model", "market_p_home", "gap"], aggfunc="first",
    )
    transitions = []
    order = [a[0] for a in ANCHORS]
    for earlier, later in pairwise(order):
        try:
            gap_a = wide[("gap", earlier)]
            gap_b = wide[("gap", later)]
            model_a = wide[("p_model", earlier)]
            model_b = wide[("p_model", later)]
            market_a = wide[("market_p_home", earlier)]
            market_b = wide[("market_p_home", later)]
        except KeyError:
            continue
        valid = gap_a.notna() & gap_b.notna()
        shrank = (gap_b.abs() < gap_a.abs()) & valid
        expanded = (gap_b.abs() > gap_a.abs()) & valid
        flipped = (np.sign(gap_a) != np.sign(gap_b)) & valid
        transitions.append({
            "from": earlier, "to": later,
            "games": int(valid.sum()),
            "gap_shrank": int(shrank.sum()),
            "gap_expanded": int(expanded.sum()),
            "gap_changed_sign": int(flipped.sum()),
            "share_shrank": float(shrank.sum() / max(valid.sum(), 1)),
            "mean_abs_model_move": float((model_b - model_a).abs()[valid].mean()),
            "mean_abs_market_move": float((market_b - market_a).abs()[valid].mean()),
            "interpretation": (
                "model movement is new availability information; market "
                "movement is repricing. Comparing the two says which side "
                "closes the gap."
            ),
        })
    return {"by_anchor": by_anchor, "transitions": transitions}


def executable_edge_at_anchor(
    predictions: pd.DataFrame, market: pd.DataFrame, anchor: str, minutes: int,
    contracts: int = 100,
) -> list[dict[str, Any]]:
    """Phase 4A0's threshold sweep, unchanged, applied at one anchor.

    Same predeclared thresholds, same taker-at-the-ask execution, same
    date-effective fees. Thresholds are *not* re-chosen per anchor: doing that
    would turn four honest tests into one search.
    """
    from nba_prediction_market.models.kalshi_fees import taker_fee_per_contract
    from nba_prediction_market.pipelines.build_market_edge_audit import (
        EDGE_THRESHOLDS,
        block_bootstrap_mean,
    )

    scoped = market[market["anchor_minutes"] == minutes]
    merged = predictions.merge(
        scoped[["nba_game_id", "yes_ask_home", "yes_ask_away", "both_usable"]],
        on="nba_game_id", how="inner",
    )
    merged = merged[
        merged["both_usable"].fillna(False)
        & merged["yes_ask_home"].notna()
        & merged["yes_ask_away"].notna()
        & merged[f"p_home_{anchor}"].notna()
    ].copy()
    if merged.empty:
        return []

    merged["trade_date"] = pd.to_datetime(
        merged["game_datetime_utc"], utc=True
    ).dt.tz_convert("America/New_York").dt.date
    merged["p_away"] = 1.0 - merged[f"p_home_{anchor}"]
    merged["away_win"] = 1 - merged["home_win"]

    for side, probability_column, ask_column in (
        ("home", f"p_home_{anchor}", "yes_ask_home"),
        ("away", "p_away", "yes_ask_away"),
    ):
        fees = [
            taker_fee_per_contract(float(ask), contracts, when)
            for ask, when in zip(merged[ask_column], merged["trade_date"], strict=True)
        ]
        merged[f"fee_{side}"] = fees
        merged[f"net_edge_{side}"] = (
            merged[probability_column] - merged[ask_column] - np.asarray(fees)
        )

    home_better = merged["net_edge_home"] > merged["net_edge_away"]
    merged["best_net_edge"] = np.where(
        home_better, merged["net_edge_home"], merged["net_edge_away"]
    )
    merged["best_ask"] = np.where(
        home_better, merged["yes_ask_home"], merged["yes_ask_away"]
    )
    merged["best_side_won"] = np.where(
        home_better, merged["home_win"], merged["away_win"]
    ).astype(int)
    merged["best_fee"] = np.where(home_better, merged["fee_home"], merged["fee_away"])

    rows: list[dict[str, Any]] = []
    for threshold in EDGE_THRESHOLDS:
        traded = merged[merged["best_net_edge"] >= threshold]
        if traded.empty:
            rows.append({"anchor": anchor, "threshold": threshold, "trades": 0,
                         "coverage": 0.0})
            continue
        cost = traded["best_ask"].to_numpy() * contracts
        fees = traded["best_fee"].to_numpy() * contracts
        net = traded["best_side_won"].to_numpy() * contracts - cost - fees
        rows.append({
            "anchor": anchor,
            "threshold": threshold,
            "eligible_games": len(merged),
            "trades": len(traded),
            "coverage": float(len(traded) / len(merged)),
            "hit_rate": float(traded["best_side_won"].mean()),
            "dollars_spent": float(cost.sum()),
            "fees_paid": float(fees.sum()),
            "net_pnl": float(net.sum()),
            "roi": float(net.sum() / cost.sum()) if cost.sum() else None,
            "mean_pnl_per_trade": float(net.mean()),
            "block_bootstrap": block_bootstrap_mean(
                net, traded["trade_date"].to_numpy()
            ),
        })
    return rows


def run_pipeline(*, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()

    features = load_anchor_features(settings)
    observations = load_observations(settings)

    # --- fit and freeze every anchor model before looking at the market ---
    frozen: dict[str, Any] = {}
    predictions: pd.DataFrame | None = None
    for anchor, label, _ in ANCHORS:
        logger.info("fitting anchor model %s", label)
        selection = fit_anchor_model(features, anchor, observations)
        frozen[anchor] = selection
        scored = score_holdout(features, anchor, selection["chosen_C"], observations)
        predictions = scored if predictions is None else predictions.merge(
            scored.drop(columns=["home_win"]), on="nba_game_id", how="outer"
        )

    if predictions is None:
        raise ConfigError("no anchor produced predictions")
    meta = features[features["season"] == HOLDOUT_SEASON][
        ["nba_game_id", "game_datetime_utc", "home_team", "away_team"]
    ]
    predictions = predictions.merge(meta, on="nba_game_id", how="left")

    # --- only now is the market consulted ---
    market = market_at_anchors(settings)
    comparisons = [
        compare_at_anchor(predictions, market, anchor, minutes)
        for anchor, _, minutes in ANCHORS
    ]
    thresholds = {
        anchor: executable_edge_at_anchor(predictions, market, anchor, minutes)
        for anchor, _, minutes in ANCHORS
    }

    report = {
        "generated_at_utc": utc_now().isoformat(),
        "status": "RESEARCH ONLY - exploratory, not validated strategy research",
        "anchors": [{"key": a, "label": lbl, "minutes": m} for a, lbl, m in ANCHORS],
        "architecture": (
            "the frozen Phase 3A3C bundle C specification, refitted per anchor "
            "on information available at that anchor; no new feature-family "
            "selection was performed"
        ),
        "frozen_anchor_models": {
            anchor: {
                "chosen_C": selection["chosen_C"],
                "development_summary": selection["development_summary"],
                "features": selection["features"],
            }
            for anchor, selection in frozen.items()
        },
        "anchor_comparison": comparisons,
        "convergence": convergence_path(predictions, market),
        "executable_thresholds_by_anchor": thresholds,
    }

    out_path = settings.paths.processed / "nba_anchor_predictions_2025_26.parquet"
    predictions.to_parquet(out_path, index=False)
    report_path = settings.paths.reports / "market_convergence_audit_2025_26.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"] = [str(out_path), str(report_path)]
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Phase 4A1 anchor models.")


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    try:
        report = run_pipeline()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"\n{'anchor':8s} {'games':>6s} {'model B':>9s} {'market B':>9s} "
          f"{'diff':>9s} {'model AUC':>10s} {'mkt AUC':>8s} {'|disagree|':>11s}")
    for row in report["anchor_comparison"]:
        if row.get("games", 0) == 0:
            continue
        print(f"{row['anchor']:8s} {row['games']:6d} {row['model']['brier']:9.5f} "
              f"{row['market']['brier']:9.5f} {row['model_minus_market_brier']:+9.5f} "
              f"{row['model']['auc']:10.4f} {row['market']['auc']:8.4f} "
              f"{row['mean_abs_disagreement']:11.4f}")
    for path in report["written_files"]:
        print(f"\nWrote {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
