"""Phase 3A3C: does genuine point-in-time availability improve the forecast?

The question is incremental value over the frozen Phase 3A3 model, so every
development fold trains two models on **identical examples**: a control using
only the frozen Phase 3A3 features, and an enhanced model that adds exactly one
availability family. Any difference is then attributable to the family rather
than to a different training set, a different split, or a different anything.

The one piece that touches actual participation is the status calibration --
learning what "questionable" is worth. It is estimated per fold from that
fold's training seasons alone, so a validation game can never inform the
mapping applied to itself, and 2025-26 informs nothing at all.
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
from nba_prediction_market.features.availability_features import STATUS_ORDER, STATUS_RANK
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.models import metrics
from nba_prediction_market.models.availability_bundles import (
    ALL_AVAILABILITY_FEATURES,
    AVAILABILITY_BUNDLES,
    PHASE_3A3_FEATURES,
    conditional_bundle_g,
)
from nba_prediction_market.models.bundles import Bundle
from nba_prediction_market.models.logistic import LogisticConfig, fit_logistic
from nba_prediction_market.models.selection import assert_no_holdout
from nba_prediction_market.models.status_calibration import (
    StatusCalibration,
    StatusObservation,
    calibrate,
    check_ordering,
)

logger = logging.getLogger(__name__)

#: Seasons whose availability is complete enough to train on.
AVAILABILITY_SEASONS: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024, 2025)
#: Development validation seasons. 2025-26 is the holdout and appears nowhere.
DEVELOPMENT_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024)
HOLDOUT_SEASON: int = 2025
#: Most recent complete seasons a fold may train on.
MAX_TRAINING_SEASONS: int = 5
#: The small, predetermined regularisation grid.
C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)
TARGET = "home_win"


def training_seasons_for(validation_season: int, available: tuple[int, ...]) -> list[int]:
    """Complete availability-enabled seasons before ``validation_season``.

    A fixed, predetermined rule: take everything prior, capped at the most
    recent five. It is not searched over, so it cannot be tuned to a result.
    """
    prior = sorted(s for s in available if s < validation_season)
    return prior[-MAX_TRAINING_SEASONS:]


def load_observations(
    settings: Settings, suffix: str = ""
) -> list[StatusObservation]:
    path = (
        settings.paths.processed
        / f"nba_player_availability_observations_2019_26{suffix}.parquet"
    )
    if not path.is_file():
        raise ConfigError(f"Missing {path}. Run build_availability_features first.")
    frame = pd.read_parquet(path)
    return [
        StatusObservation(
            season=int(r.season),
            status=str(r.status),
            baseline_minutes=float(r.baseline_minutes),
            actual_minutes=float(r.actual_minutes),
        )
        for r in frame.itertuples()
    ]


def derive_features(
    frame: pd.DataFrame, calibration: StatusCalibration
) -> pd.DataFrame:
    """Add the fold's availability feature columns.

    Expected minutes lost is linear in the per-status role-weighted totals, so
    the fold's multipliers are applied here rather than baked into the parquet.
    Two folds therefore see different numbers from the same underlying reports,
    which is exactly right: each fold may only know what its training seasons
    taught it.
    """
    out = frame.copy()
    mapping = calibration.as_mapping()

    def side(name: str, anchor: str = "t30") -> tuple[pd.Series, pd.Series]:
        return (out[f"avail_home_{anchor}_{name}"], out[f"avail_away_{anchor}_{name}"])

    for status in STATUS_ORDER:
        home, away = side(f"{status}_count")
        out[f"avail_{status}_count_diff"] = home - away
        home, away = side(f"{status}_expected_minutes")
        out[f"avail_{status}_expected_minutes_diff"] = home - away

    def loss(anchor: str, side_name: str, quality: bool = False) -> pd.Series:
        kind = "expected_quality_minutes" if quality else "expected_minutes"
        total = pd.Series(0.0, index=out.index)
        for status, multiplier in mapping.items():
            column = f"avail_{side_name}_{anchor}_{status}_{kind}"
            if column in out.columns:
                total = total + out[column].fillna(0.0) * (1.0 - multiplier)
        return total

    for anchor in ("t30", "t1h", "t3h"):
        for side_name in ("home", "away"):
            out[f"avail_{side_name}_{anchor}_loss"] = loss(anchor, side_name)

    out["avail_home_expected_minutes_lost"] = out["avail_home_t30_loss"]
    out["avail_away_expected_minutes_lost"] = out["avail_away_t30_loss"]
    out["avail_expected_minutes_lost_diff"] = (
        out["avail_home_t30_loss"] - out["avail_away_t30_loss"]
    )
    out["avail_expected_quality_lost_diff"] = (
        loss("t30", "home", quality=True) - loss("t30", "away", quality=True)
    )

    # Movement into T-30 from each earlier anchor. Where the earlier report
    # does not exist the feature is null, not zero: unknown movement is not a
    # claim that nothing moved, and the train-only imputer handles the gap.
    for anchor, tag in (("t3h", "3h"), ("t1h", "1h")):
        covered = out[f"avail_{anchor}_covered"].fillna(False).astype(bool)
        for base in ("late_downgrades", "late_upgrades",
                     "newly_out_expected_minutes"):
            home = out[f"avail_home_{anchor}_to_t30_{base}"]
            away = out[f"avail_away_{anchor}_to_t30_{base}"]
            out[f"avail_{base}_{tag}_diff"] = (home - away).where(covered)
        change_home = out["avail_home_t30_loss"] - out[f"avail_home_{anchor}_loss"]
        change_away = out["avail_away_t30_loss"] - out[f"avail_away_{anchor}_loss"]
        out[f"avail_expected_loss_change_{tag}_diff"] = (
            change_home - change_away
        ).where(covered)

    # T-3h is the canonical "late news" window: it is the widest one still
    # entirely before the anchor, so it captures the most movement.
    out["avail_late_downgrades_diff"] = out["avail_late_downgrades_3h_diff"]
    out["avail_late_upgrades_diff"] = out["avail_late_upgrades_3h_diff"]

    return out


@dataclass
class FoldOutcome:
    """One bundle's result on one validation season."""

    season: int
    bundle: str
    c_value: float
    brier: float
    log_loss: float
    auc: float | None
    n_games: int
    predictions: pd.DataFrame


def evaluate_bundle(
    features: pd.DataFrame,
    bundle: Bundle,
    validation_season: int,
    training: list[int],
    c_value: float,
) -> FoldOutcome:
    """Fit on training seasons only, score the validation season."""
    assert_no_holdout(training, where=f"bundle {bundle.name} training")
    train = features[features["season"].isin(training)]
    valid = features[features["season"] == validation_season]
    if train.empty or valid.empty:
        raise ConfigError(f"empty split for season {validation_season}")

    config = LogisticConfig(training_history=MAX_TRAINING_SEASONS, c_value=c_value)
    fitted = fit_logistic(train, config, features=bundle.features)
    probabilities = fitted.predict_proba(valid)
    truth = valid[TARGET].astype(int).to_numpy()

    predictions = pd.DataFrame({
        "nba_game_id": valid["nba_game_id"].to_numpy(),
        "season": validation_season,
        "bundle": bundle.name,
        "home_win": truth,
        "probability": probabilities,
    })
    return FoldOutcome(
        season=validation_season,
        bundle=bundle.name,
        c_value=c_value,
        brier=metrics.brier_score(truth, probabilities),
        log_loss=metrics.log_loss(truth, probabilities),
        auc=metrics.roc_auc(truth, probabilities),
        n_games=len(valid),
        predictions=predictions,
    )


def run_development(
    features: pd.DataFrame,
    observations: list[StatusObservation],
    bundles: tuple[Bundle, ...],
) -> tuple[dict[str, Any], dict[int, pd.DataFrame]]:
    """Every bundle on every development fold, control and enhanced alike."""
    assert_no_holdout(list(DEVELOPMENT_SEASONS), where="development seasons")

    results: list[dict[str, Any]] = []
    calibrations: dict[int, StatusCalibration] = {}
    control_predictions: dict[int, pd.DataFrame] = {}

    for season in DEVELOPMENT_SEASONS:
        training = training_seasons_for(season, AVAILABILITY_SEASONS)
        assert_no_holdout(training, where=f"fold {season}")
        calibration = calibrate(observations, training)
        calibrations[season] = calibration
        fold_features = derive_features(features, calibration)

        for bundle in bundles:
            for c_value in C_GRID:
                outcome = evaluate_bundle(
                    fold_features, bundle, season, training, c_value
                )
                results.append({
                    "season": season,
                    "bundle": bundle.name,
                    "C": c_value,
                    "brier": outcome.brier,
                    "log_loss": outcome.log_loss,
                    "auc": outcome.auc,
                    "n_games": outcome.n_games,
                    "training_seasons": training,
                })
                if bundle.name == "A":
                    control_predictions.setdefault(
                        c_value, {}
                    ).setdefault(season, outcome.predictions)

    frame = pd.DataFrame(results)
    summary = (
        frame.groupby(["bundle", "C"])
        .agg(mean_brier=("brier", "mean"), mean_log_loss=("log_loss", "mean"),
             mean_auc=("auc", "mean"))
        .reset_index()
        .sort_values(["mean_brier", "mean_log_loss"])
    )
    report = {
        "folds": results,
        "summary": summary.to_dict("records"),
        "calibration_by_fold": {
            str(season): cal.to_dict() for season, cal in calibrations.items()
        },
        "ordering_check": {
            str(season): check_ordering(cal, STATUS_RANK)
            for season, cal in calibrations.items()
        },
    }
    return report, calibrations


def select_configuration(
    summary: list[dict[str, Any]], bundles: dict[str, Bundle]
) -> dict[str, Any]:
    """Lowest mean development Brier, tie-broken on log loss then simplicity.

    Differences of this size are within noise, so a tolerance band is applied
    and the *simplest* configuration inside it wins. "Simplest" means the fewest
    availability features added to the control -- not alphabetical order, which
    would be arbitrary -- and a smaller C breaks any remaining tie by preferring
    the more regularised fit.
    """
    ranked = sorted(summary, key=lambda r: (r["mean_brier"], r["mean_log_loss"]))
    best = ranked[0]
    tolerance = 1e-4
    tied = [r for r in ranked if r["mean_brier"] - best["mean_brier"] <= tolerance]

    def added_features(name: str) -> int:
        bundle = bundles.get(name)
        if bundle is None:
            return 0
        return len(set(bundle.features) - set(PHASE_3A3_FEATURES))

    chosen = min(
        tied,
        key=lambda r: (added_features(r["bundle"]), r["mean_brier"], r["C"]),
    )
    return {
        "bundle": chosen["bundle"],
        "C": chosen["C"],
        "mean_brier": chosen["mean_brier"],
        "mean_log_loss": chosen["mean_log_loss"],
        "availability_features_added": added_features(chosen["bundle"]),
        "tied_within_tolerance": [
            {
                "bundle": r["bundle"],
                "C": r["C"],
                "mean_brier": r["mean_brier"],
                "availability_features_added": added_features(r["bundle"]),
            }
            for r in tied
        ],
        "tolerance": tolerance,
        "rule": (
            "lowest mean development Brier; within a 1e-4 band prefer the "
            "fewest availability features, then the lower Brier, then smaller C"
        ),
    }


def family_helped(summary: list[dict[str, Any]], bundle: str) -> bool:
    """Whether a bundle's best configuration beats the control's best."""
    def best(name: str) -> float:
        rows = [r["mean_brier"] for r in summary if r["bundle"] == name]
        return min(rows) if rows else float("inf")

    return best(bundle) < best("A")


def evaluate_predictions(truth: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    return {
        "brier": metrics.brier_score(truth, probability),
        "log_loss": metrics.log_loss(truth, probability),
        "accuracy": metrics.accuracy(truth, probability),
        "auc": metrics.roc_auc(truth, probability),
        "ece": metrics.expected_calibration_error(truth, probability),
        "n_games": len(truth),
    }


def run_holdout(
    features: pd.DataFrame,
    observations: list[StatusObservation],
    bundle: Bundle,
    c_value: float,
    label: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score the frozen configuration on 2025-26. Nothing is selected here."""
    training = training_seasons_for(HOLDOUT_SEASON, AVAILABILITY_SEASONS)
    assert_no_holdout(training, where=f"holdout training ({label})")
    calibration = calibrate(observations, training)
    scoped = derive_features(features, calibration)

    train = scoped[scoped["season"].isin(training)]
    holdout = scoped[scoped["season"] == HOLDOUT_SEASON]
    config = LogisticConfig(training_history=MAX_TRAINING_SEASONS, c_value=c_value)
    fitted = fit_logistic(train, config, features=bundle.features)
    probability = fitted.predict_proba(holdout)
    truth = holdout[TARGET].astype(int).to_numpy()

    predictions = pd.DataFrame({
        "nba_game_id": holdout["nba_game_id"].to_numpy(),
        "season": HOLDOUT_SEASON,
        "home_win": truth,
        f"probability_{label}": probability,
    })
    summary = evaluate_predictions(truth, probability)
    summary["label"] = label
    summary["training_seasons"] = training
    summary["calibration"] = calibration.to_dict()
    return summary, predictions


def report_age_distribution(frame: pd.DataFrame, seasons: tuple[int, ...]) -> dict[str, Any]:
    ages = frame[frame["season"].isin(seasons)]["avail_report_age_minutes"].dropna()
    if ages.empty:
        return {"n": 0}
    return {
        "n": len(ages),
        "median": float(ages.median()),
        "mean": float(ages.mean()),
        "p95": float(ages.quantile(0.95)),
        "max": float(ages.max()),
        "share_zero": float((ages == 0).mean()),
    }


def availability_segments(
    predictions: pd.DataFrame, features: pd.DataFrame, columns: dict[str, str]
) -> dict[str, Any]:
    """Where does availability help? Diagnostics only, never a tuning signal.

    If availability is genuinely informative the gain should concentrate in
    games carrying a real availability burden, not spread evenly. A uniform
    gain would suggest the feature is standing in for something else.
    """
    # Prediction columns live in ``predictions``; the availability columns that
    # define each segment live in ``features``.
    segment_columns = [
        "avail_expected_minutes_lost_diff",
        "avail_home_t30_out_expected_minutes",
        "avail_away_t30_out_expected_minutes",
        "avail_home_t30_questionable_expected_minutes",
        "avail_away_t30_questionable_expected_minutes",
        "avail_home_t3h_to_t30_late_downgrades",
        "avail_away_t3h_to_t30_late_downgrades",
        "avail_home_t3h_to_t30_late_upgrades",
        "avail_away_t3h_to_t30_late_upgrades",
    ]
    present = [c for c in segment_columns if c in features.columns]
    merged = predictions.merge(
        features[["nba_game_id", *present]], on="nba_game_id", how="left"
    )
    truth = merged["home_win"].to_numpy()
    loss_diff = merged["avail_expected_minutes_lost_diff"].abs()
    out_minutes = merged[["avail_home_t30_out_expected_minutes",
                          "avail_away_t30_out_expected_minutes"]].max(axis=1)
    downgrades = merged[["avail_home_t3h_to_t30_late_downgrades",
                         "avail_away_t3h_to_t30_late_downgrades"]].fillna(0).max(axis=1)
    upgrades = merged[["avail_home_t3h_to_t30_late_upgrades",
                       "avail_away_t3h_to_t30_late_upgrades"]].fillna(0).max(axis=1)

    masks = {
        "no_meaningful_burden": loss_diff <= 2.0,
        "high_minute_out_present": out_minutes >= 25.0,
        "questionable_high_minute": merged[
            ["avail_home_t30_questionable_expected_minutes",
             "avail_away_t30_questionable_expected_minutes"]
        ].max(axis=1) >= 20.0,
        "top_quartile_expected_loss": loss_diff >= loss_diff.quantile(0.75),
        "late_downgrade_games": downgrades > 0,
        "late_upgrade_games": upgrades > 0,
        "stable_status_games": (downgrades == 0) & (upgrades == 0),
    }

    out: dict[str, Any] = {}
    for name, mask in masks.items():
        selected = mask.fillna(False).to_numpy()
        if selected.sum() < 20:
            out[name] = {"n_games": int(selected.sum()), "note": "too few to report"}
            continue
        entry: dict[str, Any] = {"n_games": int(selected.sum())}
        for key, column in columns.items():
            if column in merged:
                entry[f"brier_{key}"] = metrics.brier_score(
                    truth[selected], merged[column].to_numpy()[selected]
                )
        if "control" in columns and "enhanced" in columns:
            entry["brier_delta"] = entry["brier_enhanced"] - entry["brier_control"]
        out[name] = entry
    return out


def run_leakage_audits(
    features: pd.DataFrame, observations: list[StatusObservation]
) -> dict[str, Any]:
    """Structural checks that the guarantees hold on the real artefacts."""
    audits: dict[str, Any] = {}

    anchor_columns = [c for c in features.columns if "t15" in c or "t5m" in c]
    audits["no_post_anchor_anchors_present"] = not anchor_columns

    kalshi = [c for c in features.columns if "kalshi" in c.lower()]
    audits["no_kalshi_columns"] = not kalshi

    banned = ("actual_minutes", "played", "participation", "starter")
    leaked = [c for c in ALL_AVAILABILITY_FEATURES
              if any(token in c.lower() for token in banned)]
    audits["no_participation_features_in_bundles"] = not leaked

    ages = features["avail_report_age_minutes"].dropna()
    audits["no_negative_report_age"] = bool((ages >= 0).all())
    audits["exactly_at_anchor_allowed"] = bool((ages == 0).any())

    audits["holdout_absent_from_development_training"] = all(
        HOLDOUT_SEASON not in training_seasons_for(season, AVAILABILITY_SEASONS)
        for season in DEVELOPMENT_SEASONS
    )
    audits["fold_training_excludes_its_own_season"] = all(
        season not in training_seasons_for(season, AVAILABILITY_SEASONS)
        for season in DEVELOPMENT_SEASONS
    )
    for season in DEVELOPMENT_SEASONS:
        cal = calibrate(observations, training_seasons_for(season, AVAILABILITY_SEASONS))
        audits[f"calibration_{season}_excludes_own_season"] = (
            season not in cal.training_seasons
        )
        audits[f"calibration_{season}_excludes_holdout"] = (
            HOLDOUT_SEASON not in cal.training_seasons
        )

    uncovered = features[~features["availability_coverage"].astype(bool)]
    audits["uncovered_games_left_missing"] = bool(
        uncovered["avail_report_timestamp_utc"].isna().all()
    ) if len(uncovered) else True
    audits["uncovered_game_count"] = len(uncovered)
    return audits


def load_features(settings: Settings, suffix: str = "") -> pd.DataFrame:
    processed = settings.paths.processed
    path = processed / f"nba_game_availability_features_2019_26{suffix}.parquet"
    if not path.is_file():
        raise ConfigError(f"Missing {path}. Run build_availability_features first.")
    availability = pd.read_parquet(path)

    model = pd.read_parquet(processed / "nba_model_features_3a3_2006_26.parquet")
    merged = model.merge(
        availability.drop(columns=["season", "home_team", "away_team",
                                   "game_datetime_utc"], errors="ignore"),
        on="nba_game_id", how="inner",
    )
    missing = [c for c in PHASE_3A3_FEATURES if c not in merged.columns]
    if missing:
        raise ConfigError(f"Phase 3A3 features missing after merge: {missing}")
    return merged.sort_values("game_datetime_utc", kind="stable")


def run_pipeline(*, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()

    features = load_features(settings)
    observations = load_observations(settings)

    development, _ = run_development(
        features, observations, AVAILABILITY_BUNDLES
    )
    summary = development["summary"]

    # Bundle G exists only if both of its constituents independently helped.
    simple_candidates = [
        (name, min((r["mean_brier"] for r in summary if r["bundle"] == name),
                   default=float("inf")))
        for name in ("B", "C", "D")
    ]
    best_simple_bundle = min(simple_candidates, key=lambda x: x[1])[0]
    family_for_bundle = {"B": "raw_counts", "C": "role_minutes", "D": "expected_loss"}
    late_news_helped = family_helped(summary, "F")
    simple_helped = family_helped(summary, best_simple_bundle)
    bundle_g = conditional_bundle_g(
        family_for_bundle[best_simple_bundle], late_news_helped and simple_helped
    )
    if bundle_g is not None:
        extra, _ = run_development(features, observations, (bundle_g,))
        development["folds"].extend(extra["folds"])
        summary = summary + extra["summary"]
        development["summary"] = summary

    all_bundles = {b.name: b for b in AVAILABILITY_BUNDLES}
    if bundle_g is not None:
        all_bundles[bundle_g.name] = bundle_g
    selection = select_configuration(summary, all_bundles)
    chosen_bundle = all_bundles[selection["bundle"]]
    control_bundle = next(b for b in AVAILABILITY_BUNDLES if b.name == "A")

    frozen = {
        "bundle": chosen_bundle.name,
        "bundle_description": chosen_bundle.description,
        "bundle_features": list(chosen_bundle.features),
        "availability_features": [
            f for f in chosen_bundle.features if f.startswith("avail_")
        ],
        "C": selection["C"],
        "training_history_seasons": MAX_TRAINING_SEASONS,
        "player_role": {
            "method": "shrunk mean minutes over the last 10 prior games, within season",
            "window": 10,
            "shrinkage_games": 3.0,
            "unknown_role": "contributes nothing rather than zero",
        },
        "status_calibration": {
            "method": "per-fold, training seasons only, shrunk toward the pooled ratio",
            "shrinkage_observations": 50.0,
            "min_observations": 20,
        },
        "late_news": {
            "anchors": ["t3h", "t1h"],
            "missing_earlier_report": "feature unavailable, never treated as no change",
        },
        "preprocessing": "SimpleImputer -> StandardScaler -> LogisticRegression, train-fit only",
        "development_validation_seasons": list(DEVELOPMENT_SEASONS),
        "frozen_at_utc": utc_now().isoformat(),
    }

    # --- holdout, after freezing -------------------------------------------
    native, native_predictions = run_holdout(
        features, observations, chosen_bundle, selection["C"], "3a3c_native"
    )
    control, control_predictions = run_holdout(
        features, observations, control_bundle, selection["C"], "control"
    )

    harmonized: dict[str, Any] | None = None
    harmonized_predictions = None
    harmonized_path = (
        settings.paths.processed
        / "nba_game_availability_features_2019_26_harmonized.parquet"
    )
    if harmonized_path.is_file():
        harmonized_features = load_features(settings, "_harmonized")
        harmonized_observations = load_observations(settings, "_harmonized")
        harmonized, harmonized_predictions = run_holdout(
            harmonized_features, harmonized_observations,
            chosen_bundle, selection["C"], "3a3c_harmonized",
        )

    predictions = native_predictions.merge(
        control_predictions.drop(columns=["season", "home_win"]),
        on="nba_game_id", how="left",
    )
    if harmonized_predictions is not None:
        predictions = predictions.merge(
            harmonized_predictions.drop(columns=["season", "home_win"]),
            on="nba_game_id", how="left",
        )

    # Reference models, carried over unchanged from earlier phases.
    references: dict[str, Any] = {}
    reference_columns = {
        "phase_3a3": "phase_3a3_logistic_probability",
        "phase_3a2": "phase_3a2_logistic_probability",
        "mov_elo": "mov_elo_probability",
        "kalshi_t30_normalized": "kalshi_home_probability_normalized",
    }
    prior_path = settings.paths.processed / "nba_predictions_3a3_2025_26.parquet"
    if prior_path.is_file():
        prior = pd.read_parquet(prior_path)
        keep = ["nba_game_id", *[c for c in reference_columns.values()
                                 if c in prior.columns]]
        predictions = predictions.merge(prior[keep], on="nba_game_id", how="left")

    truth = predictions["home_win"].to_numpy()
    for name, column in reference_columns.items():
        if column in predictions and predictions[column].notna().all():
            references[name] = evaluate_predictions(
                truth, predictions[column].to_numpy()
            )

    # Paired bootstrap. The helper takes per-game *losses*, and its sign
    # convention is model-minus-benchmark, so negative means 3A3C is better.
    def losses(column: str, kind: str) -> np.ndarray | None:
        if column not in predictions or not predictions[column].notna().all():
            return None
        probability = predictions[column].to_numpy()
        return (metrics.brier_losses(truth, probability) if kind == "brier"
                else metrics.log_losses(truth, probability))

    comparisons: dict[str, Any] = {}
    candidates = [("native", "probability_3a3c_native")]
    if harmonized_predictions is not None:
        candidates.append(("harmonized", "probability_3a3c_harmonized"))
    benchmarks = {
        "phase_3a3": reference_columns["phase_3a3"],
        "kalshi": reference_columns["kalshi_t30_normalized"],
        "matched_control": "probability_control",
    }
    for model_name, model_column in candidates:
        for benchmark_name, benchmark_column in benchmarks.items():
            for kind in ("brier", "log_loss"):
                model_losses = losses(model_column, kind)
                benchmark_losses = losses(benchmark_column, kind)
                if model_losses is None or benchmark_losses is None:
                    continue
                key = f"3a3c_{model_name}_minus_{benchmark_name}_{kind}"
                comparisons[key] = metrics.paired_bootstrap(
                    model_losses, benchmark_losses
                )

    report = {
        "generated_at_utc": utc_now().isoformat(),
        "availability_seasons": list(AVAILABILITY_SEASONS),
        "development_seasons": list(DEVELOPMENT_SEASONS),
        "holdout_season": HOLDOUT_SEASON,
        "training_history_policy": {
            "rule": "all complete prior availability-enabled seasons, capped at 5",
            "by_fold": {
                str(s): training_seasons_for(s, AVAILABILITY_SEASONS)
                for s in (*DEVELOPMENT_SEASONS, HOLDOUT_SEASON)
            },
        },
        "development": development,
        "selection": selection,
        "bundle_g_built": bundle_g is not None,
        "bundle_g_precondition": {
            "best_simple_bundle": best_simple_bundle,
            "simple_family_helped": simple_helped,
            "late_news_helped": late_news_helped,
        },
        "frozen_configuration": frozen,
        "holdout": {
            "native": native,
            "control": control,
            "harmonized": harmonized,
            "references": references,
        },
        "paired_bootstrap": comparisons,
        "report_age": {
            "development": report_age_distribution(features, DEVELOPMENT_SEASONS),
            "holdout_native": report_age_distribution(features, (HOLDOUT_SEASON,)),
        },
        "leakage_audits": run_leakage_audits(features, observations),
    }
    if harmonized_path.is_file():
        report["report_age"]["holdout_harmonized"] = report_age_distribution(
            load_features(settings, "_harmonized"), (HOLDOUT_SEASON,)
        )

    calibration_for_segments = calibrate(
        observations, training_seasons_for(HOLDOUT_SEASON, AVAILABILITY_SEASONS)
    )
    scoped = derive_features(features, calibration_for_segments)
    report["segments"] = availability_segments(
        predictions, scoped,
        {"control": "probability_control", "enhanced": "probability_3a3c_native"},
    )

    out_path = settings.paths.processed / "nba_predictions_3a3c_2025_26.parquet"
    predictions.to_parquet(out_path, index=False)
    report_path = settings.paths.reports / "model_availability_2025_26.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"] = [str(out_path), str(report_path)]
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Phase 3A3C availability model.")


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    try:
        report = run_pipeline()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"\n{'bundle':7s} {'C':>6s} {'mean Brier':>11s} {'mean logloss':>13s} {'mean AUC':>9s}")
    for row in sorted(report["development"]["summary"], key=lambda r: r["mean_brier"])[:10]:
        print(f"{row['bundle']:7s} {row['C']:6.1f} {row['mean_brier']:11.5f} "
              f"{row['mean_log_loss']:13.5f} {row['mean_auc']:9.4f}")
    sel = report["selection"]
    print(f"\nselected: bundle {sel['bundle']} C={sel['C']}")
    print("\nholdout 2025-26:")
    for name, block in report["holdout"].items():
        if isinstance(block, dict) and "brier" in block:
            print(f"  {name:14s} Brier={block['brier']:.5f} logloss={block['log_loss']:.5f} "
                  f"AUC={block['auc']:.4f} ECE={block['ece']:.4f}")
    for path in report["written_files"]:
        print(f"\nWrote {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
