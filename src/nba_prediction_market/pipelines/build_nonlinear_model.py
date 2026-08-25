"""Phase 3A4: is the remaining gap a model-class problem or an information one?

Phase 3A3C closed about a third of the Brier gap to Kalshi by adding genuine
T-30 availability. This phase holds that information fixed and changes only the
*model class*, then -- separately -- asks whether families rejected under linear
regression become useful once interactions are available.

The design is built so that a negative result is a real answer rather than a
prompt to search harder. The grid is compact and fixed, depth is capped at 3,
and no family is added after a result is seen. If gradient boosting does not
beat the frozen logistic model on development folds, the honest conclusion is
that what remains is information the model does not have, not structure it
cannot express.

Phase 3A3C is reproduced here as the control and must match its published
numbers before anything else is trusted.
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
from nba_prediction_market.models.bundles import Bundle
from nba_prediction_market.models.logistic import LogisticConfig, fit_logistic
from nba_prediction_market.models.nonlinear import (
    ALL_CONFIGS,
    CANDIDATE_RANGES,
    CONFIGS_BY_NAME,
    HISTGB_GRID,
    RANDOM_SEED,
    XGBOOST_GRID,
    NonlinearConfig,
    build_estimator,
)
from nba_prediction_market.models.nonlinear_bundles import (
    ALL_ALLOWED_FEATURES,
    CORE,
    EXTENDED,
    EXTENDED_ADDITIONS,
    FEATURE_SETS,
    FORBIDDEN_TOKENS,
)
from nba_prediction_market.models.probability_calibration import (
    METHODS,
    Calibrator,
    chronological_oof_predictions,
    fit_calibrator,
)
from nba_prediction_market.models.selection import assert_no_holdout
from nba_prediction_market.models.status_calibration import calibrate
from nba_prediction_market.pipelines.build_availability_model import (
    AVAILABILITY_SEASONS,
    DEVELOPMENT_SEASONS,
    HOLDOUT_SEASON,
    MAX_TRAINING_SEASONS,
    TARGET,
    derive_features,
    evaluate_predictions,
    load_features,
    load_observations,
    training_seasons_for,
)

logger = logging.getLogger(__name__)

#: The frozen Phase 3A3C logistic specification, reproduced as the control.
CONTROL_C: float = 0.1
#: Blend weights on the logistic side. Fixed before any result is seen.
BLEND_WEIGHTS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
#: A development improvement smaller than this is treated as noise, and the
#: simpler model is kept.
MATERIAL_BRIER_GAIN: float = 5e-4


def _matrix(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise ConfigError(f"feature frame is missing: {missing}")
    return frame.loc[:, list(features)].astype(float)


def fit_predict_nonlinear(
    config: NonlinearConfig,
    features: tuple[str, ...],
    train: pd.DataFrame,
    predict: pd.DataFrame,
) -> np.ndarray:
    """Fit on ``train`` only and score ``predict``. No shared state."""
    estimator = build_estimator(config)
    estimator.fit(_matrix(train, features), train[TARGET].astype(int).to_numpy())
    return estimator.predict_proba(_matrix(predict, features))[:, 1]


def fit_predict_logistic(
    features: tuple[str, ...], train: pd.DataFrame, predict: pd.DataFrame
) -> np.ndarray:
    config = LogisticConfig(training_history=MAX_TRAINING_SEASONS, c_value=CONTROL_C)
    return fit_logistic(train, config, features=features).predict_proba(predict)


@dataclass
class FoldPredictions:
    """Every model's probabilities on one validation season."""

    season: int
    truth: np.ndarray
    game_ids: np.ndarray
    columns: dict[str, np.ndarray]


def run_fold(
    features_frame: pd.DataFrame,
    season: int,
    training: list[int],
    configs: tuple[NonlinearConfig, ...],
) -> FoldPredictions:
    """Control, every nonlinear config, and each calibration, for one fold."""
    assert_no_holdout(training, where=f"fold {season}")
    train = features_frame[features_frame["season"].isin(training)]
    valid = features_frame[features_frame["season"] == season]
    if train.empty or valid.empty:
        raise ConfigError(f"empty split for validation season {season}")

    columns: dict[str, np.ndarray] = {
        "control_logistic": fit_predict_logistic(CORE.features, train, valid)
    }

    for name, bundle in FEATURE_SETS.items():
        for config in configs:
            raw = fit_predict_nonlinear(config, bundle.features, train, valid)
            columns[f"{config.name}_{name}_none"] = raw

            # Calibrators are fitted on out-of-fold predictions generated
            # entirely inside the training history; this season never sees
            # itself.
            oof = chronological_oof_predictions(
                train, training,
                lambda tr, pr, c=config, b=bundle: fit_predict_nonlinear(
                    c, b.features, tr, pr
                ),
            )
            for method in METHODS:
                if method == "none":
                    continue
                calibrator = fit_calibrator(
                    method, oof["probability"], oof[TARGET]
                ) if len(oof) else Calibrator("none", None, 0, "no out-of-fold rows")
                columns[f"{config.name}_{name}_{method}"] = calibrator.transform(raw)

    return FoldPredictions(
        season=season,
        truth=valid[TARGET].astype(int).to_numpy(),
        game_ids=valid["nba_game_id"].to_numpy(),
        columns=columns,
    )


def score_folds(folds: list[FoldPredictions]) -> pd.DataFrame:
    """Per-season and mean metrics for every candidate column."""
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for name, probability in fold.columns.items():
            rows.append({
                "season": fold.season,
                "candidate": name,
                "brier": metrics.brier_score(fold.truth, probability),
                "log_loss": metrics.log_loss(fold.truth, probability),
                "auc": metrics.roc_auc(fold.truth, probability),
                "ece": metrics.expected_calibration_error(fold.truth, probability),
            })
    return pd.DataFrame(rows)


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby("candidate")
        .agg(mean_brier=("brier", "mean"), mean_log_loss=("log_loss", "mean"),
             mean_auc=("auc", "mean"), mean_ece=("ece", "mean"),
             folds=("season", "nunique"))
        .reset_index()
        .sort_values(["mean_brier", "mean_log_loss"])
    )


def blend_scores(folds: list[FoldPredictions], nonlinear: str) -> pd.DataFrame:
    """Fixed-weight blends of the control logistic and one nonlinear column."""
    rows: list[dict[str, Any]] = []
    for weight in BLEND_WEIGHTS:
        for fold in folds:
            blended = (
                weight * fold.columns["control_logistic"]
                + (1.0 - weight) * fold.columns[nonlinear]
            )
            rows.append({
                "logistic_weight": weight,
                "season": fold.season,
                "brier": metrics.brier_score(fold.truth, blended),
                "log_loss": metrics.log_loss(fold.truth, blended),
                "auc": metrics.roc_auc(fold.truth, blended),
            })
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("logistic_weight")
        .agg(mean_brier=("brier", "mean"), mean_log_loss=("log_loss", "mean"),
             mean_auc=("auc", "mean"))
        .reset_index()
        .sort_values(["mean_brier", "mean_log_loss"])
    )


def feature_importance(
    config: NonlinearConfig,
    features: tuple[str, ...],
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> dict[str, Any]:
    """Gain and permutation importance, from development data only.

    Importance is descriptive, not causal: these models are fitted on features
    that correlate with each other, so a family absorbing another's credit is
    expected. Permutation importance is reported alongside gain because gain
    alone rewards features that are merely split on often.
    """
    from sklearn.inspection import permutation_importance

    estimator = build_estimator(config)
    x_train = _matrix(train, features)
    estimator.fit(x_train, train[TARGET].astype(int).to_numpy())

    gain: dict[str, float] = {}
    if config.family == "xgboost":
        booster = estimator.get_booster()
        raw = booster.get_score(importance_type="gain")
        total = sum(raw.values()) or 1.0
        for index, name in enumerate(features):
            gain[name] = raw.get(f"f{index}", raw.get(name, 0.0)) / total

    x_valid = _matrix(valid, features)
    permuted = permutation_importance(
        estimator, x_valid, valid[TARGET].astype(int).to_numpy(),
        n_repeats=10, random_state=RANDOM_SEED, scoring="neg_brier_score",
    )
    permutation = {
        name: float(value)
        for name, value in zip(features, permuted.importances_mean, strict=True)
    }
    ranked = sorted(permutation.items(), key=lambda x: -x[1])
    return {
        "gain_top": sorted(gain.items(), key=lambda x: -x[1])[:15] if gain else [],
        "permutation_top": ranked[:15],
        "availability_features": {
            name: value for name, value in permutation.items()
            if name.startswith("avail_")
        },
        "note": (
            "descriptive only; correlated features share credit and importance "
            "is not causal"
        ),
    }


def run_leakage_audits(features_frame: pd.DataFrame) -> dict[str, Any]:
    """Structural checks on the Phase 3A4 surface.

    ``features_frame`` must be a *derived* fold frame -- the one a model is
    actually handed. The pre-derivation frame lacks the fold-specific
    availability columns, so auditing it would report absences that never
    reach a model.
    """
    audits: dict[str, Any] = {}
    allowed = sorted(ALL_ALLOWED_FEATURES)
    audits["allowlist_is_explicit"] = len(allowed) == len(EXTENDED.features)
    audits["core_is_subset_of_extended"] = set(CORE.features) <= set(EXTENDED.features)
    audits["no_forbidden_token_in_allowlist"] = not [
        f for f in allowed
        if any(token in f.lower() for token in FORBIDDEN_TOKENS)
    ]
    audits["target_not_in_allowlist"] = TARGET not in ALL_ALLOWED_FEATURES
    audits["holdout_absent_from_training"] = all(
        HOLDOUT_SEASON not in training_seasons_for(s, AVAILABILITY_SEASONS)
        for s in (*DEVELOPMENT_SEASONS, HOLDOUT_SEASON)
    )
    audits["fold_excludes_own_season"] = all(
        s not in training_seasons_for(s, AVAILABILITY_SEASONS)
        for s in DEVELOPMENT_SEASONS
    )
    absent = [f for f in allowed if f not in features_frame.columns]
    audits["every_allowed_feature_present"] = not absent
    if absent:
        audits["absent_allowed_features"] = absent
    ages = features_frame["avail_report_age_minutes"].dropna()
    audits["no_post_anchor_report_used"] = bool((ages >= 0).all())
    audits["exactly_at_anchor_allowed"] = bool((ages == 0).any())
    return audits


def calibration_is_leakage_safe() -> dict[str, Any]:
    """Demonstrate the calibrator never sees the season it is applied to."""
    frame = pd.DataFrame({
        "season": [2019] * 40 + [2020] * 40 + [2021] * 40,
        "home_win": ([0, 1] * 20) * 3,
        "x": list(range(120)),
    })
    seen: list[list[int]] = []

    def fit_predict(train: pd.DataFrame, predict: pd.DataFrame) -> np.ndarray:
        seen.append(sorted(set(train["season"]) & set(predict["season"])))
        return np.full(len(predict), 0.5)

    oof = chronological_oof_predictions(frame, [2019, 2020, 2021], fit_predict)
    return {
        "inner_folds_share_no_season": all(not overlap for overlap in seen),
        "earliest_training_season_not_scored": 2019 not in set(oof["season"]),
        "oof_seasons": sorted(set(oof["season"].tolist())),
    }


def fold_frames(
    settings: Settings, suffix: str = ""
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """The raw merged frame plus one calibrated-feature frame per fold.

    Each fold's availability features use that fold's own status calibration,
    exactly as Phase 3A3C built them, so the control reproduces bit for bit.
    """
    raw = load_features(settings, suffix)
    observations = load_observations(settings, suffix)
    per_fold: dict[int, pd.DataFrame] = {}
    for season in (*DEVELOPMENT_SEASONS, HOLDOUT_SEASON):
        training = training_seasons_for(season, AVAILABILITY_SEASONS)
        per_fold[season] = derive_features(raw, calibrate(observations, training))
    return raw, per_fold


def select_candidate(summary: pd.DataFrame, control_brier: float) -> dict[str, Any]:
    """Best candidate, but only if it beats the control by a material margin."""
    ranked = summary.sort_values(["mean_brier", "mean_log_loss"])
    best = ranked.iloc[0]
    gain = control_brier - float(best["mean_brier"])
    return {
        "best_candidate": str(best["candidate"]),
        "best_mean_brier": float(best["mean_brier"]),
        "control_mean_brier": control_brier,
        "gain_vs_control": gain,
        "material": bool(gain >= MATERIAL_BRIER_GAIN),
        "materiality_threshold": MATERIAL_BRIER_GAIN,
    }


def run_development_stage(
    per_fold: dict[int, pd.DataFrame],
) -> tuple[dict[str, Any], list[FoldPredictions]]:
    """Everything decided before the holdout is touched."""
    assert_no_holdout(list(DEVELOPMENT_SEASONS), where="phase 3a4 development")

    # --- development ----------------------------------------------------
    folds: list[FoldPredictions] = []
    for season in DEVELOPMENT_SEASONS:
        training = training_seasons_for(season, AVAILABILITY_SEASONS)
        logger.info("fold %s: training %s", season, training)
        folds.append(run_fold(per_fold[season], season, training, ALL_CONFIGS))

    scores = score_folds(folds)
    summary = summarise(scores)
    control_row = summary[summary["candidate"] == "control_logistic"].iloc[0]
    control_brier = float(control_row["mean_brier"])

    nonlinear_summary = summary[summary["candidate"] != "control_logistic"]
    selection = select_candidate(nonlinear_summary, control_brier)

    # --- blends, on development only ------------------------------------
    best_nonlinear = selection["best_candidate"]
    blends = blend_scores(folds, best_nonlinear)
    best_blend = blends.iloc[0]
    blend_gain = control_brier - float(best_blend["mean_brier"])

    # --- freeze ---------------------------------------------------------
    use_nonlinear = selection["material"]
    use_blend = bool(
        blend_gain >= MATERIAL_BRIER_GAIN
        and float(best_blend["logistic_weight"]) not in (0.0, 1.0)
    )
    if use_blend:
        frozen_kind = "blend"
    elif use_nonlinear:
        frozen_kind = "nonlinear"
    else:
        frozen_kind = "logistic_control"

    config_name, feature_set_name, calibration_method = (
        best_nonlinear.split("_", 2) if "_" in best_nonlinear else (best_nonlinear, "CORE", "none")
    )
    frozen = {
        "model": frozen_kind,
        "reason": (
            "no nonlinear candidate beat the frozen Phase 3A3C logistic by the "
            f"materiality threshold of {MATERIAL_BRIER_GAIN}"
            if frozen_kind == "logistic_control"
            else "selected on mean development Brier"
        ),
        "nonlinear_config": config_name,
        "feature_set": feature_set_name,
        "calibration": calibration_method,
        "logistic_blend_weight": (
            float(best_blend["logistic_weight"]) if use_blend else
            (1.0 if frozen_kind == "logistic_control" else 0.0)
        ),
        "control_C": CONTROL_C,
        "training_history_seasons": MAX_TRAINING_SEASONS,
        "random_seed": RANDOM_SEED,
        "development_validation_seasons": list(DEVELOPMENT_SEASONS),
        "frozen_at_utc": utc_now().isoformat(),
    }

    report: dict[str, Any] = {
        "generated_at_utc": utc_now().isoformat(),
        "question": (
            "is the remaining gap to Kalshi a model-class problem or an "
            "information problem?"
        ),
        "search_space": {
            "candidate_ranges": {k: list(v) for k, v in CANDIDATE_RANGES.items()},
            "xgboost_configs": [c.to_dict() for c in XGBOOST_GRID],
            "histgb_configs": [c.to_dict() for c in HISTGB_GRID],
            "configs_evaluated": len(ALL_CONFIGS),
            "note": (
                "a hand-picked compact grid, not the Cartesian product of the "
                "ranges, which would be 128 fits per fold per feature set"
            ),
        },
        "feature_sets": {
            "CORE": {"n": len(CORE.features), "features": list(CORE.features)},
            "EXTENDED": {
                "n": len(EXTENDED.features),
                "added_families": {
                    k: list(v) for k, v in EXTENDED_ADDITIONS.items()
                },
            },
        },
        "development": {
            "per_season": scores.to_dict("records"),
            "summary": summary.to_dict("records"),
            "control_mean_brier": control_brier,
        },
        "selection": selection,
        "blends": blends.to_dict("records"),
        "blend_gain_vs_control": blend_gain,
        "frozen_configuration": frozen,
        "leakage_audits": {
            # Audit the derived frame: that is what a model is handed.
            **run_leakage_audits(per_fold[DEVELOPMENT_SEASONS[-1]]),
            **calibration_is_leakage_safe(),
        },
    }
    return report, folds


def build_frozen_predictions(
    frozen: dict[str, Any],
    frame: pd.DataFrame,
    training: list[int],
    target_season: int,
) -> dict[str, np.ndarray]:
    """Score the frozen configuration, plus its components, on one season."""
    train = frame[frame["season"].isin(training)]
    target = frame[frame["season"] == target_season]

    logistic = fit_predict_logistic(CORE.features, train, target)
    out: dict[str, np.ndarray] = {"phase_3a3c_logistic": logistic}

    config_name = frozen["nonlinear_config"]
    if config_name in CONFIGS_BY_NAME:
        config = CONFIGS_BY_NAME[config_name]
        bundle: Bundle = FEATURE_SETS[frozen["feature_set"]]
        raw = fit_predict_nonlinear(config, bundle.features, train, target)
        oof = chronological_oof_predictions(
            train, training,
            lambda tr, pr: fit_predict_nonlinear(config, bundle.features, tr, pr),
        )
        method = frozen["calibration"]
        calibrator = (
            fit_calibrator(method, oof["probability"], oof[TARGET])
            if method != "none" and len(oof)
            else Calibrator("none", None, len(oof), "raw kept")
        )
        out["nonlinear_raw"] = raw
        out["nonlinear_frozen"] = calibrator.transform(raw)
        out["_calibrator"] = calibrator

    weight = float(frozen["logistic_blend_weight"])
    nonlinear = out.get("nonlinear_frozen")
    out["phase_3a4"] = (
        logistic if nonlinear is None
        else weight * logistic + (1.0 - weight) * nonlinear
    )
    return out


def run_pipeline(*, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()

    _, per_fold = fold_frames(settings)
    report, _ = run_development_stage(per_fold)
    frozen = report["frozen_configuration"]

    # --- control reproduction ------------------------------------------
    holdout_training = training_seasons_for(HOLDOUT_SEASON, AVAILABILITY_SEASONS)
    holdout_frame = per_fold[HOLDOUT_SEASON]
    scored = build_frozen_predictions(
        frozen, holdout_frame, holdout_training, HOLDOUT_SEASON
    )
    calibrator = scored.pop("_calibrator", None)

    target = holdout_frame[holdout_frame["season"] == HOLDOUT_SEASON]
    truth = target[TARGET].astype(int).to_numpy()

    published = settings.paths.processed / "nba_predictions_3a3c_2025_26.parquet"
    reproduction: dict[str, Any] = {"checked": False}
    if published.is_file():
        prior = pd.read_parquet(published)
        merged = pd.DataFrame({
            "nba_game_id": target["nba_game_id"].to_numpy(),
            "reproduced": scored["phase_3a3c_logistic"],
        }).merge(
            prior[["nba_game_id", "probability_3a3c_native"]],
            on="nba_game_id", how="inner",
        )
        difference = np.abs(
            merged["reproduced"].to_numpy()
            - merged["probability_3a3c_native"].to_numpy()
        )
        reproduction = {
            "checked": True,
            "n_games": len(merged),
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
            "matches_within_tolerance": bool(difference.max() < 1e-9),
        }

    # --- holdout metrics ------------------------------------------------
    holdout: dict[str, Any] = {}
    for name in ("phase_3a4", "phase_3a3c_logistic", "nonlinear_raw",
                 "nonlinear_frozen"):
        if name in scored:
            holdout[name] = evaluate_predictions(truth, scored[name])

    predictions = pd.DataFrame({
        "nba_game_id": target["nba_game_id"].to_numpy(),
        "season": HOLDOUT_SEASON,
        "home_win": truth,
        **{f"probability_{k}": v for k, v in scored.items()},
    })

    reference_columns = {
        "phase_3a3": "phase_3a3_logistic_probability",
        "mov_elo": "mov_elo_probability",
        "kalshi_t30_normalized": "kalshi_home_probability_normalized",
    }
    prior_path = settings.paths.processed / "nba_predictions_3a3_2025_26.parquet"
    if prior_path.is_file():
        prior = pd.read_parquet(prior_path)
        keep = ["nba_game_id", *[c for c in reference_columns.values()
                                 if c in prior.columns]]
        predictions = predictions.merge(prior[keep], on="nba_game_id", how="left")
    references = {
        name: evaluate_predictions(truth, predictions[column].to_numpy())
        for name, column in reference_columns.items()
        if column in predictions and predictions[column].notna().all()
    }

    # --- cadence sensitivity, no retuning -------------------------------
    harmonized_path = (
        settings.paths.processed
        / "nba_game_availability_features_2019_26_harmonized.parquet"
    )
    harmonized: dict[str, Any] | None = None
    if harmonized_path.is_file():
        _, harmonized_folds = fold_frames(settings, "_harmonized")
        harmonized_scored = build_frozen_predictions(
            frozen, harmonized_folds[HOLDOUT_SEASON], holdout_training, HOLDOUT_SEASON
        )
        harmonized_scored.pop("_calibrator", None)
        harmonized = {
            "phase_3a4": evaluate_predictions(truth, harmonized_scored["phase_3a4"]),
            "note": "frozen configuration reapplied; nothing retuned",
        }
        predictions["probability_phase_3a4_harmonized"] = harmonized_scored["phase_3a4"]

    # --- paired bootstrap ------------------------------------------------
    def losses(column: str, kind: str) -> np.ndarray | None:
        if column not in predictions or not predictions[column].notna().all():
            return None
        probability = predictions[column].to_numpy()
        return (metrics.brier_losses(truth, probability) if kind == "brier"
                else metrics.log_losses(truth, probability))

    comparisons: dict[str, Any] = {}
    models = [("phase_3a4", "probability_phase_3a4")]
    if harmonized is not None:
        models.append(("phase_3a4_harmonized", "probability_phase_3a4_harmonized"))
    benchmarks = {
        "phase_3a3c": "probability_phase_3a3c_logistic",
        "kalshi": reference_columns["kalshi_t30_normalized"],
    }
    for model_name, model_column in models:
        for benchmark_name, benchmark_column in benchmarks.items():
            for kind in ("brier", "log_loss"):
                a, b = losses(model_column, kind), losses(benchmark_column, kind)
                if a is None or b is None:
                    continue
                comparisons[f"{model_name}_minus_{benchmark_name}_{kind}"] = (
                    metrics.paired_bootstrap(a, b)
                )

    report.update({
        "control_reproduction": reproduction,
        "calibrator_used": calibrator.to_dict() if calibrator else None,
        "holdout": holdout,
        "holdout_references": references,
        "cadence_sensitivity": harmonized,
        "paired_bootstrap": comparisons,
        "segments": holdout_segments(predictions, holdout_frame, truth),
    })

    # Importance, from development data only.
    if frozen["nonlinear_config"] in CONFIGS_BY_NAME:
        last = DEVELOPMENT_SEASONS[-1]
        frame = per_fold[last]
        training = training_seasons_for(last, AVAILABILITY_SEASONS)
        report["feature_importance"] = feature_importance(
            CONFIGS_BY_NAME[frozen["nonlinear_config"]],
            FEATURE_SETS[frozen["feature_set"]].features,
            frame[frame["season"].isin(training)],
            frame[frame["season"] == last],
        )

    out_path = settings.paths.processed / "nba_predictions_3a4_2025_26.parquet"
    predictions.to_parquet(out_path, index=False)
    report_path = settings.paths.reports / "model_nonlinear_2025_26.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"] = [str(out_path), str(report_path)]
    return report


def holdout_segments(
    predictions: pd.DataFrame, features: pd.DataFrame, truth: np.ndarray
) -> dict[str, Any]:
    """Where the frozen model differs from the control. Diagnostics only.

    These cannot alter selection -- the configuration was frozen before the
    holdout was scored -- and exist to check that any gain lands where the
    information should matter rather than uniformly.
    """
    wanted = [
        "avail_expected_minutes_lost_diff",
        "avail_home_t30_out_expected_minutes",
        "avail_away_t30_out_expected_minutes",
        "avail_home_t30_questionable_expected_minutes",
        "avail_away_t30_questionable_expected_minutes",
        "game_datetime_utc",
    ]
    present = [c for c in wanted if c in features.columns]
    merged = predictions.merge(
        features[["nba_game_id", *present]].drop_duplicates("nba_game_id"),
        on="nba_game_id", how="left",
    )

    control = merged["probability_phase_3a3c_logistic"].to_numpy()
    model = merged["probability_phase_3a4"].to_numpy()
    # When the frozen model is the control, its deltas are all zero by
    # construction. The standalone nonlinear column is reported alongside so
    # the diagnostics still say *where* boosting differed, which is the
    # informative part of a negative result.
    nonlinear = (
        merged["probability_nonlinear_frozen"].to_numpy()
        if "probability_nonlinear_frozen" in merged else None
    )
    burden = merged["avail_expected_minutes_lost_diff"].abs()
    out_minutes = merged[["avail_home_t30_out_expected_minutes",
                          "avail_away_t30_out_expected_minutes"]].max(axis=1)
    questionable = merged[["avail_home_t30_questionable_expected_minutes",
                           "avail_away_t30_questionable_expected_minutes"]].max(axis=1)
    order = merged["game_datetime_utc"].rank(method="first")

    masks = {
        "high_availability_burden": burden >= burden.quantile(0.75),
        "low_availability_burden": burden <= burden.quantile(0.25),
        "high_minute_out_present": out_minutes >= 25.0,
        "high_minute_questionable": questionable >= 20.0,
        "both_sides_designated": (
            (merged["avail_home_t30_out_expected_minutes"] >= 15.0)
            & (merged["avail_away_t30_out_expected_minutes"] >= 15.0)
        ),
        "favourites": pd.Series(control >= 0.6, index=merged.index),
        "underdogs": pd.Series(control <= 0.4, index=merged.index),
        "close_games": pd.Series(
            (control > 0.4) & (control < 0.6), index=merged.index
        ),
        "early_season": order <= len(merged) / 2,
        "late_season": order > len(merged) / 2,
    }

    out: dict[str, Any] = {}
    for name, mask in masks.items():
        selected = mask.fillna(False).to_numpy()
        if selected.sum() < 20:
            out[name] = {"n_games": int(selected.sum()), "note": "too few to report"}
            continue
        control_brier = metrics.brier_score(truth[selected], control[selected])
        model_brier = metrics.brier_score(truth[selected], model[selected])
        entry = {
            "n_games": int(selected.sum()),
            "brier_phase_3a3c": control_brier,
            "brier_phase_3a4": model_brier,
            "brier_delta": model_brier - control_brier,
        }
        if nonlinear is not None:
            nonlinear_brier = metrics.brier_score(
                truth[selected], nonlinear[selected]
            )
            entry["brier_nonlinear"] = nonlinear_brier
            entry["nonlinear_minus_3a3c"] = nonlinear_brier - control_brier
        out[name] = entry
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Phase 3A4 nonlinear model.")


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    try:
        report = run_pipeline()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"\n{'candidate':32s} {'Brier':>9s} {'logloss':>9s} {'AUC':>7s} {'ECE':>7s}")
    for row in report["development"]["summary"][:12]:
        print(f"{row['candidate']:32s} {row['mean_brier']:9.5f} "
              f"{row['mean_log_loss']:9.5f} {row['mean_auc']:7.4f} {row['mean_ece']:7.4f}")
    sel = report["selection"]
    print(f"\ncontrol mean Brier : {sel['control_mean_brier']:.5f}")
    print(f"best nonlinear     : {sel['best_candidate']} {sel['best_mean_brier']:.5f} "
          f"(gain {sel['gain_vs_control']:+.5f}, material={sel['material']})")
    print(f"frozen model       : {report['frozen_configuration']['model']}")
    rep = report["control_reproduction"]
    if rep.get("checked"):
        print(f"3A3C reproduction  : max|diff|={rep['max_abs_difference']:.2e} "
              f"match={rep['matches_within_tolerance']}")
    print("\nholdout 2025-26:")
    for name, block in report["holdout"].items():
        print(f"  {name:22s} Brier={block['brier']:.5f} LL={block['log_loss']:.5f} "
              f"AUC={block['auc']:.4f} ECE={block['ece']:.4f}")
    for name, block in report["holdout_references"].items():
        print(f"  {name:22s} Brier={block['brier']:.5f} LL={block['log_loss']:.5f} "
              f"AUC={block['auc']:.4f} ECE={block['ece']:.4f}")
    for path in report["written_files"]:
        print(f"\nWrote {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
