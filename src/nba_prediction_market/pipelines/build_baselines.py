"""Phase 3A1: forecasting baselines and history-window selection.

Three stages, deliberately separated so holdout data cannot influence a modelling
choice:

1. :func:`stage_features` builds lookahead-safe sequential features for every
   modelling-eligible regular-season game.
2. :func:`stage_select` runs development validation on seasons before the
   holdout and returns a :class:`FrozenConfig`. It never sees season 2025.
3. :func:`stage_holdout` consumes a *frozen* config and evaluates 2025-26 once.

Kalshi is a benchmark only and never enters a model matrix.

Run with::

    python -m nba_prediction_market.pipelines.build_baselines
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.features.elo import EloConfig, run_elo
from nba_prediction_market.features.feature_engine import FEATURE_COLUMNS, build_features
from nba_prediction_market.features.feature_spec import MODEL_FEATURES, TARGET
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.models import metrics as M
from nba_prediction_market.models.logistic import (
    LogisticConfig,
    fit_logistic,
    training_seasons,
)
from nba_prediction_market.models.selection import (
    DEVELOPMENT_VALIDATION_SEASONS,
    HOLDOUT_SEASON,
    CandidateResult,
    assert_no_holdout,
    select_elo,
    select_logistic,
    tied_with_best,
)

logger = logging.getLogger(__name__)

ELO_FEATURE_COLUMNS = ("home_elo", "away_elo", "elo_diff", "elo_probability")

PREDICTION_COLUMNS: list[str] = [
    "nba_game_id",
    "season",
    "game_datetime_utc",
    "home_team",
    "away_team",
    "home_win",
    *[c for c in MODEL_FEATURES],
    "constant_probability",
    "elo_probability",
    "logistic_probability",
    "kalshi_home_midpoint",
    "kalshi_away_midpoint",
    "kalshi_home_probability_normalized",
]


@dataclass
class FrozenConfig:
    """Every modelling choice, fixed before the holdout is read."""

    elo: dict[str, Any]
    logistic: dict[str, Any]
    feature_allowlist: list[str]
    constant_baseline_probability: float
    constant_baseline_seasons: list[int]
    logistic_training_seasons: list[int]
    development_validation_seasons: list[int]
    frozen_at_utc: str = field(default_factory=lambda: utc_now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- stage 1: features -----------------------------------------------------


def stage_features(settings: Settings) -> pd.DataFrame:
    """Build sequential features for every modelling-eligible regular-season game."""
    source = settings.paths.processed / "nba_regular_season_games_2006_26.parquet"
    if not source.is_file():
        raise ConfigError(
            f"Missing {source}. Run the Phase 3A0 pipeline first: "
            "python -m nba_prediction_market.pipelines.build_history"
        )
    games = pd.read_parquet(source)
    eligible = games[games["modeling_eligible"]].copy()
    eligible["game_datetime_utc"] = pd.to_datetime(eligible["game_datetime_utc"], utc=True)
    eligible = eligible.sort_values(
        ["game_datetime_utc", "nba_game_id"], kind="stable"
    ).reset_index(drop=True)
    logger.info("Building features for %d eligible regular-season games", len(eligible))
    return pd.DataFrame(build_features(eligible.to_dict("records")), columns=list(FEATURE_COLUMNS))


def attach_elo(features: pd.DataFrame, config: EloConfig) -> pd.DataFrame:
    """Join pregame Elo produced by one chronological run of ``config``."""
    ordered = features.sort_values(["game_datetime_utc", "nba_game_id"], kind="stable")
    predictions = run_elo(ordered.to_dict("records"), config)
    elo = pd.DataFrame(
        [
            {
                "nba_game_id": p.nba_game_id,
                "home_elo": p.home_elo,
                "away_elo": p.away_elo,
                "elo_diff": p.elo_diff,
                "elo_probability": p.home_win_probability,
            }
            for p in predictions
        ]
    )
    merged = features.merge(elo, on="nba_game_id", how="left", validate="one_to_one")
    if merged["elo_diff"].isna().any():
        raise ValueError("every game must receive an Elo rating")
    return merged


# --- stage 2: development selection ---------------------------------------


def stage_select(features: pd.DataFrame) -> tuple[FrozenConfig, dict[str, Any]]:
    """Run development validation and freeze every modelling choice.

    Receives only pre-holdout seasons; :func:`assert_no_holdout` enforces it.
    """
    development = features[features["season"] < HOLDOUT_SEASON].copy()
    assert_no_holdout(development["season"].unique(), where="stage_select")

    logger.info("Selecting Elo configuration over %d games", len(development))
    elo_config, elo_ranked = select_elo(development)

    # elo_diff for the logistic model comes from the *selected* Elo config.
    with_elo = attach_elo(features, elo_config)
    development_with_elo = with_elo[with_elo["season"] < HOLDOUT_SEASON]
    assert_no_holdout(development_with_elo["season"].unique(), where="stage_select(logistic)")

    logger.info("Selecting logistic configuration")
    logistic_config, logistic_ranked = select_logistic(development_with_elo)

    prior_seasons = sorted(development["season"].unique().tolist())
    constant = float(development[TARGET].astype(int).mean())
    training = training_seasons(logistic_config, HOLDOUT_SEASON, prior_seasons)

    frozen = FrozenConfig(
        elo=elo_config.to_dict(),
        logistic=logistic_config.to_dict(),
        feature_allowlist=list(MODEL_FEATURES),
        constant_baseline_probability=constant,
        constant_baseline_seasons=prior_seasons,
        logistic_training_seasons=training,
        development_validation_seasons=list(DEVELOPMENT_VALIDATION_SEASONS),
    )
    report = {
        "elo_candidates": [c.to_dict() for c in elo_ranked],
        "elo_tied_with_best": [c.label for c in tied_with_best(elo_ranked)],
        "logistic_candidates": [c.to_dict() for c in logistic_ranked],
        "logistic_tied_with_best": [c.label for c in tied_with_best(logistic_ranked)],
        "development_folds": _fold_table(elo_ranked, logistic_ranked),
    }
    return frozen, report


def _fold_table(
    elo_ranked: list[CandidateResult], logistic_ranked: list[CandidateResult]
) -> dict[str, Any]:
    return {
        "validation_seasons": list(DEVELOPMENT_VALIDATION_SEASONS),
        "best_elo_by_season": [f.to_dict() for f in elo_ranked[0].folds],
        "best_logistic_by_season": [f.to_dict() for f in logistic_ranked[0].folds],
    }


# --- stage 3: holdout ------------------------------------------------------


def load_kalshi(settings: Settings) -> pd.DataFrame:
    """Phase 2 pregame quotes -- benchmark only, never a model input."""
    path = settings.paths.processed / "nba_kalshi_pregame_t30_2025_26.parquet"
    if not path.is_file():
        raise ConfigError(
            f"Missing {path}. Run the Phase 2 pipeline first: "
            "python -m nba_prediction_market.pipelines.build_pregame_quotes"
        )
    quotes = pd.read_parquet(path)
    out = quotes[
        ["nba_game_id", "home_market_midpoint", "away_market_midpoint"]
    ].rename(
        columns={
            "home_market_midpoint": "kalshi_home_midpoint",
            "away_market_midpoint": "kalshi_away_midpoint",
        }
    )
    total = out["kalshi_home_midpoint"] + out["kalshi_away_midpoint"]
    # Normalising removes the market's overround; raw values are preserved.
    out["kalshi_home_probability_normalized"] = out["kalshi_home_midpoint"] / total
    return out


def stage_holdout(
    features_with_elo: pd.DataFrame, frozen: FrozenConfig, kalshi: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate the frozen models on 2025-26 exactly once."""
    holdout = features_with_elo[features_with_elo["season"] == HOLDOUT_SEASON].copy()
    training = features_with_elo[
        features_with_elo["season"].isin(frozen.logistic_training_seasons)
    ]
    if training.empty:
        raise ValueError("no training rows for the frozen logistic configuration")

    config = LogisticConfig(
        training_history=frozen.logistic["training_history"],
        c_value=frozen.logistic["C"],
        half_life=frozen.logistic["half_life_seasons"],
    )
    fitted = fit_logistic(training, config)

    holdout = holdout.sort_values(
        ["game_datetime_utc", "nba_game_id"], kind="stable"
    ).reset_index(drop=True)
    holdout["constant_probability"] = frozen.constant_baseline_probability
    holdout["logistic_probability"] = fitted.predict_proba(holdout)

    merged = holdout.merge(kalshi, on="nba_game_id", how="inner", validate="one_to_one")
    if set(merged["nba_game_id"]) != set(kalshi["nba_game_id"]):
        raise ValueError(
            "holdout predictions and the Kalshi dataset must cover exactly the same games"
        )

    predictions = merged.loc[:, [c for c in PREDICTION_COLUMNS if c in merged.columns]]
    evaluation = evaluate_holdout(predictions)
    evaluation["logistic_training"] = {
        "seasons": fitted.training_seasons,
        "rows": fitted.n_training_rows,
    }
    return predictions, evaluation


PREDICTORS = {
    "constant": "constant_probability",
    "elo": "elo_probability",
    "logistic": "logistic_probability",
    "kalshi_raw_midpoint": "kalshi_home_midpoint",
    "kalshi_normalized": "kalshi_home_probability_normalized",
}


def evaluate_holdout(predictions: pd.DataFrame) -> dict[str, Any]:
    """Metrics, calibration, paired bootstrap, and diagnostic segments."""
    y = predictions[TARGET].astype(int).to_numpy()

    per_model: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    for name, column in PREDICTORS.items():
        prob = predictions[column].to_numpy(dtype=float)
        per_model[name] = M.summary(y, prob)
        calibration[name] = {
            "bins": M.calibration_table(y, prob),
            "expected_calibration_error": M.expected_calibration_error(y, prob),
        }

    comparisons = {}
    for label, model, benchmark in (
        ("logistic_vs_kalshi_normalized", "logistic", "kalshi_normalized"),
        ("elo_vs_kalshi_normalized", "elo", "kalshi_normalized"),
        ("logistic_vs_elo", "logistic", "elo"),
    ):
        m = predictions[PREDICTORS[model]].to_numpy(dtype=float)
        b = predictions[PREDICTORS[benchmark]].to_numpy(dtype=float)
        comparisons[label] = {
            "model": model,
            "benchmark": benchmark,
            "brier": M.paired_bootstrap(M.brier_losses(y, m), M.brier_losses(y, b)),
            "log_loss": M.paired_bootstrap(M.log_losses(y, m), M.log_losses(y, b)),
        }

    return {
        "metrics": per_model,
        "calibration": calibration,
        "paired_comparisons": comparisons,
        "segments": diagnostic_segments(predictions, y),
    }


def diagnostic_segments(predictions: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    """Predefined segments, reported for diagnosis only -- never used to tune."""
    segments: dict[str, pd.Series] = {
        "first_10_games_either_team": (predictions["home_games_played"] < 10)
        | (predictions["away_games_played"] < 10),
        "after_first_10_games_both": (predictions["home_games_played"] >= 10)
        & (predictions["away_games_played"] >= 10),
        "kalshi_home_favourite": predictions["kalshi_home_probability_normalized"] >= 0.5,
        "kalshi_home_underdog": predictions["kalshi_home_probability_normalized"] < 0.5,
        "back_to_back_involved": predictions["home_back_to_back"].astype(bool)
        | predictions["away_back_to_back"].astype(bool),
        "no_back_to_back": ~(
            predictions["home_back_to_back"].astype(bool)
            | predictions["away_back_to_back"].astype(bool)
        ),
    }
    confidence = predictions["kalshi_home_probability_normalized"]
    for low, high in ((0.0, 0.35), (0.35, 0.65), (0.65, 1.01)):
        segments[f"kalshi_confidence_{low:g}_{high:g}"] = (confidence >= low) & (
            confidence < high
        )

    out: dict[str, Any] = {}
    for name, mask in segments.items():
        selected = mask.to_numpy()
        if selected.sum() == 0:
            continue
        out[name] = {
            "n": int(selected.sum()),
            "actual_home_win_rate": float(y[selected].mean()),
            "models": {
                model: {
                    "brier_score": M.brier_score(
                        y[selected], predictions[column].to_numpy(dtype=float)[selected]
                    ),
                    "log_loss": M.log_loss(
                        y[selected], predictions[column].to_numpy(dtype=float)[selected]
                    ),
                }
                for model, column in PREDICTORS.items()
            },
        }
    return out


# --- orchestration ---------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.bool_ | np.integer | np.floating):
        return value.item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return str(value)


def run_pipeline(
    *, settings: Settings | None = None, write_csv: bool = True
) -> dict[str, Any]:
    """Features -> development selection -> freeze -> holdout, in that order."""
    settings = settings or load_settings()
    settings.paths.ensure()
    started = utc_now().isoformat()

    features = stage_features(settings)
    frozen, selection_report = stage_select(features)
    logger.info("Frozen Elo: %s", frozen.elo)
    logger.info("Frozen logistic: %s", frozen.logistic)

    elo_config = EloConfig(
        k_factor=frozen.elo["k_factor"],
        home_advantage=frozen.elo["home_advantage"],
        regression_factor=frozen.elo["regression_factor"],
        history=frozen.elo["history"],
    )
    features_with_elo = attach_elo(features, elo_config)

    kalshi = load_kalshi(settings)
    predictions, evaluation = stage_holdout(features_with_elo, frozen, kalshi)

    written: list[Path] = []
    feature_path = settings.paths.processed / "nba_model_features_2006_26.parquet"
    features_with_elo.to_parquet(feature_path, index=False)
    written.append(feature_path)
    prediction_path = settings.paths.processed / "nba_predictions_2025_26.parquet"
    predictions.to_parquet(prediction_path, index=False)
    written.append(prediction_path)
    if write_csv:
        for frame, path in (
            (features_with_elo, feature_path.with_suffix(".csv")),
            (predictions, prediction_path.with_suffix(".csv")),
        ):
            frame.to_csv(path, index=False)
            written.append(path)

    report = {
        "generated_at_utc": utc_now().isoformat(),
        "started_at_utc": started,
        "source_counts": {
            "feature_rows": len(features_with_elo),
            "holdout_predictions": len(predictions),
            "development_rows": int((features_with_elo["season"] < HOLDOUT_SEASON).sum()),
            "seasons": sorted(features_with_elo["season"].unique().tolist()),
        },
        "feature_missingness": {
            column: {
                "missing": int(features_with_elo[column].isna().sum()),
                "pct": round(
                    100.0 * features_with_elo[column].isna().sum() / len(features_with_elo), 4
                ),
            }
            for column in features_with_elo.columns
            if features_with_elo[column].isna().any()
        },
        "leakage_audits": run_leakage_audits(features),
        "development": selection_report,
        "frozen_configuration": frozen.to_dict(),
        "holdout": evaluation,
        "warnings": _warnings(features_with_elo, predictions, evaluation),
    }
    report_path = settings.paths.reports / "model_baselines_2025_26.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    written.append(report_path)
    report["written_files"] = [str(p) for p in written]
    return report


def run_leakage_audits(features: pd.DataFrame) -> dict[str, Any]:
    """Machine-checkable leakage properties of the produced feature frame."""
    matrix_columns = list(MODEL_FEATURES)
    return {
        "model_feature_allowlist": matrix_columns,
        "target_excluded_from_matrix": TARGET not in matrix_columns,
        "no_kalshi_columns_in_matrix": not any("kalshi" in c for c in matrix_columns),
        "no_score_columns_in_matrix": not any("score" in c for c in matrix_columns),
        "no_provenance_columns_in_matrix": not any(
            c.startswith("source_") or c.endswith("_corrected") for c in matrix_columns
        ),
        "first_game_rest_is_null": int(
            features[features["home_games_played"] == 0]["home_rest_days"].notna().sum()
        )
        == 0,
        "no_first_game_back_to_back": int(
            features[features["home_rest_days"].isna()]["home_back_to_back"].sum()
        )
        == 0,
        "features_sorted_by_actual_tipoff": bool(
            features["game_datetime_utc"].is_monotonic_increasing
        ),
    }


def _warnings(
    features: pd.DataFrame, predictions: pd.DataFrame, evaluation: dict[str, Any]
) -> list[str]:
    out: list[str] = []
    if len(predictions) != 1230:
        out.append(f"expected 1230 holdout predictions, got {len(predictions)}")
    for name, summary in evaluation["metrics"].items():
        if summary["n"] != len(predictions):
            out.append(f"{name} scored {summary['n']} of {len(predictions)} games")
    for label, comparison in evaluation["paired_comparisons"].items():
        if comparison["brier"]["inconclusive"]:
            out.append(f"{label}: Brier difference is not statistically conclusive")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nba_prediction_market.pipelines.build_baselines",
        description=(
            "Phase 3A1: sequential features, development validation, and a single "
            "2025-26 holdout evaluation."
        ),
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    try:
        report = run_pipeline(
            settings=load_settings(args.data_dir), write_csv=not args.no_csv
        )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    print()
    print("Frozen configuration")
    print(f"  Elo      : {report['frozen_configuration']['elo']}")
    print(f"  Logistic : {report['frozen_configuration']['logistic']}")
    print()
    print(f"2025-26 holdout ({report['source_counts']['holdout_predictions']} games)")
    print(f"  {'model':<22s} {'Brier':>9s} {'LogLoss':>9s} {'Acc':>7s} {'AUC':>7s}")
    for name, summary in report["holdout"]["metrics"].items():
        auc = summary["roc_auc"]
        print(
            f"  {name:<22s} {summary['brier_score']:9.5f} {summary['log_loss']:9.5f} "
            f"{summary['accuracy']:7.4f} {auc:7.4f}"
        )
    print()
    for label, comparison in report["holdout"]["paired_comparisons"].items():
        b = comparison["brier"]
        print(
            f"  {label:<34s} ΔBrier {b['mean_loss_difference']:+.5f} "
            f"[{b['ci_low']:+.5f}, {b['ci_high']:+.5f}]"
        )
    print()
    for warning in report["warnings"]:
        print(f"  WARNING: {warning}")
    print("Files written:")
    for path in report["written_files"]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
