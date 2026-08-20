"""Phase 3A2: improved team-strength representation.

Extends Phase 3A1 without touching it. The Phase 3A1 artefacts and frozen
configuration remain exactly as they were; this pipeline writes its own feature
table, prediction table, and report so the two can be compared on identical
games.

Stages mirror Phase 3A1's separation:

1. :func:`stage_features` builds every rating and feature family.
2. :func:`stage_select` runs development validation -- refined Elo, MOV Elo,
   bundle ablation, logistic tuning, blending -- and freezes the result. It never
   reads season 2025.
3. :func:`stage_holdout` evaluates the frozen model on 2025-26 once, alongside
   the Phase 3A1 model and the Kalshi benchmark.

Run with::

    python -m nba_prediction_market.pipelines.build_team_strength
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.features.adjusted_margin import run_adjusted_margin
from nba_prediction_market.features.elo import EloConfig, run_elo
from nba_prediction_market.features.feature_engine import FEATURE_COLUMNS, build_features
from nba_prediction_market.features.mov_elo import MovEloConfig, run_mov_elo
from nba_prediction_market.features.team_strength import (
    TEAM_STRENGTH_COLUMNS,
    build_team_strength_features,
)
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.models import metrics as M
from nba_prediction_market.models.bundles import BUNDLES, BUNDLES_BY_NAME, validate_bundle
from nba_prediction_market.models.logistic import (
    HISTORY_WEIGHTED,
    LogisticConfig,
    fit_logistic,
    training_seasons,
)
from nba_prediction_market.models.selection import (
    DEVELOPMENT_VALIDATION_SEASONS,
    HOLDOUT_SEASON,
    LOGISTIC_C_GRID,
    LOGISTIC_HALF_LIVES,
    LOGISTIC_HISTORY_GRID,
    assert_no_holdout,
    rank_candidates,
    tied_with_best,
)
from nba_prediction_market.models.team_strength_selection import (
    BLEND_WEIGHT_GRID,
    BUNDLE_PROBE_CONFIG,
    EWMA_HALF_LIFE_GRID,
    blend,
    evaluate_blend,
    evaluate_bundle,
    evaluate_logistic_on_bundle,
    select_mov_elo,
    select_refined_elo,
)
from nba_prediction_market.pipelines.build_baselines import _json_default, load_kalshi

logger = logging.getLogger(__name__)

#: Phase 3A1's frozen choices, reproduced so the control is exact.
PHASE_3A1_ELO = EloConfig(
    k_factor=20.0, home_advantage=40.0, regression_factor=0.5, history="all_available"
)
PHASE_3A1_LOGISTIC = LogisticConfig(training_history=5, c_value=10.0)


@dataclass
class FrozenTeamStrengthConfig:
    """Every Phase 3A2 choice, fixed before the holdout is read."""

    elo: dict[str, Any]
    mov_elo: dict[str, Any]
    bundle: str
    bundle_features: list[str]
    logistic: dict[str, Any]
    ewma_half_life: float
    blend: dict[str, Any] | None
    logistic_training_seasons: list[int]
    development_validation_seasons: list[int]
    frozen_at_utc: str = field(default_factory=lambda: utc_now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- stage 1: features -----------------------------------------------------


def stage_features(
    settings: Settings,
    elo_config: EloConfig,
    mov_config: MovEloConfig,
    *,
    ewma_half_life: float,
) -> pd.DataFrame:
    """Build every Phase 3A2 rating and feature family for eligible games."""
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
    records = eligible.to_dict("records")

    frame = pd.DataFrame(build_features(records), columns=list(FEATURE_COLUMNS))

    elo = run_elo(records, elo_config)
    frame = frame.merge(
        pd.DataFrame(
            [
                {
                    "nba_game_id": p.nba_game_id, "home_elo": p.home_elo,
                    "away_elo": p.away_elo, "elo_diff": p.elo_diff,
                    "elo_probability": p.home_win_probability,
                }
                for p in elo
            ]
        ),
        on="nba_game_id", validate="one_to_one",
    )

    mov = run_mov_elo(records, mov_config)
    frame = frame.merge(
        pd.DataFrame(
            [
                {
                    "nba_game_id": p.nba_game_id, "home_mov_elo": p.home_mov_elo,
                    "away_mov_elo": p.away_mov_elo, "mov_elo_diff": p.mov_elo_diff,
                    "mov_elo_probability": p.home_win_probability,
                }
                for p in mov
            ]
        ),
        on="nba_game_id", validate="one_to_one",
    )

    adjusted = run_adjusted_margin(records)
    frame = frame.merge(
        pd.DataFrame(
            [
                {
                    "nba_game_id": p.nba_game_id,
                    "home_adjusted_margin_rating": p.home_adjusted_margin_rating,
                    "away_adjusted_margin_rating": p.away_adjusted_margin_rating,
                    "adjusted_margin_diff": p.adjusted_margin_diff,
                }
                for p in adjusted
            ]
        ),
        on="nba_game_id", validate="one_to_one",
    )

    elo_by_game = {p.nba_game_id: (p.home_elo, p.away_elo) for p in elo}
    strength = build_team_strength_features(
        records, elo_by_game, ewma_half_life=ewma_half_life
    )
    return frame.merge(
        pd.DataFrame(strength, columns=list(TEAM_STRENGTH_COLUMNS)),
        on="nba_game_id", validate="one_to_one",
    )


# --- stage 2: development selection ---------------------------------------


def stage_select(
    settings: Settings,
) -> tuple[FrozenTeamStrengthConfig, pd.DataFrame, dict[str, Any]]:
    """Run every Phase 3A2 development experiment and freeze the outcome."""
    source = settings.paths.processed / "nba_regular_season_games_2006_26.parquet"
    games = pd.read_parquet(source)
    eligible = games[games["modeling_eligible"]].copy()
    eligible["game_datetime_utc"] = pd.to_datetime(eligible["game_datetime_utc"], utc=True)
    eligible = eligible.sort_values(
        ["game_datetime_utc", "nba_game_id"], kind="stable"
    ).reset_index(drop=True)
    development_games = eligible[eligible["season"] < HOLDOUT_SEASON]
    assert_no_holdout(development_games["season"].unique(), where="stage_select")

    logger.info("Refining binary Elo (home-court grid extended to zero)")
    elo_config, elo_ranked = select_refined_elo(development_games)

    hca_grid = sorted(
        {
            max(0.0, elo_config.home_advantage - 10.0),
            elo_config.home_advantage,
            elo_config.home_advantage + 10.0,
        }
    )
    logger.info("Searching margin-of-victory Elo")
    mov_config, mov_ranked = select_mov_elo(development_games, hca_grid=hca_grid)

    # Bundles are compared at each EWMA half-life with a fixed logistic, so the
    # ablation measures features rather than a lucky hyperparameter pairing.
    bundle_results: list[dict[str, Any]] = []
    best_key: tuple[float, str] | None = None
    best_score = float("inf")
    features_by_half_life: dict[float, pd.DataFrame] = {}
    for half_life in EWMA_HALF_LIFE_GRID:
        frame = stage_features(settings, elo_config, mov_config, ewma_half_life=half_life)
        features_by_half_life[half_life] = frame
        development = frame[frame["season"] < HOLDOUT_SEASON]
        for bundle in BUNDLES:
            validate_bundle(bundle)
            result = evaluate_bundle(development, bundle, BUNDLE_PROBE_CONFIG)
            bundle_results.append(
                {"ewma_half_life": half_life, **result.to_dict()}
            )
            if result.mean_brier < best_score:
                best_score, best_key = result.mean_brier, (half_life, bundle.name)

    half_life, bundle_name = best_key
    bundle = BUNDLES_BY_NAME[bundle_name]
    features = features_by_half_life[half_life]
    development = features[features["season"] < HOLDOUT_SEASON]
    logger.info("Selected bundle %s at EWMA half-life %g", bundle_name, half_life)

    configs = [
        LogisticConfig(training_history=h, c_value=c)
        for h in LOGISTIC_HISTORY_GRID
        for c in LOGISTIC_C_GRID
    ] + [
        LogisticConfig(training_history=HISTORY_WEIGHTED, c_value=c, half_life=hl)
        for hl in LOGISTIC_HALF_LIVES
        for c in LOGISTIC_C_GRID
    ]
    logistic_ranked = rank_candidates(
        [evaluate_logistic_on_bundle(development, bundle, c) for c in configs]
    )
    logistic_config = _preferred_logistic(tied_with_best(logistic_ranked))

    blend_results, chosen_blend = _evaluate_blends(
        development, bundle, logistic_config
    )

    prior_seasons = sorted(development["season"].unique().tolist())
    frozen = FrozenTeamStrengthConfig(
        elo=elo_config.to_dict(),
        mov_elo=mov_config.to_dict(),
        bundle=bundle.name,
        bundle_features=list(bundle.features),
        logistic=logistic_config.to_dict(),
        ewma_half_life=half_life,
        blend=chosen_blend,
        logistic_training_seasons=training_seasons(
            logistic_config, HOLDOUT_SEASON, prior_seasons
        ),
        development_validation_seasons=list(DEVELOPMENT_VALIDATION_SEASONS),
    )
    report = {
        "refined_elo_candidates": [c.to_dict() for c in elo_ranked],
        "mov_elo_candidates": [c.to_dict() for c in mov_ranked],
        "bundle_ablation": bundle_results,
        "logistic_candidates": [c.to_dict() for c in logistic_ranked],
        "logistic_tied_with_best": [c.label for c in tied_with_best(logistic_ranked)],
        "blend_candidates": blend_results,
        "phase_3a1_baseline_development": _phase_3a1_development(development),
    }
    return frozen, features, report


def _preferred_logistic(tied: list[Any]) -> LogisticConfig:
    """Same documented tie rule as Phase 3A1: simplest history wins."""
    finite = [c for c in tied if isinstance(c.config["logistic"]["training_history"], int)]
    pool = finite or list(tied)
    if finite:
        shortest = min(c.config["logistic"]["training_history"] for c in finite)
        pool = [c for c in finite if c.config["logistic"]["training_history"] == shortest]
    best = min(pool, key=lambda c: (c.mean_brier, c.mean_log_loss)).config["logistic"]
    return LogisticConfig(
        training_history=best["training_history"],
        c_value=best["C"],
        half_life=best["half_life_seasons"],
    )


def _fold_probabilities(
    development: pd.DataFrame, bundle: Any, config: LogisticConfig
) -> tuple[dict[int, np.ndarray], ...]:
    available = sorted(development["season"].unique().tolist())
    y, logistic, elo, mov, adjusted = {}, {}, {}, {}, {}
    for season in DEVELOPMENT_VALIDATION_SEASONS:
        training = development[
            development["season"].isin(training_seasons(config, season, available))
        ]
        validation = development[development["season"] == season]
        if training.empty or validation.empty:
            continue
        y[season] = validation["home_win"].astype(int).to_numpy()
        logistic[season] = fit_logistic(training, config, bundle.features).predict_proba(
            validation
        )
        elo[season] = validation["elo_probability"].to_numpy()
        mov[season] = validation["mov_elo_probability"].to_numpy()
        adjusted[season] = fit_logistic(
            training, config, ["adjusted_margin_diff"]
        ).predict_proba(validation)
    return y, logistic, elo, mov, adjusted


def _evaluate_blends(
    development: pd.DataFrame, bundle: Any, config: LogisticConfig
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Test small blends; keep one only if it beats the pure model meaningfully."""
    y, logistic, elo, mov, adjusted = _fold_probabilities(development, bundle, config)
    results = []
    for label, partner in (
        ("binary_elo", elo), ("mov_elo", mov), ("adjusted_margin", adjusted)
    ):
        for weight in BLEND_WEIGHT_GRID:
            results.append(evaluate_blend(y, logistic, partner, weight, label).to_dict())

    pure = min(r["mean_brier_score"] for r in results if r["config"]["weight"] == 0.0)
    best = min(results, key=lambda r: (r["mean_brier_score"], r["mean_log_loss"]))
    # A blend must clear the same tie tolerance used everywhere else.
    if best["config"]["weight"] > 0.0 and pure - best["mean_brier_score"] > 1e-4:
        return results, best["config"]
    return results, None


def _phase_3a1_development(development: pd.DataFrame) -> dict[str, Any]:
    """The Phase 3A1 control, scored on the same folds."""
    result = evaluate_logistic_on_bundle(
        development, BUNDLES_BY_NAME["A"], PHASE_3A1_LOGISTIC
    )
    return result.to_dict()


# --- stage 3: holdout ------------------------------------------------------


PREDICTION_COLUMNS: list[str] = [
    "nba_game_id", "season", "game_datetime_utc", "home_team", "away_team", "home_win",
    "elo_diff", "mov_elo_diff", "adjusted_margin_diff",
    "phase_3a1_logistic_probability", "phase_3a2_logistic_probability",
    "elo_probability", "mov_elo_probability",
    "kalshi_home_midpoint", "kalshi_away_midpoint", "kalshi_home_probability_normalized",
]


def stage_holdout(
    features: pd.DataFrame, frozen: FrozenTeamStrengthConfig, kalshi: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate the frozen Phase 3A2 model, the Phase 3A1 control, and the market."""
    holdout = features[features["season"] == HOLDOUT_SEASON].copy()
    bundle = BUNDLES_BY_NAME[frozen.bundle]
    config = LogisticConfig(
        training_history=frozen.logistic["training_history"],
        c_value=frozen.logistic["C"],
        half_life=frozen.logistic["half_life_seasons"],
    )
    training = features[features["season"].isin(frozen.logistic_training_seasons)]

    phase_3a2 = fit_logistic(training, config, bundle.features)
    # The Phase 3A1 control uses its own frozen configuration and feature set.
    control_training = features[
        features["season"].isin(
            training_seasons(
                PHASE_3A1_LOGISTIC,
                HOLDOUT_SEASON,
                sorted(features[features["season"] < HOLDOUT_SEASON]["season"].unique()),
            )
        )
    ]
    phase_3a1 = fit_logistic(
        control_training, PHASE_3A1_LOGISTIC, BUNDLES_BY_NAME["A"].features
    )

    holdout = holdout.sort_values(
        ["game_datetime_utc", "nba_game_id"], kind="stable"
    ).reset_index(drop=True)
    holdout["phase_3a2_logistic_probability"] = phase_3a2.predict_proba(holdout)
    holdout["phase_3a1_logistic_probability"] = phase_3a1.predict_proba(holdout)

    merged = holdout.merge(kalshi, on="nba_game_id", how="inner", validate="one_to_one")
    if set(merged["nba_game_id"]) != set(kalshi["nba_game_id"]):
        raise ValueError(
            "holdout predictions and the Kalshi dataset must cover exactly the same games"
        )
    if frozen.blend is not None:
        partner = {
            "binary_elo": "elo_probability", "mov_elo": "mov_elo_probability"
        }.get(frozen.blend["partner"])
        if partner:
            merged["phase_3a2_blend_probability"] = blend(
                merged["phase_3a2_logistic_probability"].to_numpy(),
                merged[partner].to_numpy(),
                frozen.blend["weight"],
            )

    predictions = merged.loc[
        :, [c for c in PREDICTION_COLUMNS if c in merged.columns]
        + (["phase_3a2_blend_probability"] if "phase_3a2_blend_probability" in merged else [])
    ]
    return predictions, _evaluate(predictions, frozen)


def _evaluate(
    predictions: pd.DataFrame, frozen: FrozenTeamStrengthConfig
) -> dict[str, Any]:
    y = predictions["home_win"].astype(int).to_numpy()
    predictors = {
        "phase_3a1_logistic": "phase_3a1_logistic_probability",
        "phase_3a2_logistic": "phase_3a2_logistic_probability",
        "elo": "elo_probability",
        "mov_elo": "mov_elo_probability",
        "kalshi_normalized": "kalshi_home_probability_normalized",
    }
    if "phase_3a2_blend_probability" in predictions.columns:
        predictors["phase_3a2_blend"] = "phase_3a2_blend_probability"

    per_model, calibration = {}, {}
    for name, column in predictors.items():
        prob = predictions[column].to_numpy(dtype=float)
        per_model[name] = M.summary(y, prob)
        calibration[name] = {
            "bins": M.calibration_table(y, prob),
            "expected_calibration_error": M.expected_calibration_error(y, prob),
        }

    comparisons = {}
    for label, model, benchmark in (
        ("phase_3a2_vs_phase_3a1", "phase_3a2_logistic", "phase_3a1_logistic"),
        ("phase_3a2_vs_kalshi", "phase_3a2_logistic", "kalshi_normalized"),
        ("phase_3a1_vs_kalshi", "phase_3a1_logistic", "kalshi_normalized"),
    ):
        m = predictions[predictors[model]].to_numpy(dtype=float)
        b = predictions[predictors[benchmark]].to_numpy(dtype=float)
        comparisons[label] = {
            "model": model, "benchmark": benchmark,
            "brier": M.paired_bootstrap(M.brier_losses(y, m), M.brier_losses(y, b)),
            "log_loss": M.paired_bootstrap(M.log_losses(y, m), M.log_losses(y, b)),
        }
    return {"metrics": per_model, "calibration": calibration, "paired_comparisons": comparisons}


# --- orchestration ---------------------------------------------------------


def run_pipeline(
    *, settings: Settings | None = None, write_csv: bool = True
) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    started = utc_now().isoformat()

    frozen, features, selection_report = stage_select(settings)
    logger.info("Frozen Phase 3A2: %s", frozen.to_dict())

    kalshi = load_kalshi(settings)
    predictions, evaluation = stage_holdout(features, frozen, kalshi)

    written: list[Path] = []
    feature_path = settings.paths.processed / "nba_team_strength_features_2006_26.parquet"
    features.to_parquet(feature_path, index=False)
    written.append(feature_path)
    prediction_path = settings.paths.processed / "nba_predictions_3a2_2025_26.parquet"
    predictions.to_parquet(prediction_path, index=False)
    written.append(prediction_path)
    if write_csv:
        for frame, path in (
            (features, feature_path.with_suffix(".csv")),
            (predictions, prediction_path.with_suffix(".csv")),
        ):
            frame.to_csv(path, index=False)
            written.append(path)

    report = {
        "generated_at_utc": utc_now().isoformat(),
        "started_at_utc": started,
        "source_counts": {
            "feature_rows": len(features),
            "holdout_predictions": len(predictions),
            "seasons": sorted(features["season"].unique().tolist()),
        },
        "feature_missingness": {
            column: {
                "missing": int(features[column].isna().sum()),
                "pct": round(100.0 * features[column].isna().sum() / len(features), 4),
            }
            for column in features.columns
            if features[column].isna().any()
        },
        "leakage_audits": run_leakage_audits(features, frozen),
        "development": selection_report,
        "frozen_configuration": frozen.to_dict(),
        "holdout": evaluation,
    }
    report_path = settings.paths.reports / "model_team_strength_2025_26.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    written.append(report_path)
    report["written_files"] = [str(p) for p in written]
    return report


def run_leakage_audits(
    features: pd.DataFrame, frozen: FrozenTeamStrengthConfig
) -> dict[str, Any]:
    """Machine-checkable properties of the Phase 3A2 feature frame."""
    bundle = BUNDLES_BY_NAME[frozen.bundle]
    return {
        "bundle_features": list(bundle.features),
        "no_kalshi_in_bundle": not any("kalshi" in c for c in bundle.features),
        "no_target_in_bundle": "home_win" not in bundle.features,
        "no_raw_score_in_bundle": not any(
            c in {"home_score", "away_score"} for c in bundle.features
        ),
        "features_sorted_by_actual_tipoff": bool(
            features["game_datetime_utc"].is_monotonic_increasing
        ),
        "first_game_rest_is_null": int(
            features[features["home_games_played"] == 0]["home_rest_days"].notna().sum()
        ) == 0,
        "adjusted_margin_sparse_early": bool(
            features[features["home_games_played"] < 3]["home_adjusted_margin_rating"]
            .isna()
            .all()
        ),
        "holdout_excluded_from_training": max(frozen.logistic_training_seasons)
        < HOLDOUT_SEASON,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nba_prediction_market.pipelines.build_team_strength",
        description="Phase 3A2: improved team-strength features and a single holdout check.",
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
        report = run_pipeline(settings=load_settings(args.data_dir), write_csv=not args.no_csv)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    frozen = report["frozen_configuration"]
    print()
    print("Frozen Phase 3A2 configuration")
    print(f"  bundle    : {frozen['bundle']} ({len(frozen['bundle_features'])} features)")
    print(f"  Elo       : {frozen['elo']}")
    print(f"  MOV Elo   : {frozen['mov_elo']}")
    print(f"  logistic  : {frozen['logistic']}")
    print(f"  blend     : {frozen['blend'] or 'none (did not help)'}")
    print()
    print(f"2025-26 holdout ({report['source_counts']['holdout_predictions']} games)")
    print(f"  {'model':<24s} {'Brier':>9s} {'LogLoss':>9s} {'Acc':>7s} {'AUC':>7s}")
    for name, summary in report["holdout"]["metrics"].items():
        print(
            f"  {name:<24s} {summary['brier_score']:9.5f} {summary['log_loss']:9.5f} "
            f"{summary['accuracy']:7.4f} {summary['roc_auc']:7.4f}"
        )
    print()
    for label, comparison in report["holdout"]["paired_comparisons"].items():
        b = comparison["brier"]
        print(
            f"  {label:<26s} ΔBrier {b['mean_loss_difference']:+.5f} "
            f"[{b['ci_low']:+.5f}, {b['ci_high']:+.5f}]"
        )
    print()
    print("Files written:")
    for path in report["written_files"]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
