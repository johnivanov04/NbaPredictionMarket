"""Phase 3A3: do the paid feeds improve forecasting beyond Phase 3A2?

Stages are separated exactly as in earlier phases so the holdout cannot
influence a choice:

1. :func:`stage_features` derives team-game box scores, possessions, four
   factors, and the lagged efficiency/rotation features.
2. :func:`stage_select` ablates the predetermined bundles on development folds
   and freezes the outcome. It never reads season 2025.
3. :func:`stage_holdout` evaluates the frozen model on 2025-26 once.

Phase 3A1 and 3A2 artefacts are read but never written.

Run with::

    python -m nba_prediction_market.pipelines.build_paid_features
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
from nba_prediction_market.features.paid_features import (
    PAID_FEATURE_COLUMNS,
    build_paid_features,
)
from nba_prediction_market.features.team_box import (
    aggregate_team_games,
)
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.models import metrics as M
from nba_prediction_market.models.bundles import Bundle
from nba_prediction_market.models.logistic import (
    HISTORY_WEIGHTED,
    LogisticConfig,
    fit_logistic,
    training_seasons,
)
from nba_prediction_market.models.paid_bundles import (
    PAID_BUNDLES,
    PAID_BUNDLES_BY_NAME,
    PAID_FAMILIES,
    combined_bundle,
)
from nba_prediction_market.models.selection import (
    DEVELOPMENT_VALIDATION_SEASONS,
    HOLDOUT_SEASON,
    LOGISTIC_C_GRID,
    LOGISTIC_HALF_LIVES,
    LOGISTIC_HISTORY_GRID,
    TIE_TOLERANCE,
    CandidateResult,
    assert_no_holdout,
    rank_candidates,
    tied_with_best,
)
from nba_prediction_market.models.team_strength_selection import (
    BUNDLE_PROBE_CONFIG,
    evaluate_bundle,
)
from nba_prediction_market.pipelines.build_baselines import _json_default, load_kalshi

logger = logging.getLogger(__name__)

#: Phase 3A2's frozen logistic, reproduced so the control is exact.
PHASE_3A2_LOGISTIC = LogisticConfig(training_history=5, c_value=1.0)
EWMA_HALF_LIFE_GRID: tuple[float, ...] = (3.0, 5.0, 10.0)


@dataclass
class FrozenPaidConfig:
    """Every Phase 3A3 choice, fixed before the holdout is read."""

    bundle: str
    bundle_features: list[str]
    #: Families present in the *chosen* bundle.
    bundle_families: list[str]
    #: Families that independently beat the control, which may be a larger set
    #: than the chosen bundle if the combination did not add anything.
    families_that_helped: list[str]
    logistic: dict[str, Any]
    ewma_half_life: float
    logistic_training_seasons: list[int]
    development_validation_seasons: list[int]
    frozen_at_utc: str = field(default_factory=lambda: utc_now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- stage 1: features -----------------------------------------------------


def stage_features(
    settings: Settings, *, ewma_half_life: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Derive team-game measures, then lag them into pregame features."""
    processed = settings.paths.processed
    for name in (
        "nba_team_strength_features_2006_26.parquet",
        "nba_player_game_stats_2006_26.parquet",
        "nba_regular_season_games_2006_26.parquet",
    ):
        if not (processed / name).is_file():
            raise ConfigError(f"Missing {processed / name}. Run the earlier phases first.")

    base = pd.read_parquet(processed / "nba_team_strength_features_2006_26.parquet")
    games = pd.read_parquet(processed / "nba_regular_season_games_2006_26.parquet")
    games = games[games["modeling_eligible"]].copy()
    games["game_datetime_utc"] = pd.to_datetime(games["game_datetime_utc"], utc=True)
    games = games.sort_values(
        ["game_datetime_utc", "nba_game_id"], kind="stable"
    ).reset_index(drop=True)

    players = pd.read_parquet(processed / "nba_player_game_stats_2006_26.parquet")
    players = players[players["nba_game_id"].isin(set(games["nba_game_id"]))]

    team_games, quality = aggregate_team_games(players, games)

    lookup = {
        (row["nba_game_id"], row["team_id"]): row for row in team_games.to_dict("records")
    }
    appearances: dict[tuple[Any, Any], dict[Any, tuple[float | None, float | None]]] = {}
    for row in players.to_dict("records"):
        key = (row["nba_game_id"], row["team_id"])
        appearances.setdefault(key, {})[row["player_id"]] = (
            row["minutes"], row["plus_minus"]
        )

    ordered = base.merge(
        games[["nba_game_id", "game_datetime_utc"]].rename(
            columns={"game_datetime_utc": "_tipoff"}
        ),
        on="nba_game_id", validate="one_to_one",
    ).sort_values(["_tipoff", "nba_game_id"], kind="stable")

    paid = build_paid_features(
        ordered.to_dict("records"), lookup, appearances, ewma_half_life=ewma_half_life
    )
    features = base.merge(
        pd.DataFrame(paid, columns=list(PAID_FEATURE_COLUMNS)),
        on="nba_game_id", validate="one_to_one",
    )
    return features, team_games, quality


# --- stage 2: development selection ---------------------------------------


def stage_select(
    settings: Settings,
) -> tuple[FrozenPaidConfig, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Ablate the bundles on development folds and freeze the outcome."""
    ablation: list[dict[str, Any]] = []
    best_by_half_life: dict[float, dict[str, CandidateResult]] = {}
    frames: dict[float, pd.DataFrame] = {}
    team_games = pd.DataFrame()
    quality: dict[str, Any] = {}

    for half_life in EWMA_HALF_LIFE_GRID:
        features, team_games, quality = stage_features(settings, ewma_half_life=half_life)
        frames[half_life] = features
        development = features[features["season"] < HOLDOUT_SEASON]
        assert_no_holdout(development["season"].unique(), where="stage_select")
        results: dict[str, CandidateResult] = {}
        for bundle in PAID_BUNDLES:
            result = evaluate_bundle(development, bundle, BUNDLE_PROBE_CONFIG)
            results[bundle.name] = result
            ablation.append({"ewma_half_life": half_life, **result.to_dict()})
        best_by_half_life[half_life] = results

    # A family is kept only if it beats the control by more than tie tolerance.
    control_by_half_life = {
        hl: results["A"].mean_brier for hl, results in best_by_half_life.items()
    }
    family_for_bundle = {
        "B": "efficiency", "C": "roster_continuity",
        "D": "rotation_disruption", "E": "player_quality",
    }
    best_half_life = min(control_by_half_life, key=lambda hl: control_by_half_life[hl])
    scores = best_by_half_life[best_half_life]
    control = scores["A"].mean_brier
    helped = [
        family_for_bundle[name]
        for name in ("B", "C", "D", "E")
        if control - scores[name].mean_brier > TIE_TOLERANCE
    ]

    features = frames[best_half_life]
    development = features[features["season"] < HOLDOUT_SEASON]
    candidates = [PAID_BUNDLES_BY_NAME[n] for n in ("A", "B", "C", "D", "E")]
    if helped:
        combined = combined_bundle(helped)
        result = evaluate_bundle(development, combined, BUNDLE_PROBE_CONFIG)
        ablation.append({"ewma_half_life": best_half_life, **result.to_dict()})
        candidates.append(combined)

    ranked_bundles = sorted(
        candidates,
        key=lambda b: (
            evaluate_bundle(development, b, BUNDLE_PROBE_CONFIG).mean_brier,
            len(b.features),
        ),
    )
    chosen = _preferred_bundle(development, candidates)

    configs = [
        LogisticConfig(training_history=h, c_value=c)
        for h in LOGISTIC_HISTORY_GRID for c in LOGISTIC_C_GRID
    ] + [
        LogisticConfig(training_history=HISTORY_WEIGHTED, c_value=c, half_life=hl)
        for hl in LOGISTIC_HALF_LIVES for c in LOGISTIC_C_GRID
    ]
    logistic_ranked = rank_candidates(
        [evaluate_bundle(development, chosen, c) for c in configs]
    )
    logistic_config = _preferred_logistic(tied_with_best(logistic_ranked))

    prior_seasons = sorted(development["season"].unique().tolist())
    bundle_families = [
        name for name, feats in PAID_FAMILIES.items()
        if set(feats) <= set(chosen.features)
    ]
    frozen = FrozenPaidConfig(
        bundle=chosen.name,
        bundle_features=list(chosen.features),
        bundle_families=bundle_families,
        families_that_helped=helped,
        logistic=logistic_config.to_dict(),
        ewma_half_life=best_half_life,
        logistic_training_seasons=training_seasons(
            logistic_config, HOLDOUT_SEASON, prior_seasons
        ),
        development_validation_seasons=list(DEVELOPMENT_VALIDATION_SEASONS),
    )
    report = {
        "bundle_ablation": ablation,
        "families_that_helped": helped,
        "families_tested": sorted(PAID_FAMILIES),
        "control_mean_brier": control,
        "tie_tolerance": TIE_TOLERANCE,
        "bundle_order_by_brier": [b.name for b in ranked_bundles],
        "logistic_candidates": [c.to_dict() for c in logistic_ranked],
        "team_game_quality": quality,
    }
    return frozen, features, team_games, report


def _preferred_bundle(development: pd.DataFrame, candidates: list[Bundle]) -> Bundle:
    """Best by mean Brier; among statistical ties, the fewest features."""
    scored = [
        (evaluate_bundle(development, b, BUNDLE_PROBE_CONFIG).mean_brier, b)
        for b in candidates
    ]
    best = min(score for score, _ in scored)
    tied = [b for score, b in scored if score - best <= TIE_TOLERANCE]
    return min(tied, key=lambda b: (len(b.features), b.name))


def _preferred_logistic(tied: list[CandidateResult]) -> LogisticConfig:
    """Same documented rule as earlier phases: simplest history wins."""
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


# --- stage 3: holdout ------------------------------------------------------


def stage_holdout(
    features: pd.DataFrame, frozen: FrozenPaidConfig, kalshi: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate the frozen model, the Phase 3A2 control, MOV Elo, and the market."""
    holdout = features[features["season"] == HOLDOUT_SEASON].copy()
    bundle = (
        PAID_BUNDLES_BY_NAME[frozen.bundle]
        if frozen.bundle in PAID_BUNDLES_BY_NAME
        else combined_bundle(frozen.bundle_families)
    )
    config = LogisticConfig(
        training_history=frozen.logistic["training_history"],
        c_value=frozen.logistic["C"],
        half_life=frozen.logistic["half_life_seasons"],
    )
    training = features[features["season"].isin(frozen.logistic_training_seasons)]
    phase_3a3 = fit_logistic(training, config, bundle.features)

    control_seasons = training_seasons(
        PHASE_3A2_LOGISTIC,
        HOLDOUT_SEASON,
        sorted(features[features["season"] < HOLDOUT_SEASON]["season"].unique()),
    )
    phase_3a2 = fit_logistic(
        features[features["season"].isin(control_seasons)],
        PHASE_3A2_LOGISTIC,
        PAID_BUNDLES_BY_NAME["A"].features,
    )

    holdout = holdout.sort_values(
        ["game_datetime_utc", "nba_game_id"], kind="stable"
    ).reset_index(drop=True)
    holdout["phase_3a3_logistic_probability"] = phase_3a3.predict_proba(holdout)
    holdout["phase_3a2_logistic_probability"] = phase_3a2.predict_proba(holdout)

    merged = holdout.merge(kalshi, on="nba_game_id", how="inner", validate="one_to_one")
    if set(merged["nba_game_id"]) != set(kalshi["nba_game_id"]):
        raise ValueError("holdout and Kalshi datasets must cover exactly the same games")

    keep = [
        "nba_game_id", "season", "game_datetime_utc", "home_team", "away_team", "home_win",
        "elo_diff", "mov_elo_diff", "home_games_played", "away_games_played",
        "home_back_to_back", "away_back_to_back",
        "season_net_efficiency_diff", "rotation_disruption_score_diff",
        "expected_rotation_strength_diff",
        "phase_3a2_logistic_probability", "phase_3a3_logistic_probability",
        "mov_elo_probability", "elo_probability",
        "kalshi_home_midpoint", "kalshi_away_midpoint",
        "kalshi_home_probability_normalized",
    ]
    predictions = merged.loc[:, [c for c in keep if c in merged.columns]]
    return predictions, _evaluate(predictions)


PREDICTORS = {
    "phase_3a2_logistic": "phase_3a2_logistic_probability",
    "phase_3a3_logistic": "phase_3a3_logistic_probability",
    "mov_elo": "mov_elo_probability",
    "kalshi_normalized": "kalshi_home_probability_normalized",
}


def _evaluate(predictions: pd.DataFrame) -> dict[str, Any]:
    y = predictions["home_win"].astype(int).to_numpy()
    per_model, calibration = {}, {}
    for name, column in PREDICTORS.items():
        prob = predictions[column].to_numpy(dtype=float)
        per_model[name] = M.summary(y, prob)
        calibration[name] = {
            "bins": M.calibration_table(y, prob),
            "expected_calibration_error": M.expected_calibration_error(y, prob),
        }
    comparisons = {}
    for label, model, benchmark in (
        ("phase_3a3_vs_phase_3a2", "phase_3a3_logistic", "phase_3a2_logistic"),
        ("phase_3a3_vs_mov_elo", "phase_3a3_logistic", "mov_elo"),
        ("phase_3a3_vs_kalshi", "phase_3a3_logistic", "kalshi_normalized"),
    ):
        m = predictions[PREDICTORS[model]].to_numpy(dtype=float)
        b = predictions[PREDICTORS[benchmark]].to_numpy(dtype=float)
        comparisons[label] = {
            "model": model, "benchmark": benchmark,
            "brier": M.paired_bootstrap(M.brier_losses(y, m), M.brier_losses(y, b)),
            "log_loss": M.paired_bootstrap(M.log_losses(y, m), M.log_losses(y, b)),
        }
    return {
        "metrics": per_model,
        "calibration": calibration,
        "paired_comparisons": comparisons,
        "segments": _segments(predictions, y),
    }


def _segments(predictions: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    """Predefined diagnostic segments. Never used to tune."""
    disruption = predictions.get("rotation_disruption_score_diff")
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
    if disruption is not None:
        magnitude = disruption.abs()
        threshold = magnitude.quantile(0.75)
        segments["large_rotation_disruption_gap"] = magnitude >= threshold
        segments["small_rotation_disruption_gap"] = magnitude < threshold

    out: dict[str, Any] = {}
    for name, mask in segments.items():
        selected = mask.fillna(False).to_numpy()
        if selected.sum() == 0:
            continue
        out[name] = {
            "n": int(selected.sum()),
            "actual_home_win_rate": float(y[selected].mean()),
            "models": {
                model: {
                    "brier_score": M.brier_score(
                        y[selected], predictions[column].to_numpy(dtype=float)[selected]
                    )
                }
                for model, column in PREDICTORS.items()
            },
        }
    return out


# --- orchestration ---------------------------------------------------------


def run_pipeline(
    *, settings: Settings | None = None, write_csv: bool = False
) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    started = utc_now().isoformat()

    frozen, features, team_games, selection = stage_select(settings)
    logger.info("Frozen Phase 3A3: %s", frozen.to_dict())

    kalshi = load_kalshi(settings)
    predictions, evaluation = stage_holdout(features, frozen, kalshi)

    written: list[Path] = []
    for frame, stem in (
        (team_games, "nba_team_game_paid_features_2006_26"),
        (features, "nba_model_features_3a3_2006_26"),
        (predictions, "nba_predictions_3a3_2025_26"),
    ):
        path = settings.paths.processed / f"{stem}.parquet"
        frame.to_parquet(path, index=False)
        written.append(path)
        if write_csv:
            frame.to_csv(path.with_suffix(".csv"), index=False)
            written.append(path.with_suffix(".csv"))

    report = {
        "generated_at_utc": utc_now().isoformat(),
        "started_at_utc": started,
        "source_counts": {
            "feature_rows": len(features),
            "team_game_rows": len(team_games),
            "holdout_predictions": len(predictions),
        },
        "team_game_quality": selection["team_game_quality"],
        "possession_validation": _possession_validation(team_games),
        "feature_missingness": {
            column: {
                "missing": int(features[column].isna().sum()),
                "pct": round(100.0 * features[column].isna().sum() / len(features), 4),
            }
            for column in PAID_FEATURE_COLUMNS
            if column in features.columns and features[column].isna().any()
        },
        "leakage_audits": run_leakage_audits(features, frozen),
        "development": selection,
        "frozen_configuration": frozen.to_dict(),
        "holdout": evaluation,
    }
    report_path = settings.paths.reports / "model_paid_features_2025_26.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    written.append(report_path)
    report["written_files"] = [str(p) for p in written]
    return report


def _possession_validation(team_games: pd.DataFrame) -> dict[str, Any]:
    possessions = team_games["estimated_possessions"].dropna()
    corrected = {28012, 32587, 34714, 48851}
    overtime = team_games[team_games["nba_game_id"].isin(corrected)]
    return {
        "n": len(possessions),
        "p01": float(possessions.quantile(0.01)),
        "median": float(possessions.median()),
        "p99": float(possessions.quantile(0.99)),
        "min": float(possessions.min()),
        "max": float(possessions.max()),
        "implausible_outside_60_140": int(
            ((possessions < 60) | (possessions > 140)).sum()
        ),
        "non_positive": int((possessions <= 0).sum()),
        "corrected_4ot_games_rows": len(overtime),
        "corrected_4ot_median_possessions": float(
            overtime["estimated_possessions"].median()
        ) if len(overtime) else None,
        "formula": (
            "Oliver: 0.5 * (team + opponent) where each side is "
            "FGA + 0.44*FTA - 1.07*(OREB/(OREB+OppDREB))*(FGA-FGM) + TOV. "
            "This is an ESTIMATE, not an official NBA statistic."
        ),
    }


def run_leakage_audits(
    features: pd.DataFrame, frozen: FrozenPaidConfig
) -> dict[str, Any]:
    forbidden = ("kalshi", "injur", "lineup", "starter", "season_average")
    return {
        "bundle_features": frozen.bundle_features,
        "no_kalshi_in_bundle": not any("kalshi" in c for c in frozen.bundle_features),
        "no_injury_or_lineup_field_in_bundle": not any(
            token in c for c in frozen.bundle_features for token in forbidden[1:]
        ),
        "no_target_in_bundle": "home_win" not in frozen.bundle_features,
        "no_raw_same_game_stat_in_bundle": not any(
            c in {"pts", "fga", "minutes", "plus_minus", "offensive_efficiency"}
            for c in frozen.bundle_features
        ),
        "every_bundle_feature_is_a_lagged_difference_or_3a2_feature": all(
            c.endswith("_diff") or c in PAID_BUNDLES_BY_NAME["A"].features
            for c in frozen.bundle_features
        ),
        "features_sorted_by_actual_tipoff": bool(
            features["game_datetime_utc"].is_monotonic_increasing
        ),
        "holdout_excluded_from_training": max(frozen.logistic_training_seasons)
        < HOLDOUT_SEASON,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nba_prediction_market.pipelines.build_paid_features",
        description="Phase 3A3: lagged advanced team and rotation features.",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--csv", action="store_true")
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
        report = run_pipeline(settings=load_settings(args.data_dir), write_csv=args.csv)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    frozen = report["frozen_configuration"]
    print()
    print("Frozen Phase 3A3 configuration")
    print(f"  bundle    : {frozen['bundle']} ({len(frozen['bundle_features'])} features)")
    print(f"  families  : {frozen['bundle_families'] or 'none'}")
    print(f"  helped    : {frozen['families_that_helped'] or 'none'}")
    print(f"  logistic  : {frozen['logistic']}")
    print()
    print(f"2025-26 holdout ({report['source_counts']['holdout_predictions']} games)")
    print(f"  {'model':<22s} {'Brier':>9s} {'LogLoss':>9s} {'Acc':>7s} {'AUC':>7s}")
    for name, summary in report["holdout"]["metrics"].items():
        print(
            f"  {name:<22s} {summary['brier_score']:9.5f} {summary['log_loss']:9.5f} "
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
