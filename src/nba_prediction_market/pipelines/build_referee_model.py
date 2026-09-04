"""Phase 4A4: does leakage-safe referee information beat the frozen 3A3C model?

The design is the same one every earlier phase used, for the same reason: the
question is whether a *specific* new information family adds anything, so each
bundle differs from the control by exactly one family, the ablation is fixed
before results are seen, and the 2025-26 season is scored once, afterwards.

Referee features are joined, never recomputed here. They arrive from
``build_referee_features``, which walks games chronologically and enforces the
read-before-write ordering that keeps a game out of its own features.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.models import metrics
from nba_prediction_market.models.bundles import Bundle
from nba_prediction_market.models.logistic import (
    LogisticConfig,
    fit_logistic,
    training_seasons,
)
from nba_prediction_market.models.selection import (
    DEVELOPMENT_VALIDATION_SEASONS,
    HOLDOUT_SEASON,
    assert_no_holdout,
)
from nba_prediction_market.referees.bundles import (
    BASE_BUNDLES,
    REFEREE_FAMILIES,
    UNBUILDABLE_BUNDLES,
    build_bundle_f,
)
from nba_prediction_market.referees.diagnostics import (
    segment_report,
    team_referee_support,
)
from nba_prediction_market.referees.state import SHRINKAGE_K_GRID

logger = logging.getLogger(__name__)

TARGET = "home_win"

#: The rolling training window Phase 3A3C froze. Matching it exactly matters:
#: with "all_available" the control would train on a different set of seasons
#: than the frozen model did, and every referee family would then be measured
#: against something that is not actually the control.
MAX_TRAINING_SEASONS = 5

#: Tiny and predetermined. This phase tests information, not model class.
C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)

#: Interpretation bands for a Brier improvement, from the phase brief. These
#: describe a result; they are never used to pick one.
MATERIALITY_BANDS: tuple[tuple[float, str], ...] = (
    (0.0001, "effectively no useful signal"),
    (0.0005, "small / potentially real; requires strong fold consistency"),
    (0.0010, "meaningful"),
    (float("inf"), "surprisingly strong; audit aggressively before believing it"),
)

FEATURES_3A3 = "nba_model_features_3a3_2006_26.parquet"
AVAILABILITY = "nba_game_availability_features_2019_26.parquet"
REFEREE_FEATURES = "nba_referee_features_2019_26.parquet"
ASSIGNMENTS_FILE = "nba_referee_assignments_2019_26.parquet"

STATUS_ORDER = ("out", "doubtful", "questionable", "probable", "available")


def materiality(delta_brier: float) -> str:
    """Plain-language reading of an improvement (negative = referee better)."""
    improvement = -delta_brier
    if improvement <= 0:
        return "no improvement; the referee bundle is worse than the control"
    for threshold, label in MATERIALITY_BANDS:
        if improvement < threshold:
            return label
    return MATERIALITY_BANDS[-1][1]


def load_frame(settings: Settings) -> pd.DataFrame:
    """Control features, availability features, and referee features joined.

    Uses the *native* availability frame, which is what Phase 3A3C's own
    pipeline loads by default.

    On fidelity to the frozen control: this reproduces Phase 3A3C exactly, on
    both halves. Development mean Brier is 0.21296859376431693 at C=0.1,
    matching the published selection to every digit, and the 2025-26 holdout
    predictions match the canonical artefact with **max absolute difference
    0.0** across all 1,230 games (0.20039 / 0.58539 / 0.75051 / 0.02754).

    That was not true at first. ``fit_logistic`` fits whatever rows it is
    handed -- ``training_history`` only selects recency weighting -- so passing
    it every prior season trained the control on six seasons where the frozen
    model uses five. Development was unaffected, because no development fold
    ever has more than five prior seasons, which is why the development
    numbers matched while the holdout did not. ``evaluate`` now applies the
    window itself; see the comment there.
    """
    processed = settings.paths.processed
    for name in (FEATURES_3A3, AVAILABILITY, REFEREE_FEATURES):
        if not (processed / name).is_file():
            raise ConfigError(f"missing {processed / name}")

    base = pd.read_parquet(processed / FEATURES_3A3)
    base = base[base["season"] >= 2019]
    avail = pd.read_parquet(processed / AVAILABILITY)
    referees = pd.read_parquet(processed / REFEREE_FEATURES)

    frame = base.merge(
        avail[[c for c in avail.columns
               if c == "nba_game_id" or c.startswith("avail_")]],
        on="nba_game_id", how="inner",
    )
    ref_cols = [
        c for c in referees.columns
        if c.startswith("ref_crew_") or c in ("nba_game_id", "referee_crew_known",
                                              "referee_crew_size")
    ]
    frame = frame.merge(referees[ref_cols], on="nba_game_id", how="inner")

    # The role-minute family is a plain home-minus-away difference of columns
    # the availability parquet already carries; no fold-specific calibration is
    # involved, so deriving it here reproduces the frozen control exactly.
    for status in STATUS_ORDER:
        home = frame.get(f"avail_home_t30_{status}_expected_minutes")
        away = frame.get(f"avail_away_t30_{status}_expected_minutes")
        if home is not None and away is not None:
            frame[f"avail_{status}_expected_minutes_diff"] = home - away
    return frame


def evaluate(
    frame: pd.DataFrame,
    bundle: Bundle,
    validation_season: int,
    training: list[int],
    c_value: float,
) -> dict[str, Any]:
    """Fit on earlier seasons only, score the validation season."""
    assert_no_holdout(training, where=f"referee bundle {bundle.name} training")
    config = LogisticConfig(training_history=MAX_TRAINING_SEASONS, c_value=c_value)

    # Apply the rolling window *here*, not by trusting the caller and not by
    # trusting ``fit_logistic``. ``fit_logistic`` fits on whatever rows it is
    # handed -- ``training_history`` only decides recency weighting -- so
    # passing it every prior season silently trains a wider model than the
    # frozen Phase 3A3C control, which fits on the last five seasons only.
    windowed = training_seasons(config, validation_season, training)
    train = frame[frame["season"].isin(windowed)]
    valid = frame[frame["season"] == validation_season]
    if train.empty or valid.empty:
        raise ConfigError(f"empty split for season {validation_season}")

    fitted = fit_logistic(train, config, features=bundle.features)
    probability = fitted.predict_proba(valid)
    truth = valid[TARGET].astype(int).to_numpy()
    return {
        "season": validation_season,
        "bundle": bundle.name,
        "c_value": c_value,
        "training_seasons": list(windowed),
        "brier": metrics.brier_score(truth, probability),
        "log_loss": metrics.log_loss(truth, probability),
        "auc": metrics.roc_auc(truth, probability),
        "ece": metrics.expected_calibration_error(truth, probability),
        "n_games": len(valid),
        "probability": probability,
        "truth": truth,
        "nba_game_id": valid["nba_game_id"].to_numpy(),
    }


def run_development(
    frame: pd.DataFrame, bundles: tuple[Bundle, ...], k: float
) -> dict[str, Any]:
    """Every bundle on every development fold, control included."""
    assert_no_holdout(list(DEVELOPMENT_VALIDATION_SEASONS), where="development")
    available = tuple(sorted(s for s in frame["season"].unique() if s < HOLDOUT_SEASON))
    folds: list[dict[str, Any]] = []

    for bundle in bundles:
        for c_value in C_GRID:
            for season in DEVELOPMENT_VALIDATION_SEASONS:
                training = [s for s in available if s < season]
                if not training:
                    continue
                result = evaluate(frame, bundle, season, training, c_value)
                folds.append({
                    k2: v for k2, v in result.items()
                    if k2 not in ("probability", "truth", "nba_game_id")
                })

    table = pd.DataFrame(folds)
    by_config = (
        table.groupby(["bundle", "c_value"])
        .agg(mean_brier=("brier", "mean"), mean_log_loss=("log_loss", "mean"),
             mean_auc=("auc", "mean"), folds=("brier", "size"))
        .reset_index()
        .sort_values(["mean_brier", "mean_log_loss"])
    )
    # The control's own best C is the reference point every family is judged
    # against, so a family cannot look good merely by preferring a better C.
    control = by_config[by_config["bundle"] == "A"].iloc[0]

    families: dict[str, Any] = {}
    for bundle in bundles:
        if bundle.name == "A":
            continue
        best = by_config[by_config["bundle"] == bundle.name].iloc[0]
        per_season = table[
            (table["bundle"] == bundle.name) & (table["c_value"] == best["c_value"])
        ].set_index("season")["brier"]
        control_seasons = table[
            (table["bundle"] == "A") & (table["c_value"] == control["c_value"])
        ].set_index("season")["brier"]
        beaten = [
            int(s) for s in per_season.index
            if per_season[s] < control_seasons.get(s, np.inf)
        ]
        families[bundle.name] = {
            "best_c": float(best["c_value"]),
            "mean_brier": float(best["mean_brier"]),
            "delta_vs_control": float(best["mean_brier"] - control["mean_brier"]),
            "seasons_improved": beaten,
            "seasons_total": len(per_season),
            # "More than one isolated season" is the brief's own bar.
            "fold_consistent": len(beaten) >= 2,
            "materiality": materiality(
                float(best["mean_brier"] - control["mean_brier"])
            ),
            "per_season_brier": {int(s): float(v) for s, v in per_season.items()},
        }

    helped = [
        name for name, info in families.items()
        if info["delta_vs_control"] < 0 and info["fold_consistent"]
        and name in REFEREE_FAMILIES
    ]
    return {
        "shrinkage_k": k,
        "development_seasons": list(DEVELOPMENT_VALIDATION_SEASONS),
        "c_grid": list(C_GRID),
        "control": {
            "best_c": float(control["c_value"]),
            "mean_brier": float(control["mean_brier"]),
            "mean_log_loss": float(control["mean_log_loss"]),
        },
        "by_configuration": by_config.to_dict("records"),
        "families": families,
        "families_that_helped": helped,
        "unbuildable_bundles": UNBUILDABLE_BUNDLES,
        "fold_table": folds,
    }


def frame_for_k(settings: Settings, base: pd.DataFrame, k: float) -> pd.DataFrame:
    """The joined frame with referee features rebuilt at shrinkage constant k.

    Rebuilt rather than reloaded: k is baked into the feature values, so
    reading one parquet and calling it three different k values would compare
    the same numbers three times and silently make the grid decorative.
    """
    from nba_prediction_market.pipelines.build_referee_features import (
        build_features,
        load_inputs,
    )

    features = build_features(load_inputs(settings), k=k)
    keep = [c for c in features.columns if c.startswith("ref_crew_")]
    keep += ["nba_game_id", "referee_crew_known", "referee_crew_size"]
    stripped = base.drop(
        columns=[c for c in base.columns if c.startswith("ref_crew_")]
        + [c for c in ("referee_crew_known", "referee_crew_size") if c in base],
        errors="ignore",
    )
    return stripped.merge(features[keep], on="nba_game_id", how="inner")


def select_k(
    settings: Settings, base: pd.DataFrame, bundles: tuple[Bundle, ...]
) -> dict[str, Any]:
    """Choose the shrinkage constant on development folds only.

    A small predetermined grid, scored exactly like everything else. The winner
    is the k whose best non-control bundle has the lowest mean development
    Brier; ties keep the smaller k, which shrinks harder and so claims less.
    """
    trials: list[dict[str, Any]] = []
    frames: dict[float, pd.DataFrame] = {}
    for k in SHRINKAGE_K_GRID:
        frame = frame_for_k(settings, base, k)
        frames[k] = frame
        development = run_development(frame, bundles, k)
        best = min(
            (f["mean_brier"] for f in development["families"].values()),
            default=float("inf"),
        )
        trials.append({
            "k": k,
            "control_mean_brier": development["control"]["mean_brier"],
            "best_family_mean_brier": best,
            "families_that_helped": development["families_that_helped"],
        })
        logger.info("k=%s: best family mean Brier %.6f", k, best)
    ordered = sorted(trials, key=lambda t: (t["best_family_mean_brier"], t["k"]))
    chosen = ordered[0]["k"]
    return {
        "grid": list(SHRINKAGE_K_GRID),
        "trials": trials,
        "chosen_k": chosen,
        "frame": frames[chosen],
    }


def run_benchmark(
    frame: pd.DataFrame, control: Bundle, enhanced: Bundle | None,
    c_control: float, c_enhanced: float,
) -> dict[str, Any]:
    """Score 2025-26 once, with the configuration frozen beforehand."""
    available = tuple(sorted(s for s in frame["season"].unique() if s < HOLDOUT_SEASON))
    training = list(available)

    control_result = evaluate(frame, control, HOLDOUT_SEASON, training, c_control)
    out: dict[str, Any] = {
        "season": HOLDOUT_SEASON,
        "training_seasons": training,
        "control": _metrics_only(control_result),
    }
    # Kept for the post-freeze diagnostics, which need per-game predictions.
    # They run after this and can change nothing about it.
    out["_predictions"] = pd.DataFrame({
        "nba_game_id": control_result["nba_game_id"],
        "home_win": control_result["truth"],
        "control_probability": control_result["probability"],
    })
    if enhanced is None:
        out["enhanced"] = None
        out["note"] = (
            "No referee family earned inclusion on development folds, so there "
            "is no enhanced model to benchmark."
        )
        return out

    enhanced_result = evaluate(frame, enhanced, HOLDOUT_SEASON, training, c_enhanced)
    out["enhanced"] = _metrics_only(enhanced_result)
    out["_predictions"]["referee_probability"] = enhanced_result["probability"]

    control_losses = metrics.brier_losses(
        control_result["truth"], control_result["probability"]
    )
    enhanced_losses = metrics.brier_losses(
        enhanced_result["truth"], enhanced_result["probability"]
    )
    # Sign convention: negative favours the referee model, as everywhere else.
    boot = metrics.paired_bootstrap(
        enhanced_losses, control_losses, n_resamples=10_000, seed=20260903
    )
    out["paired_bootstrap_brier"] = boot
    out["delta_brier"] = (
        out["enhanced"]["brier"] - out["control"]["brier"]
    )
    out["materiality"] = materiality(out["delta_brier"])
    return out


def _metrics_only(result: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v for k, v in result.items()
        if k not in ("probability", "truth", "nba_game_id")
    }


def kalshi_benchmark(settings: Settings) -> dict[str, Any] | None:
    """Kalshi's own T-30 prices on the same season, for context only.

    Uses the two-sided midpoint normalised by the pair sum, the convention
    Phase 4A0 established: the home and away contracts are mutually exclusive
    but their midpoints do not sum to one, so dividing by the sum removes the
    book's vig without pretending either side was individually executable.
    This is a reference line, never a model input.
    """
    path = settings.paths.processed / "nba_kalshi_pregame_t30_2025_26.parquet"
    if not path.is_file():
        return None
    frame = pd.read_parquet(path)
    needed = {"home_market_midpoint", "away_market_midpoint", "home_win",
              "both_sides_usable"}
    if not needed <= set(frame.columns):
        return None

    usable = frame[frame["both_sides_usable"].fillna(False)].copy()
    pair_sum = usable["home_market_midpoint"] + usable["away_market_midpoint"]
    usable = usable[pair_sum > 0]
    probability = usable["home_market_midpoint"] / (
        usable["home_market_midpoint"] + usable["away_market_midpoint"]
    )
    valid = probability.notna()
    if not valid.any():
        return None
    return {
        "n_games": int(valid.sum()),
        "games_in_season": len(frame),
        "definition": "home midpoint / (home + away midpoint), both sides usable",
        **metrics.summary(
            usable.loc[valid, "home_win"].astype(int), probability[valid]
        ),
    }


def run_pipeline(*, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    frame = load_frame(settings)

    coverage = {
        "games": len(frame),
        "with_crew": int(frame["referee_crew_known"].sum()),
        "coverage": round(float(frame["referee_crew_known"].mean()), 4),
        "by_season": {
            str(s): {
                "games": len(part),
                "with_crew": int(part["referee_crew_known"].sum()),
                "coverage": round(float(part["referee_crew_known"].mean()), 4),
            }
            for s, part in frame.groupby("season")
        },
    }

    k_selection = select_k(settings, frame, BASE_BUNDLES)
    frame = k_selection.pop("frame")
    chosen_k = k_selection["chosen_k"]
    development = run_development(frame, BASE_BUNDLES, chosen_k)
    helped = development["families_that_helped"]
    bundle_f = build_bundle_f(helped)
    if bundle_f is not None:
        extra = run_development(frame, (BASE_BUNDLES[0], bundle_f), chosen_k)
        development["bundle_f"] = extra["families"].get("F")

    # Freeze before the benchmark season is touched.
    enhanced: Bundle | None = None
    c_enhanced = development["control"]["best_c"]
    if bundle_f is not None and development.get("bundle_f"):
        enhanced = bundle_f
        c_enhanced = development["bundle_f"]["best_c"]
    elif helped:
        best = min(helped, key=lambda n: development["families"][n]["delta_vs_control"])
        enhanced = next(b for b in BASE_BUNDLES if b.name == best)
        c_enhanced = development["families"][best]["best_c"]

    frozen = {
        "frozen_at_utc": utc_now().isoformat(),
        "control_bundle": "A (frozen Phase 3A3C)",
        "control_c": development["control"]["best_c"],
        "referee_bundle": enhanced.name if enhanced else None,
        "referee_features": list(enhanced.features[len(BASE_BUNDLES[0].features):])
        if enhanced else [],
        "referee_c": c_enhanced if enhanced else None,
        "shrinkage_k": chosen_k,
        "shrinkage_grid_searched": list(SHRINKAGE_K_GRID),
        "history_policy": "all completed games before the game being predicted",
        "source_rules": (
            "Basketball-Reference box scores, joined on Eastern date plus both "
            "canonical team codes; officials keyed by source slug; no fuzzy "
            "name matching; games without a crew are excluded from state"
        ),
        "preprocessing": "SimpleImputer -> StandardScaler -> LogisticRegression",
    }
    path = settings.paths.reports / "referee_frozen_configuration.json"
    path.write_text(json.dumps(frozen, indent=2, default=str), encoding="utf-8")

    benchmark = run_benchmark(
        frame, BASE_BUNDLES[0], enhanced,
        development["control"]["best_c"], c_enhanced,
    )

    # Diagnostics run strictly after the freeze and can influence nothing.
    predictions = benchmark.pop("_predictions")
    holdout = frame[frame["season"] == HOLDOUT_SEASON]
    context = [c for c in holdout.columns
               if c.startswith("ref_crew_") or c.startswith("avail_")]
    diagnostics = segment_report(
        predictions.merge(
            holdout[["nba_game_id", *context]], on="nba_game_id", how="left"
        )
    )

    report = {
        "generated_at_utc": utc_now().isoformat(),
        "coverage": coverage,
        "development": {
            k: v for k, v in development.items() if k != "fold_table"
        },
        "shrinkage_selection": k_selection,
        "frozen_configuration": frozen,
        "benchmark_2025_26": benchmark,
        "kalshi_t30_reference": kalshi_benchmark(settings),
        "post_freeze_diagnostics": diagnostics,
        "team_referee_interaction": team_referee_support(
            pd.read_parquet(settings.paths.processed / ASSIGNMENTS_FILE)
        ) if (settings.paths.processed / ASSIGNMENTS_FILE).is_file() else None,
        "recommendation": _recommend(development, benchmark),
    }
    out = settings.paths.reports / "referee_model_2025_26.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"] = [str(out), str(path)]
    return report


def _recommend(development: dict[str, Any], benchmark: dict[str, Any]) -> str:
    """INCLUDE, CAPTURE ONLY, or DROP -- decided by the stated bands."""
    if not development["families_that_helped"]:
        # Not DROP. The hypothesis is only partly tested: the position-specific
        # family could not be built at all, because no historical source
        # carries crew chief / referee / umpire. Capture costs about 48
        # requests a day and is the only thing that makes that family testable
        # later, so discarding it would forfeit the untested half of the idea.
        return (
            "CAPTURE ONLY -- no buildable referee family beat the frozen "
            "control consistently on development folds, and the "
            "position-specific family remains untestable for want of a "
            "historical source that records officiating position"
        )
    delta = benchmark.get("delta_brier")
    if delta is None:
        return "CAPTURE ONLY -- nothing earned a benchmark evaluation"
    ci = (benchmark.get("paired_bootstrap_brier") or {}).get("ci_95")
    crosses_zero = ci is None or (ci[0] < 0 < ci[1])
    if -delta >= 0.0005 and not crosses_zero:
        return "INCLUDE -- meaningful and the paired interval excludes zero"
    if -delta > 0:
        return (
            "CAPTURE ONLY -- an improvement too small or too uncertain to act on"
        )
    return "CAPTURE ONLY -- the referee model did not improve on the benchmark"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Referee incremental signal test.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s"
    )
    try:
        report = run_pipeline()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, default=str)[:4000] if args.json
          else json.dumps(report["development"]["families"], indent=2, default=str))
    print("\nRecommendation:", report["recommendation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
