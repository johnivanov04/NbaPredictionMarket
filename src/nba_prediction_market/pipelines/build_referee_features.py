"""Build leakage-safe referee tendency features, one row per game.

The walk is strictly chronological and the read/write ordering is enforced in
one place, here, so that no caller can accidentally invert it.

One subtlety is worth stating plainly. Sorting by tip-off is not sufficient:
a game tipping at 19:00 is still being played when the 19:30 game starts, so
its result cannot inform the later game's features even though it *started*
earlier. State updates are therefore deferred until a game has plausibly
finished -- see ``ASSUMED_GAME_DURATION_HOURS``. Erring long is the safe
direction: it withholds information rather than leaking it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from typing import Any

import pandas as pd

from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.ingestion.raw_store import utc_now
from nba_prediction_market.referees.state import (
    DEFAULT_K,
    FEATURE_ALLOWLIST,
    GameOutcome,
    RefereeTendencyState,
)

logger = logging.getLogger(__name__)

ASSIGNMENTS_FILE = "nba_referee_assignments_2019_26.parquet"
GAMES_FILE = "nba_regular_season_games_2006_26.parquet"
STRENGTH_FILE = "nba_team_strength_features_2006_26.parquet"
PLAYER_STATS_FILE = "nba_player_game_stats_2006_26.parquet"
OUTPUT_FILE = "nba_referee_features_2019_26.parquet"

#: How long after tip-off a game is treated as finished. NBA games average
#: about 2h15m of wall clock; 3 hours is deliberately generous so that a game
#: never informs another that was already under way.
ASSUMED_GAME_DURATION_HOURS: float = 3.0


def load_inputs(settings: Settings) -> pd.DataFrame:
    """One row per game: crew, outcome measures, and the pregame expectation."""
    processed = settings.paths.processed
    for name in (ASSIGNMENTS_FILE, GAMES_FILE, STRENGTH_FILE, PLAYER_STATS_FILE):
        if not (processed / name).is_file():
            raise ConfigError(f"missing {processed / name}")

    crews = pd.read_parquet(processed / ASSIGNMENTS_FILE)
    games = pd.read_parquet(
        processed / GAMES_FILE,
        columns=["nba_game_id", "season", "game_datetime_utc", "home_team",
                 "away_team", "home_score", "away_score", "home_win",
                 "modeling_eligible"],
    )
    strength = pd.read_parquet(
        processed / STRENGTH_FILE, columns=["nba_game_id", "mov_elo_probability"]
    )
    stats = pd.read_parquet(
        processed / PLAYER_STATS_FILE,
        columns=["nba_game_id", "season", "is_home", "pf", "fta", "pts"],
    )
    stats = stats[stats["season"] >= 2019]

    totals = stats.groupby("nba_game_id").agg(
        total_personal_fouls=("pf", "sum"),
        total_free_throw_attempts=("fta", "sum"),
        total_points=("pts", "sum"),
    )
    sides = stats.groupby(["nba_game_id", "is_home"]).agg(
        pf=("pf", "sum"), fta=("fta", "sum")
    ).unstack("is_home")
    sides.columns = [
        f"{'home' if bool(is_home) else 'away'}_{stat}"
        for stat, is_home in sides.columns
    ]

    frame = (
        games.merge(crews[["nba_game_id", "referee_slugs", "crew_size",
                           "mapping_quality"]], on="nba_game_id", how="left")
        .merge(strength, on="nba_game_id", how="left")
        .merge(totals, on="nba_game_id", how="left")
        .merge(sides, on="nba_game_id", how="left")
    )
    frame["game_datetime_utc"] = pd.to_datetime(
        frame["game_datetime_utc"], utc=True
    )
    return frame.sort_values(
        ["game_datetime_utc", "nba_game_id"], kind="mergesort"
    ).reset_index(drop=True)


def _crew(slugs: Any) -> list[str]:
    """Officials for a game, or an empty crew.

    A left join leaves NaN -- a float -- where no assignment exists, which is
    neither None nor a sized object, so the check has to be on the value's
    shape rather than on identity.
    """
    if slugs is None:
        return []
    if isinstance(slugs, float):  # NaN from an unmatched join
        return []
    return [str(s) for s in slugs]


def _outcome(row: Any) -> GameOutcome | None:
    """A completed game's contribution, or None when anything is missing.

    Missing measures are never imputed. A game we cannot measure simply does
    not update anyone's history.
    """
    values = [
        row.total_personal_fouls, row.total_free_throw_attempts,
        row.total_points, row.home_fta, row.away_fta, row.home_pf, row.away_pf,
        row.mov_elo_probability, row.home_win,
    ]
    if any(pd.isna(v) for v in values):
        return None
    return GameOutcome(
        total_personal_fouls=float(row.total_personal_fouls),
        total_free_throw_attempts=float(row.total_free_throw_attempts),
        total_points=float(row.total_points),
        home_free_throw_attempts=float(row.home_fta),
        away_free_throw_attempts=float(row.away_fta),
        home_personal_fouls=float(row.home_pf),
        away_personal_fouls=float(row.away_pf),
        home_win=int(row.home_win),
        expected_home_win_prob=float(row.mov_elo_probability),
    )


def build_features(frame: pd.DataFrame, *, k: float = DEFAULT_K) -> pd.DataFrame:
    """Walk games in order, reading state before writing it.

    Deferred updates are held in ``pending`` until their game has finished, so
    a game never contributes to another that was already being played.
    """
    state = RefereeTendencyState(k=k)
    pending: list[tuple[pd.Timestamp, list[str], GameOutcome]] = []
    rows: list[dict[str, Any]] = []

    for row in frame.itertuples():
        now = row.game_datetime_utc

        # Flush every game that has finished before this one tipped.
        still_pending = []
        for finish_at, crew, outcome in pending:
            if finish_at <= now:
                state.update(crew, outcome)
            else:
                still_pending.append((finish_at, crew, outcome))
        pending = still_pending

        crew = _crew(row.referee_slugs)

        # READ -- uses only games already folded in above.
        features = state.features_for(crew)

        rows.append({
            "nba_game_id": row.nba_game_id,
            "season": row.season,
            "game_datetime_utc": now,
            "referee_crew_size": len(crew),
            "referee_crew_known": bool(crew),
            "referee_state_games": state.league.games,
            **features,
        })

        # WRITE -- queued, not applied, so it cannot reach its own features.
        #
        # A game with no crew updates nothing, the league baseline included.
        # That keeps the comparison coherent: a referee's mean and the baseline
        # it is measured against are then computed over the same population.
        # At full crew coverage the two choices coincide; while the backfill is
        # incomplete, this one avoids scoring officials against games that
        # could not have been attributed to anyone.
        outcome = _outcome(row)
        if outcome is not None and crew:
            pending.append((
                now + timedelta(hours=ASSUMED_GAME_DURATION_HOURS), crew, outcome
            ))

    result = pd.DataFrame(rows)
    produced = [c for c in result.columns if c.startswith("ref_crew_")]
    if set(produced) != set(FEATURE_ALLOWLIST):
        raise AssertionError(
            f"produced features do not match the allowlist: "
            f"{sorted(set(produced) ^ set(FEATURE_ALLOWLIST))}"
        )
    return result


def run_pipeline(
    *, settings: Settings | None = None, k: float = DEFAULT_K
) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    frame = load_inputs(settings)
    features = build_features(frame, k=k)

    out = settings.paths.processed / OUTPUT_FILE
    features.to_parquet(out, index=False)

    known = features["referee_crew_known"]
    report = {
        "generated_at_utc": utc_now().isoformat(),
        "shrinkage_k": k,
        "games": len(features),
        "games_with_known_crew": int(known.sum()),
        "crew_coverage": round(float(known.mean()), 4),
        "by_season": {
            str(s): {
                "games": len(part),
                "with_crew": int(part["referee_crew_known"].sum()),
            }
            for s, part in features.groupby("season")
        },
        "feature_summary": {
            c: {
                "mean": round(float(features[c].mean()), 5),
                "std": round(float(features[c].std()), 5),
                "min": round(float(features[c].min()), 4),
                "max": round(float(features[c].max()), 4),
            }
            for c in FEATURE_ALLOWLIST
        },
        "written_files": [str(out)],
    }
    path = settings.paths.reports / "referee_features_summary.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"].append(str(path))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build referee features.")
    parser.add_argument("--k", type=float, default=DEFAULT_K)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s"
    )
    try:
        report = run_pipeline(k=args.k)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
