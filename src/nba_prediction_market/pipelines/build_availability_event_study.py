"""How fast does Kalshi price official NBA availability news?

This is the part of Phase 4A1 that the archive makes uniquely possible. We hold
the league's own reports with authoritative timestamps, so a status change is
observable as a dated *event* rather than inferred from what happened later.

The event is the **report change itself**, never the player's eventual
participation. Using participation to define an event would be using the
future to pick which moments to study.

Timing discipline is strict, because this is where an event study quietly
cheats:

* a report stamped 17:30 cannot affect a quote at 17:29:59;
* the "immediately after" state is the *first observation that actually
  exists* at or after the stamp, and its true latency is recorded rather than
  smoothed away;
* nothing is interpolated across the news timestamp and then called executable.

Bid and ask are carried alongside the midpoint throughout: a midpoint that
moves is not the same as a price anyone could have traded.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_prediction_market.config import ConfigError, Settings, load_settings
from nba_prediction_market.features.availability_features import STATUS_RANK
from nba_prediction_market.ingestion.raw_store import utc_now

logger = logging.getLogger(__name__)

HOLDOUT_SEASON = 2025

#: How long after the report stamp the market is followed.
HORIZONS_MINUTES: tuple[int, ...] = (5, 15, 30, 60)
#: Only events inside the fetched market window can be studied.
WINDOW_START_MINUTES: int = 360
WINDOW_END_MINUTES: int = 30

#: Role bands, fixed in advance. Never derived from subsequent market movement.
ROLE_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 8.0, "low role"),
    (8.0, 20.0, "medium role"),
    (20.0, 100.0, "high role"),
)

#: Transitions worth studying, each a change in expected availability.
MEANINGFUL_TRANSITIONS: frozenset[tuple[str, str]] = frozenset({
    ("questionable", "out"), ("probable", "out"), ("available", "out"),
    ("doubtful", "out"), ("questionable", "available"), ("out", "available"),
    ("doubtful", "available"), ("questionable", "probable"),
    ("questionable", "doubtful"), ("probable", "questionable"),
    ("available", "questionable"),
})


def role_band(minutes: float | None) -> str:
    if minutes is None or pd.isna(minutes):
        return "unknown role"
    for low, high, label in ROLE_BANDS:
        if low <= minutes < high:
            return label
    return "high role"


@dataclass(frozen=True)
class Event:
    """One official status change, with only pre-event information attached."""

    nba_game_id: Any
    team_code: str
    player_id: Any
    player_name: str
    from_status: str
    to_status: str
    event_ts_utc: pd.Timestamp
    previous_report_ts_utc: pd.Timestamp
    expected_minutes: float | None
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "nba_game_id": self.nba_game_id,
            "team_code": self.team_code,
            "player_id": self.player_id,
            "player_name": self.player_name,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "transition": f"{self.from_status} -> {self.to_status}",
            "event_ts_utc": self.event_ts_utc,
            "previous_report_ts_utc": self.previous_report_ts_utc,
            "expected_minutes": self.expected_minutes,
            "role_band": role_band(self.expected_minutes),
            "direction": self.direction,
        }


def find_events(
    reports: pd.DataFrame, roles: pd.DataFrame, games: pd.DataFrame
) -> list[Event]:
    """Status changes between consecutive reports for the same game and player.

    Role weight comes from the availability observations built for Phase 3A3C,
    which are lagged by construction: they use games strictly before this one.
    """
    role_lookup = {
        (row.nba_game_id, row.player_id): row.baseline_minutes
        for row in roles.itertuples()
    }
    tipoffs = {row.nba_game_id: row.game_datetime_utc for row in games.itertuples()}

    events: list[Event] = []
    keys = ["nba_game_id", "team_code", "balldontlie_player_id"]
    ordered = reports.sort_values([*keys, "report_timestamp_utc"], kind="stable")
    for (game_id, team, player), group in ordered.groupby(keys, sort=False):
        tipoff = tipoffs.get(game_id)
        if tipoff is None:
            continue
        window_start = tipoff - timedelta(minutes=WINDOW_START_MINUTES)
        window_end = tipoff - timedelta(minutes=WINDOW_END_MINUTES)
        rows = list(group.itertuples())
        for earlier, later in pairwise(rows):
            if earlier.status_normalized == later.status_normalized:
                continue
            transition = (earlier.status_normalized, later.status_normalized)
            if transition not in MEANINGFUL_TRANSITIONS:
                continue
            stamp = later.report_timestamp_utc
            if not (window_start <= stamp <= window_end):
                continue
            old_rank = STATUS_RANK.get(earlier.status_normalized)
            new_rank = STATUS_RANK.get(later.status_normalized)
            direction = (
                "downgrade" if (old_rank is not None and new_rank is not None
                                and new_rank > old_rank)
                else "upgrade"
            )
            events.append(Event(
                nba_game_id=game_id,
                team_code=team,
                player_id=player,
                player_name=later.player_name,
                from_status=earlier.status_normalized,
                to_status=later.status_normalized,
                event_ts_utc=stamp,
                previous_report_ts_utc=earlier.report_timestamp_utc,
                expected_minutes=role_lookup.get((game_id, player)),
                direction=direction,
            ))
    return events


def market_around(
    observations: pd.DataFrame, event: Event, home_team: str
) -> dict[str, Any] | None:
    """The affected team's market state before and after the report stamp.

    "Before" is the last observation *strictly earlier* than the stamp; "after"
    is the first at or later than it. Both latencies are recorded, because a
    reaction measured against a stale pre-quote or a late post-quote is not the
    same thing as an immediate one.
    """
    side = "home" if event.team_code == home_team else "away"
    scoped = observations[
        (observations["nba_game_id"] == event.nba_game_id)
        & (observations["side"] == side)
    ].sort_values("observed_at_utc")
    if scoped.empty:
        return None

    stamp = event.event_ts_utc
    before = scoped[scoped["observed_at_utc"] < stamp]
    after = scoped[scoped["observed_at_utc"] >= stamp]
    if before.empty or after.empty:
        return None

    pre = before.iloc[-1]
    post = after.iloc[0]
    result: dict[str, Any] = {
        "side": side,
        "pre_ts_utc": pre["observed_at_utc"],
        "pre_latency_seconds": float(
            (stamp - pre["observed_at_utc"]).total_seconds()
        ),
        "pre_midpoint": pre["midpoint"],
        "pre_bid": pre["yes_bid"],
        "pre_ask": pre["yes_ask"],
        "pre_spread": pre["spread"],
        "post_ts_utc": post["observed_at_utc"],
        "post_latency_seconds": float(
            (post["observed_at_utc"] - stamp).total_seconds()
        ),
        "post_midpoint": post["midpoint"],
        "post_bid": post["yes_bid"],
        "post_ask": post["yes_ask"],
        "post_spread": post["spread"],
    }
    if pre["midpoint"] is not None and post["midpoint"] is not None:
        result["immediate_move"] = float(post["midpoint"] - pre["midpoint"])

    for horizon in HORIZONS_MINUTES:
        target = stamp + timedelta(minutes=horizon)
        window = scoped[scoped["observed_at_utc"] <= target]
        if window.empty:
            continue
        latest = window.iloc[-1]
        # Only count it if an observation actually exists near the horizon;
        # otherwise we would be reporting a stale quote as a timed reaction.
        if (target - latest["observed_at_utc"]).total_seconds() > 300:
            continue
        if latest["midpoint"] is not None and pre["midpoint"] is not None:
            result[f"move_{horizon}m"] = float(latest["midpoint"] - pre["midpoint"])
            result[f"ask_move_{horizon}m"] = (
                float(latest["yes_ask"] - pre["yes_ask"])
                if latest["yes_ask"] is not None and pre["yes_ask"] is not None
                else None
            )
            result[f"spread_{horizon}m"] = latest["spread"]
    return result


def signed_move(move: float | None, direction: str) -> float | None:
    """Move in the direction the news implies for the affected team.

    A downgrade should push the team's probability down, so its sign is
    flipped. Events that move the other way are kept, not discarded -- the
    distribution is the finding.
    """
    if move is None or pd.isna(move):
        return None
    return -move if direction == "downgrade" else move


def summarise_moves(rows: pd.DataFrame, column: str) -> dict[str, Any]:
    values = rows[column].dropna()
    if len(values) < 10:
        return {"n": len(values), "note": "too few to report"}
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "share_positive": float((values > 0).mean()),
    }


def cluster_bootstrap_mean(
    values: np.ndarray, games: np.ndarray, *, n_resamples: int = 10_000,
    seed: int = 20260825, confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap a mean by resampling whole games.

    Several players can change status in the same game, and those events share
    a market. Treating them as independent would overstate precision.
    """
    values = np.asarray(values, dtype=float)
    mask = ~np.isnan(values)
    values, games = values[mask], np.asarray(games)[mask]
    if values.size == 0:
        return {"n": 0}
    unique = pd.unique(games)
    grouped = [values[games == g] for g in unique]
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        picked = rng.integers(0, len(grouped), size=len(grouped))
        means[i] = np.concatenate([grouped[j] for j in picked]).mean()
    tail = (1.0 - confidence) / 2.0
    return {
        "n": int(values.size),
        "n_games": len(unique),
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, tail)),
        "ci_high": float(np.quantile(means, 1.0 - tail)),
        "method": "cluster bootstrap over games",
    }


def reaction_speed(rows: pd.DataFrame) -> dict[str, Any]:
    """What fraction of the 30-minute move is present immediately?

    Computed only where the 30-minute move is large enough for a ratio to
    mean anything; dividing by a near-zero denominator manufactures noise.
    """
    usable = rows[
        rows["signed_move_30m"].notna()
        & rows["signed_immediate_move"].notna()
        & (rows["signed_move_30m"].abs() >= 0.005)
    ]
    if len(usable) < 10:
        return {"n": len(usable), "note": "too few sizeable moves to report"}

    out: dict[str, Any] = {"n": len(usable)}
    for label, column in (
        ("immediate", "signed_immediate_move"),
        ("5m", "signed_move_5m"),
        ("15m", "signed_move_15m"),
    ):
        if column not in usable:
            continue
        share = usable[column] / usable["signed_move_30m"]
        share = share[share.notna()]
        out[f"share_of_30m_move_by_{label}"] = {
            "median": float(share.median()),
            "mean": float(share.clip(-3, 3).mean()),
            "n": len(share),
        }
    return out


def run_pipeline(*, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    settings.paths.ensure()
    processed = settings.paths.processed

    for name in ("nba_official_availability_events_2025_26.parquet",
                 "nba_market_observations_6h_2025_26.parquet",
                 "nba_player_availability_observations_2019_26.parquet"):
        if not (processed / name).is_file():
            raise ConfigError(f"Missing {processed / name}.")

    games = pd.read_parquet(processed / "nba_regular_season_games_2006_26.parquet")
    games = games[
        (games["season"] == HOLDOUT_SEASON) & games["modeling_eligible"]
    ].copy()
    games["game_datetime_utc"] = pd.to_datetime(games["game_datetime_utc"], utc=True)
    home_by_game = {r.nba_game_id: r.home_team for r in games.itertuples()}

    reports = pd.read_parquet(
        processed / "nba_official_availability_events_2025_26.parquet"
    )
    reports["report_timestamp_utc"] = pd.to_datetime(
        reports["report_timestamp_utc"], utc=True
    )
    reports = reports[reports["balldontlie_player_id"].notna()]

    # Attach the trusted game id via the same key the salvage pipeline uses.
    reports["gd"] = pd.to_datetime(
        reports["game_date"], format="%m/%d/%Y", errors="coerce"
    ).dt.date.astype(str)
    games["et_date"] = games["game_datetime_utc"].dt.tz_convert(
        "America/New_York"
    ).dt.date.astype(str)
    key = {
        (r.et_date, r.away_team, r.home_team): r.nba_game_id
        for r in games.itertuples()
    }
    reports["nba_game_id"] = [
        key.get((gd, away, home))
        for gd, away, home in zip(
            reports["gd"], reports["away_team"], reports["home_team"], strict=True
        )
    ]
    reports = reports[reports["nba_game_id"].notna()]

    roles = pd.read_parquet(
        processed / "nba_player_availability_observations_2019_26.parquet"
    )
    roles = roles[roles["season"] == HOLDOUT_SEASON]

    events = find_events(reports, roles, games)
    logger.info("found %d official status-change events in the market window", len(events))

    observations = pd.read_parquet(
        processed / "nba_market_observations_6h_2025_26.parquet"
    )
    observations["observed_at_utc"] = pd.to_datetime(
        observations["observed_at_utc"], utc=True
    )

    rows: list[dict[str, Any]] = []
    for event in events:
        home_team = home_by_game.get(event.nba_game_id)
        if home_team is None:
            continue
        reaction = market_around(observations, event, home_team)
        if reaction is None:
            continue
        row = {**event.to_dict(), **reaction}
        row["signed_immediate_move"] = signed_move(
            reaction.get("immediate_move"), event.direction
        )
        for horizon in HORIZONS_MINUTES:
            row[f"signed_move_{horizon}m"] = signed_move(
                reaction.get(f"move_{horizon}m"), event.direction
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ConfigError("no events with usable market observations")

    move_columns = ["signed_immediate_move"] + [
        f"signed_move_{h}m" for h in HORIZONS_MINUTES
    ]
    report: dict[str, Any] = {
        "generated_at_utc": utc_now().isoformat(),
        "status": "RESEARCH ONLY - descriptive event study",
        "definition": (
            "an event is a change between consecutive official reports for the "
            "same player and game; participation is never used to define one"
        ),
        "window": f"T-{WINDOW_START_MINUTES}m to T-{WINDOW_END_MINUTES}m",
        "events_found": len(events),
        "events_with_market_observations": len(frame),
        "by_transition": (
            frame.groupby("transition").size().sort_values(ascending=False).to_dict()
        ),
        "by_direction": frame.groupby("direction").size().to_dict(),
        "by_role_band": frame.groupby("role_band").size().to_dict(),
        "expected_minutes": {
            "median": float(frame["expected_minutes"].median()),
            "p90": float(frame["expected_minutes"].quantile(0.90)),
        },
        "observation_latency_seconds": {
            "pre_median": float(frame["pre_latency_seconds"].median()),
            "post_median": float(frame["post_latency_seconds"].median()),
            "post_p95": float(frame["post_latency_seconds"].quantile(0.95)),
            "note": (
                "the post-event quote is the first that actually exists at or "
                "after the report stamp; nothing is interpolated across it"
            ),
        },
        "signed_moves_all_events": {
            column: summarise_moves(frame, column) for column in move_columns
        },
        "signed_moves_with_uncertainty": {
            column: cluster_bootstrap_mean(
                frame[column].to_numpy(), frame["nba_game_id"].to_numpy()
            )
            for column in move_columns
        },
        "reaction_speed": reaction_speed(frame),
        "by_role_band_moves": {
            band: {
                column: summarise_moves(subset, column) for column in move_columns
            }
            for band, subset in frame.groupby("role_band")
        },
        "high_role_events_only": {
            column: cluster_bootstrap_mean(
                frame[frame["role_band"] == "high role"][column].to_numpy(),
                frame[frame["role_band"] == "high role"]["nba_game_id"].to_numpy(),
            )
            for column in move_columns
        },
        "executability": {
            "pre_spread_median": float(frame["pre_spread"].median()),
            "post_spread_median": float(frame["post_spread"].median()),
            "spread_widened_after_news": int(
                (frame["post_spread"] > frame["pre_spread"]).sum()
            ),
            "spread_unchanged": int(
                (frame["post_spread"] == frame["pre_spread"]).sum()
            ),
            "events_with_executable_ask_both_sides": int(
                (frame["pre_ask"].notna() & frame["post_ask"].notna()).sum()
            ),
            "note": (
                "a midpoint that moves is not a price anyone could trade; the "
                "ask is carried alongside so reaction and executability are "
                "distinguishable"
            ),
        },
        "leakage_controls": {
            "pre_quote_strictly_before_stamp": bool(
                (frame["pre_ts_utc"] < frame["event_ts_utc"]).all()
            ),
            "post_quote_at_or_after_stamp": bool(
                (frame["post_ts_utc"] >= frame["event_ts_utc"]).all()
            ),
            "no_interpolation_across_the_stamp": True,
            "events_defined_by_report_change_not_participation": True,
        },
    }

    out_path = processed / "nba_availability_market_events_2025_26.parquet"
    frame.to_parquet(out_path, index=False)
    report_path = settings.paths.reports / "availability_market_reaction_2025_26.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["written_files"] = [str(out_path), str(report_path)]
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Phase 4A1 availability event study.")


def main(argv: list[str] | None = None) -> int:
    build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    try:
        report = run_pipeline()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"\nevents: {report['events_found']} found, "
          f"{report['events_with_market_observations']} with market observations")
    print(f"\n{'horizon':14s} {'n':>6s} {'mean':>9s} {'median':>9s} {'>0':>7s} {'95% CI'}")
    for column, stats in report["signed_moves_all_events"].items():
        if "mean" not in stats:
            print(f"{column:14s} {stats['n']:>6d}  {stats.get('note', '')}")
            continue
        ci = report["signed_moves_with_uncertainty"][column]
        print(f"{column:14s} {stats['n']:>6d} {stats['mean']:>+9.5f} "
              f"{stats['median']:>+9.5f} {stats['share_positive']:>6.1%} "
              f"[{ci['ci_low']:+.5f},{ci['ci_high']:+.5f}]")
    print(f"\nreaction speed: {report['reaction_speed']}")
    for path in report["written_files"]:
        print(f"\nWrote {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
