"""Post-freeze diagnostics and the team x referee sample-size audit.

Everything here runs **after** the configuration is frozen and can therefore
change nothing about it. That ordering is the point: these views exist to
explain a result, and a diagnostic that could still influence selection would
just be an unlogged search over segments.

The team x referee audit is deliberately a sample-size report and nothing
more. "Team X is 12-3 under referee Y" is the archetypal spurious basketball
statistic -- sparse, confounded by which fixtures an official is assigned, and
so numerous that something always looks remarkable. This module counts the
support and stops.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

#: Games one team must have under one official before a team x referee cell
#: could even be discussed. Chosen from the sampling literature's usual floor
#: rather than from anything observed here.
MIN_INTERACTION_GAMES: int = 30


def team_referee_support(assignments: pd.DataFrame) -> dict[str, Any]:
    """How much evidence a team x referee feature would actually have.

    Reports the distribution of cell sizes. It does **not** report win rates,
    best cells, or extremes: producing that table is how a team x referee
    "finding" gets manufactured.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for row in assignments.itertuples():
        slugs = row.referee_slugs
        if slugs is None or len(slugs) == 0:
            continue
        for slug in slugs:
            counts[(row.home_team, slug)] += 1
            counts[(row.away_team, slug)] += 1

    if not counts:
        return {"cells": 0, "note": "no assignments available"}

    sizes = np.array(sorted(counts.values()))
    return {
        "cells": len(counts),
        "median_games_per_cell": float(np.median(sizes)),
        "p90_games_per_cell": float(np.percentile(sizes, 90)),
        "max_games_per_cell": int(sizes.max()),
        "cells_at_or_above_min": int((sizes >= MIN_INTERACTION_GAMES).sum()),
        "share_at_or_above_min": round(
            float((sizes >= MIN_INTERACTION_GAMES).mean()), 4
        ),
        "min_interaction_games": MIN_INTERACTION_GAMES,
        "verdict": (
            "insufficient support for a team x referee feature"
            if float((sizes >= MIN_INTERACTION_GAMES).mean()) < 0.5
            else "support exists; a highly shrunk interaction could be "
                 "predetermined and tested in a later phase"
        ),
    }


def _segment(frame: pd.DataFrame, mask: pd.Series, label: str) -> dict[str, Any]:
    part = frame[mask]
    if part.empty:
        return {"segment": label, "n_games": 0}
    out = {
        "segment": label,
        "n_games": len(part),
        "home_win_rate": round(float(part["home_win"].mean()), 4),
    }
    # A segment's home-win rate on its own is not evidence of a referee effect:
    # crews are not assigned at random, so a high-home-tendency group may simply
    # contain stronger home teams. What matters is whether the control already
    # predicts the difference. The gap, with its standard error, is the honest
    # version of that comparison -- and a gap that is positive in *both* tails
    # is a calibration artefact rather than a monotone referee effect.
    if "control_probability" in part.columns and len(part):
        gap = float(part["home_win"].mean() - part["control_probability"].mean())
        out["control_predicted_home_rate"] = round(
            float(part["control_probability"].mean()), 4
        )
        out["residual_gap"] = round(gap, 4)
        out["residual_gap_stderr"] = round(float(np.sqrt(0.25 / len(part))), 4)
        out["gap_in_stderrs"] = round(gap / np.sqrt(0.25 / len(part)), 2)
    for column, name in (
        ("control_probability", "control"), ("referee_probability", "referee")
    ):
        if column in part.columns:
            losses = (part[column] - part["home_win"]) ** 2
            out[f"{name}_brier"] = round(float(losses.mean()), 5)
    if {"control_brier", "referee_brier"} <= set(out):
        out["delta_brier"] = round(out["referee_brier"] - out["control_brier"], 5)
    return out


def segment_report(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Where a referee-enhanced model differs from the control, if anywhere.

    Requires ``home_win`` plus per-game probabilities. Segments are the ones
    named in the phase brief, fixed in advance, and reported whether or not
    they flatter the result.
    """
    rows: list[dict[str, Any]] = []
    rows.append(_segment(frame, pd.Series(True, index=frame.index), "all games"))

    if "ref_crew_pf_rel" in frame:
        cut = frame["ref_crew_pf_rel"]
        hi = cut >= cut.quantile(0.8)
        lo = cut <= cut.quantile(0.2)
        rows.append(_segment(frame, hi, "high-whistle crews (top quintile)"))
        rows.append(_segment(frame, lo, "low-whistle crews (bottom quintile)"))

    if "ref_crew_home_win_residual" in frame:
        cut = frame["ref_crew_home_win_residual"]
        rows.append(_segment(frame, cut >= cut.quantile(0.8),
                             "high home-tendency crews"))
        rows.append(_segment(frame, cut <= cut.quantile(0.2),
                             "low home-tendency crews"))

    if "ref_crew_experience_mean" in frame:
        cut = frame["ref_crew_experience_mean"]
        rows.append(_segment(frame, cut >= cut.quantile(0.8), "veteran crews"))
        rows.append(_segment(frame, cut <= cut.quantile(0.2), "inexperienced crews"))

    if "control_probability" in frame:
        p = frame["control_probability"]
        rows.append(_segment(frame, (p > 0.45) & (p < 0.55), "close games (0.45-0.55)"))
        rows.append(_segment(frame, p >= 0.65, "home favourites (>=0.65)"))
        rows.append(_segment(frame, p <= 0.35, "home underdogs (<=0.35)"))

    burden = next(
        (c for c in ("avail_expected_minutes_lost_diff",
                     "avail_out_expected_minutes_diff") if c in frame), None
    )
    if burden:
        magnitude = frame[burden].abs()
        rows.append(_segment(frame, magnitude >= magnitude.quantile(0.8),
                             "high availability-burden games"))
    return rows
