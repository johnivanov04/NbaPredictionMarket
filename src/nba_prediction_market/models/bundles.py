"""Predetermined feature bundles for the Phase 3A2 ablation.

Bundles are declared up front and nested, so each step answers one question:
does *this* family add anything on top of what came before? Throwing every
feature in at once and reporting the total would not tell us which family earned
its place -- and would invite reading a lucky combination as a discovery.

Each bundle is an explicit allowlist. A column not named here cannot reach a
model matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: Phase 3A1's frozen feature set, reproduced exactly as the control.
BUNDLE_A_BASELINE: Final[tuple[str, ...]] = (
    "elo_diff",
    "win_pct_diff",
    "last5_win_pct_diff",
    "last10_win_pct_diff",
    "last5_point_diff_difference",
    "last10_point_diff_difference",
    "rest_days_diff",
    "home_back_to_back",
    "away_back_to_back",
    "home_games_played",
    "away_games_played",
)

#: Margin-aware Elo, which distinguishes a one-point win from a blowout.
_MOV_ELO: Final[tuple[str, ...]] = ("mov_elo_diff",)

#: Opponent-adjusted strength plus the schedule strength that produced it.
_ADJUSTED_STRENGTH: Final[tuple[str, ...]] = (
    "adjusted_margin_diff",
    "season_sos_diff",
    "last5_sos_diff",
    "last10_sos_diff",
)

#: Points scored/allowed. Not "ratings" -- there are no possessions here.
_SCORING: Final[tuple[str, ...]] = (
    "season_points_scored_diff",
    "season_points_allowed_diff",
    "last5_points_scored_diff",
    "last5_points_allowed_diff",
    "last10_points_scored_diff",
    "last10_points_allowed_diff",
    "ewma_margin_diff",
    "ewma_points_scored_diff",
    "ewma_points_allowed_diff",
    "points_scored_vs_league_diff",
    "points_allowed_vs_league_diff",
)

#: Performance in the venue role each team is about to occupy.
_VENUE_SPLITS: Final[tuple[str, ...]] = (
    "venue_split_win_pct_diff",
    "venue_split_margin_diff",
)

#: Schedule density from actual played chronology.
_FATIGUE: Final[tuple[str, ...]] = (
    "games_last3d_diff",
    "games_last5d_diff",
    "games_last7d_diff",
    "home_3_games_in_4_days",
    "away_3_games_in_4_days",
    "home_4_games_in_6_days",
    "away_4_games_in_6_days",
)


@dataclass(frozen=True)
class Bundle:
    """A named, ordered feature allowlist."""

    name: str
    description: str
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.features)) != len(self.features):
            raise ValueError(f"bundle {self.name} repeats a feature")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "n_features": len(self.features),
            "features": list(self.features),
        }


BUNDLES: Final[tuple[Bundle, ...]] = (
    Bundle("A", "Phase 3A1 baseline (control)", BUNDLE_A_BASELINE),
    Bundle("B", "A + margin-of-victory Elo", BUNDLE_A_BASELINE + _MOV_ELO),
    Bundle(
        "C",
        "B + opponent-adjusted margin and strength of schedule",
        BUNDLE_A_BASELINE + _MOV_ELO + _ADJUSTED_STRENGTH,
    ),
    Bundle(
        "D",
        "C + points scored/allowed features",
        BUNDLE_A_BASELINE + _MOV_ELO + _ADJUSTED_STRENGTH + _SCORING,
    ),
    Bundle(
        "E",
        "D + home/away venue splits",
        BUNDLE_A_BASELINE + _MOV_ELO + _ADJUSTED_STRENGTH + _SCORING + _VENUE_SPLITS,
    ),
    Bundle(
        "F",
        "E + schedule fatigue",
        BUNDLE_A_BASELINE
        + _MOV_ELO
        + _ADJUSTED_STRENGTH
        + _SCORING
        + _VENUE_SPLITS
        + _FATIGUE,
    ),
)

BUNDLES_BY_NAME: Final[dict[str, Bundle]] = {b.name: b for b in BUNDLES}

#: Nothing matching these may appear in any bundle. The allowlist is the
#: mechanism; this is the tripwire.
BANNED_SUBSTRINGS: Final[tuple[str, ...]] = (
    "kalshi", "midpoint", "probability", "home_win", "home_score", "away_score",
    "source_", "_corrected", "exclusion_", "franchise", "team", "date", "phase",
)


def validate_bundle(bundle: Bundle) -> None:
    """Raise if a bundle names an obviously leaky column."""
    for feature in bundle.features:
        for banned in BANNED_SUBSTRINGS:
            if banned in feature:
                raise ValueError(
                    f"bundle {bundle.name} names a leaky-looking feature {feature!r} "
                    f"(matched {banned!r})"
                )
