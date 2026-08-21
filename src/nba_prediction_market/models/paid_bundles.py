"""Phase 3A3 feature bundles.

Each bundle adds exactly one paid family on top of the frozen Phase 3A2 set, so
the ablation answers "does *this* information help?" rather than "does more data
help?". Bundle F combines only the families that independently earned their
place, decided from development folds.
"""

from __future__ import annotations

from typing import Final

from nba_prediction_market.models.bundles import BUNDLES_BY_NAME, Bundle, validate_bundle

#: Phase 3A2's frozen 12-feature set, the control.
PHASE_3A2_FEATURES: Final[tuple[str, ...]] = BUNDLES_BY_NAME["B"].features

#: Possession-adjusted efficiency and four factors, lagged. Rolling windows are
#: kept short and few: the point is to test the *family*, not to search windows.
EFFICIENCY_FAMILY: Final[tuple[str, ...]] = (
    "season_net_efficiency_diff",
    "last5_net_efficiency_diff",
    "ewma_net_efficiency_diff",
    "season_offensive_efficiency_diff",
    "season_defensive_efficiency_diff",
    "last5_efg_pct_diff",
    "last5_turnover_rate_diff",
    "last5_oreb_rate_diff",
    "last5_ft_rate_diff",
    "season_estimated_pace_diff",
    "offensive_efficiency_vs_league_diff",
    "defensive_efficiency_vs_league_diff",
)

#: How a team's minutes are currently distributed.
ROSTER_CONTINUITY_FAMILY: Final[tuple[str, ...]] = (
    "recent_rotation_player_count_diff",
    "recent_rotation_minutes_hhi_diff",
    "top5_recent_minutes_share_diff",
    "top8_recent_minutes_share_diff",
    "last_game_vs_trailing5_minutes_overlap_diff",
    "last3_vs_prior10_minutes_overlap_diff",
)

#: Whether that distribution has recently changed. Not an injury feature.
ROTATION_DISRUPTION_FAMILY: Final[tuple[str, ...]] = (
    "expected_rotation_minutes_missing_diff",
    "high_minutes_player_absence_count_diff",
    "rotation_disruption_score_diff",
)

#: Shrunk, minutes-weighted player quality.
PLAYER_QUALITY_FAMILY: Final[tuple[str, ...]] = ("expected_rotation_strength_diff",)

PAID_FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    "efficiency": EFFICIENCY_FAMILY,
    "roster_continuity": ROSTER_CONTINUITY_FAMILY,
    "rotation_disruption": ROTATION_DISRUPTION_FAMILY,
    "player_quality": PLAYER_QUALITY_FAMILY,
}

PAID_BUNDLES: Final[tuple[Bundle, ...]] = (
    Bundle("A", "Phase 3A2 frozen bundle (control)", PHASE_3A2_FEATURES),
    Bundle("B", "A + possession-adjusted efficiency / four factors",
           PHASE_3A2_FEATURES + EFFICIENCY_FAMILY),
    Bundle("C", "A + roster continuity",
           PHASE_3A2_FEATURES + ROSTER_CONTINUITY_FAMILY),
    Bundle("D", "A + rotation disruption",
           PHASE_3A2_FEATURES + ROTATION_DISRUPTION_FAMILY),
    Bundle("E", "A + expected rotation strength (player quality)",
           PHASE_3A2_FEATURES + PLAYER_QUALITY_FAMILY),
)

PAID_BUNDLES_BY_NAME: Final[dict[str, Bundle]] = {b.name: b for b in PAID_BUNDLES}


def combined_bundle(family_names: list[str]) -> Bundle:
    """Bundle F: the control plus every family that independently helped."""
    features = PHASE_3A2_FEATURES
    for name in family_names:
        features = features + PAID_FAMILIES[name]
    bundle = Bundle(
        "F",
        "A + families that independently improved development Brier: "
        + (", ".join(family_names) if family_names else "none"),
        features,
    )
    validate_bundle(bundle)
    return bundle


for _bundle in PAID_BUNDLES:
    validate_bundle(_bundle)
