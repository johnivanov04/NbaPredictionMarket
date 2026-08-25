"""Phase 3A3C feature bundles.

Each bundle adds exactly one availability family on top of the frozen Phase 3A3
control, so the ablation answers "does *this* availability information help?"
rather than "does availability help?". The bundles are fixed before any result
is seen; nothing is assembled afterwards to fit an outcome.

Bundle G is conditional by construction: it may only be built if both of its
constituent families independently beat the control on development folds.
"""

from __future__ import annotations

from typing import Final

from nba_prediction_market.models.bundles import Bundle

#: The frozen Phase 3A3 bundle D, unchanged. This is the control every
#: availability bundle is measured against.
PHASE_3A3_FEATURES: Final[tuple[str, ...]] = (
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
    "mov_elo_diff",
    "expected_rotation_minutes_missing_diff",
    "high_minutes_player_absence_count_diff",
    "rotation_disruption_score_diff",
)

#: Raw designation counts. Deliberately weak: it knows how many players were
#: listed but nothing about who they were.
RAW_COUNT_FAMILY: Final[tuple[str, ...]] = (
    "avail_out_count_diff",
    "avail_doubtful_count_diff",
    "avail_questionable_count_diff",
    "avail_probable_count_diff",
    "avail_available_count_diff",
)

#: The same designations weighted by each player's lagged expected minutes.
#: This is the primary family: a 35-minute starter listed OUT should not weigh
#: the same as a two-way player listed OUT.
ROLE_MINUTE_FAMILY: Final[tuple[str, ...]] = (
    "avail_out_expected_minutes_diff",
    "avail_doubtful_expected_minutes_diff",
    "avail_questionable_expected_minutes_diff",
    "avail_probable_expected_minutes_diff",
)

#: One number per side, using a fold-specific training-derived status mapping.
EXPECTED_LOSS_FAMILY: Final[tuple[str, ...]] = (
    "avail_home_expected_minutes_lost",
    "avail_away_expected_minutes_lost",
    "avail_expected_minutes_lost_diff",
)

#: Availability interacted with lagged player quality. Phase 3A3 found generic
#: player quality unhelpful; this tests only whether quality matters *given*
#: that a player is actually missing.
QUALITY_LOSS_FAMILY: Final[tuple[str, ...]] = (
    "avail_expected_quality_lost_diff",
)

#: Movement between earlier anchors and T-30. Genuinely new information: it
#: needs intraday report history, which nothing before Phase 3A3B2 had.
LATE_NEWS_FAMILY: Final[tuple[str, ...]] = (
    "avail_newly_out_expected_minutes_3h_diff",
    "avail_newly_out_expected_minutes_1h_diff",
    "avail_expected_loss_change_3h_diff",
    "avail_expected_loss_change_1h_diff",
    "avail_late_downgrades_diff",
    "avail_late_upgrades_diff",
)

AVAILABILITY_FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    "raw_counts": RAW_COUNT_FAMILY,
    "role_minutes": ROLE_MINUTE_FAMILY,
    "expected_loss": EXPECTED_LOSS_FAMILY,
    "quality_loss": QUALITY_LOSS_FAMILY,
    "late_news": LATE_NEWS_FAMILY,
}

AVAILABILITY_BUNDLES: Final[tuple[Bundle, ...]] = (
    Bundle("A", "Phase 3A3 frozen bundle (control)", PHASE_3A3_FEATURES),
    Bundle("B", "A + raw status counts",
           PHASE_3A3_FEATURES + RAW_COUNT_FAMILY),
    Bundle("C", "A + role-weighted status minutes",
           PHASE_3A3_FEATURES + ROLE_MINUTE_FAMILY),
    Bundle("D", "A + training-calibrated expected minutes lost",
           PHASE_3A3_FEATURES + EXPECTED_LOSS_FAMILY),
    Bundle("E", "A + expected minutes lost + quality-weighted loss",
           PHASE_3A3_FEATURES + EXPECTED_LOSS_FAMILY + QUALITY_LOSS_FAMILY),
    Bundle("F", "A + expected minutes lost + late news",
           PHASE_3A3_FEATURES + EXPECTED_LOSS_FAMILY + LATE_NEWS_FAMILY),
)

AVAILABILITY_BUNDLES_BY_NAME: Final[dict[str, Bundle]] = {
    b.name: b for b in AVAILABILITY_BUNDLES
}

#: Every availability feature any bundle can use. Nothing outside this set may
#: reach a model, which is what keeps Kalshi and same-game participation out by
#: construction rather than by vigilance.
ALL_AVAILABILITY_FEATURES: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(f for family in AVAILABILITY_FAMILIES.values() for f in family)
)


def conditional_bundle_g(
    simple_family: str, include_late_news: bool
) -> Bundle | None:
    """Bundle G, buildable only when both constituents earned their place.

    ``None`` when the precondition fails, so the caller cannot quietly fall
    back to combining families that did not independently help.
    """
    if not include_late_news or simple_family not in AVAILABILITY_FAMILIES:
        return None
    if simple_family == "late_news":
        return None
    return Bundle(
        "G",
        f"best simple availability family ({simple_family}) + late news",
        PHASE_3A3_FEATURES
        + AVAILABILITY_FAMILIES[simple_family]
        + LATE_NEWS_FAMILY,
    )
