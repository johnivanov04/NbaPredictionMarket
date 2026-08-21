"""Audited capability and pregame-safety classification of BALLDONTLIE endpoints.

Every entry was verified empirically against the live GOAT-tier API on
2026-08-20, not taken from documentation alone -- two documented claims turned
out to be wrong (see ``doc_discrepancy``).

The safety classification is the part that matters for modelling:

``A`` safe as a historical input **when lagged to prior games**
    The values describe a completed game. They may inform a *later* game's
    features and never the game they came from.
``B`` usable only prospectively
    The endpoint exposes current state with no as-of timestamp, so it can be
    recorded going forward but cannot be reconstructed for past games.
``C`` post-tip or post-game, unsafe for a T-30 prediction of the same game
    Known only at or after tip-off.
``D`` unclear, needs investigation

Class ``B`` and ``C`` sources are **prohibited** as historical T-30 features.
There is no way to recover what was known 30 minutes before a 2014 game from a
feed that only reports today's state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

SAFETY_LAGGED_HISTORICAL: Final = "A_safe_when_lagged"
SAFETY_PROSPECTIVE_ONLY: Final = "B_prospective_only"
SAFETY_POST_TIP: Final = "C_post_tip_unsafe"
SAFETY_UNCLEAR: Final = "D_unclear"


@dataclass(frozen=True)
class EndpointAudit:
    """One endpoint's audited capabilities."""

    name: str
    path: str
    tier: str
    verified_first_season: int | None
    verified_last_season: int | None
    granularity: str
    safety_class: str
    safety_reason: str
    pagination: str = "cursor, per_page max 100"
    doc_discrepancy: str | None = None
    ingested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "tier": self.tier,
            "verified_first_season": self.verified_first_season,
            "verified_last_season": self.verified_last_season,
            "granularity": self.granularity,
            "safety_class": self.safety_class,
            "safety_reason": self.safety_reason,
            "pagination": self.pagination,
            "doc_discrepancy": self.doc_discrepancy,
            "ingested_in_phase_3a3a0": self.ingested,
        }


ENDPOINT_AUDIT: Final[tuple[EndpointAudit, ...]] = (
    EndpointAudit(
        name="game_player_stats",
        path="/v1/stats",
        tier="ALL-STAR",
        verified_first_season=2006,
        verified_last_season=2025,
        granularity="player x game",
        safety_class=SAFETY_LAGGED_HISTORICAL,
        safety_reason=(
            "Box-score output of a completed game. Safe only as a lagged input: a "
            "player's minutes or points in the game being predicted are not known "
            "at T-30 and must never be used for that game."
        ),
        ingested=True,
    ),
    EndpointAudit(
        name="game_advanced_stats_v1",
        path="/v1/stats/advanced",
        tier="GOAT",
        verified_first_season=2006,
        verified_last_season=2025,
        granularity="player x game (no team-level row)",
        safety_class=SAFETY_LAGGED_HISTORICAL,
        safety_reason=(
            "Derived from a completed game's play-by-play. Safe when lagged; the "
            "current game's ratings are a post-game quantity."
        ),
        ingested=True,
    ),
    EndpointAudit(
        name="game_advanced_stats_v2",
        path="/v2/stats/advanced",
        tier="GOAT",
        verified_first_season=2012,
        verified_last_season=2025,
        granularity="player x game x period",
        safety_class=SAFETY_LAGGED_HISTORICAL,
        safety_reason=(
            "Same post-game character as V1, with tracking and hustle detail. Safe "
            "when lagged."
        ),
        doc_discrepancy=(
            "Documentation states 2015 season onward; empirically the feed returns "
            "data from 2012. Tracking fields (speed, distance, touches, passes, "
            "possessions) are populated from 2012; hustle fields (deflections, "
            "contested shots, box-outs) only from 2016."
        ),
        ingested=False,
    ),
    EndpointAudit(
        name="box_scores",
        path="/v1/box_scores",
        tier="GOAT",
        verified_first_season=2006,
        verified_last_season=2025,
        granularity="game",
        safety_class=SAFETY_LAGGED_HISTORICAL,
        safety_reason=(
            "Returns the same game-level fields already ingested in Phase 3A0 "
            "(scores, quarters, status). No additional pregame information."
        ),
        ingested=False,
    ),
    EndpointAudit(
        name="season_averages",
        path="/v1/season_averages/{category}",
        tier="GOAT",
        verified_first_season=1996,
        verified_last_season=2025,
        granularity="player x season (full-season aggregate)",
        safety_class=SAFETY_POST_TIP,
        safety_reason=(
            "A completed season's aggregate. Attaching it to a game inside that "
            "season imports results that had not happened yet -- a direct "
            "lookahead. Usable only if reconstructed game-by-game, which is what "
            "the player-game feed already provides."
        ),
        ingested=False,
    ),
    EndpointAudit(
        name="player_injuries",
        path="/v1/player_injuries",
        tier="ALL-STAR",
        verified_first_season=None,
        verified_last_season=None,
        granularity="player (current state)",
        safety_class=SAFETY_PROSPECTIVE_ONLY,
        safety_reason=(
            "Records carry only player, status, return_date and description -- "
            "there is no as-of timestamp, no history, and no date filter. Verified: "
            "descriptions discuss the 2025-26 season in the past tense, so the feed "
            "is today's state. A current injury must never be attached to an old "
            "game. Usable prospectively by snapshotting it forward."
        ),
        ingested=False,
    ),
    EndpointAudit(
        name="lineups",
        path="/v1/lineups",
        tier="GOAT",
        verified_first_season=2025,
        verified_last_season=2025,
        granularity="player x game (starter flag)",
        safety_class=SAFETY_POST_TIP,
        safety_reason=(
            "Documented as available only once a game begins, and verified to "
            "return zero rows for 2006 and 2024 games while returning 20 rows for a "
            "2025-26 game. Starter identity at T-30 is exactly what it does not "
            "provide. Prohibited as a historical feature."
        ),
        ingested=False,
    ),
)

ENDPOINTS_BY_NAME: Final[dict[str, EndpointAudit]] = {e.name: e for e in ENDPOINT_AUDIT}

#: Endpoints that may never supply a historical T-30 feature, with the reason.
PROHIBITED_FOR_HISTORICAL_FEATURES: Final[dict[str, str]] = {
    e.name: e.safety_reason
    for e in ENDPOINT_AUDIT
    if e.safety_class in {SAFETY_PROSPECTIVE_ONLY, SAFETY_POST_TIP, SAFETY_UNCLEAR}
}

#: V2 metric families and the first season each is actually populated.
V2_METRIC_ERAS: Final[dict[str, int]] = {
    "core_advanced": 2012,
    "tracking_speed_distance_touches_passes": 2012,
    "possessions": 2012,
    "hustle_deflections_contested_boxouts": 2016,
}


def is_prohibited(endpoint_name: str) -> bool:
    """Whether an endpoint may supply a historical T-30 feature."""
    return endpoint_name in PROHIBITED_FOR_HISTORICAL_FEATURES
