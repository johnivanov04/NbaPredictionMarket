"""Phase 3A4 feature sets: CORE and EXTENDED.

Two sets, fixed before any result is seen, answering two different questions.

**CORE** is the frozen Phase 3A3C allowlist, unchanged. Holding the information
constant while changing only the model class is what makes the comparison mean
"did nonlinearity help?" rather than "did more data help?".

**EXTENDED** adds families that were *already built and audited in earlier
phases and then rejected under linear logistic regression*. Nothing here is new.
The question it answers is narrow and pre-committed: did linearity make
previously weak information look useless? A family that failed linearly might
still carry conditional value -- rotation disruption may matter only when a
high-minute player is out, efficiency only against a particular opponent shape.

It is deliberately **not** "engineer until development improves". No family is
added after seeing a result, and the allowlist is explicit rather than "every
numeric column".
"""

from __future__ import annotations

from typing import Final

from nba_prediction_market.models.availability_bundles import (
    LATE_NEWS_FAMILY,
    PHASE_3A3_FEATURES,
    QUALITY_LOSS_FAMILY,
    ROLE_MINUTE_FAMILY,
)
from nba_prediction_market.models.bundles import Bundle
from nba_prediction_market.models.paid_bundles import (
    EFFICIENCY_FAMILY,
    PLAYER_QUALITY_FAMILY,
    ROSTER_CONTINUITY_FAMILY,
)

#: Exactly the frozen Phase 3A3C bundle C. This is the control's information.
CORE_FEATURES: Final[tuple[str, ...]] = PHASE_3A3_FEATURES + ROLE_MINUTE_FAMILY

#: Families previously constructed, audited, and rejected under logistic
#: regression. Each is listed with the phase that built and then dropped it.
EXTENDED_ADDITIONS: Final[dict[str, tuple[str, ...]]] = {
    # Phase 3A3: helped in isolation but lost to rotation disruption.
    "efficiency_four_factors": EFFICIENCY_FAMILY,
    # Phase 3A3: helped, but not on top of rotation disruption.
    "roster_continuity": ROSTER_CONTINUITY_FAMILY,
    # Phase 3A3: a generic player-quality summary that did not help.
    "player_quality": PLAYER_QUALITY_FAMILY,
    # Phase 3A3C: quality conditioned on a player actually being missing.
    "availability_quality_loss": QUALITY_LOSS_FAMILY,
    # Phase 3A3C: real movement, but redundant once the T-30 level is known.
    "late_news": LATE_NEWS_FAMILY,
}

EXTENDED_FEATURES: Final[tuple[str, ...]] = CORE_FEATURES + tuple(
    dict.fromkeys(f for family in EXTENDED_ADDITIONS.values() for f in family)
)

CORE: Final = Bundle(
    "CORE", "frozen Phase 3A3C allowlist, unchanged", CORE_FEATURES
)
EXTENDED: Final = Bundle(
    "EXTENDED",
    "CORE + families built and rejected under linear logistic regression",
    EXTENDED_FEATURES,
)

FEATURE_SETS: Final[dict[str, Bundle]] = {"CORE": CORE, "EXTENDED": EXTENDED}

#: Nothing outside this may reach a Phase 3A4 model.
ALL_ALLOWED_FEATURES: Final[frozenset[str]] = frozenset(EXTENDED_FEATURES)

#: Substrings that must never appear in any allowlisted feature. Kalshi is a
#: benchmark, never an input; the rest would be same-game outcome leakage.
FORBIDDEN_TOKENS: Final[tuple[str, ...]] = (
    "kalshi", "midpoint", "odds", "vegas", "spread_line", "moneyline",
    "actual_minutes", "did_play", "participation", "starter", "home_win",
    "final_score", "margin_actual", "t15", "t5m",
)
