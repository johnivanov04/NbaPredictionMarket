"""Predetermined referee ablations.

Fixed before any development result was seen. Each bundle adds exactly one
family on top of the frozen Phase 3A3C control, so the ablation answers "does
*this* referee information help?" rather than the vaguer "do referees matter?".

Bundle D is declared and deliberately left unbuildable. The NBA's own
assignment page carries Crew Chief / Referee / Umpire, but the historical
source does not: Basketball-Reference lists officials alphabetically, and the
league's date-filtered pages are disallowed by its robots.txt. Rather than
quietly dropping the family -- or worse, inventing positions from name order --
it is kept here with its reason recorded, so the gap is visible in the report.
"""

from __future__ import annotations

from typing import Final

from nba_prediction_market.models.availability_bundles import (
    AVAILABILITY_BUNDLES_BY_NAME,
)
from nba_prediction_market.models.bundles import Bundle
from nba_prediction_market.referees.state import FEATURE_FAMILIES

#: The frozen Phase 3A3C selection, unchanged and unrefitted in structure.
#: This is the control every referee bundle is measured against.
CONTROL_FEATURES: Final[tuple[str, ...]] = AVAILABILITY_BUNDLES_BY_NAME["C"].features

WHISTLE_FAMILY: Final[tuple[str, ...]] = FEATURE_FAMILIES["B_whistle_environment"]
HOME_FAMILY: Final[tuple[str, ...]] = FEATURE_FAMILIES["C_home_expectation_adjusted"]
EXPERIENCE_FAMILY: Final[tuple[str, ...]] = FEATURE_FAMILIES["E_experience"]

#: Families that could be built from the available source, by bundle letter.
REFEREE_FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    "B": WHISTLE_FAMILY,
    "C": HOME_FAMILY,
    "E": EXPERIENCE_FAMILY,
}

#: Declared but not buildable, with the reason. Reported, never silently
#: skipped.
UNBUILDABLE_BUNDLES: Final[dict[str, str]] = {
    "D": (
        "crew-chief-specific tendencies require the officiating position, and "
        "no usable historical source carries it. Basketball-Reference lists a "
        "game's officials alphabetically (verified: Cutler, Twardoski, "
        "Williams on 2023-10-24 DEN), so name order is not position order. The "
        "NBA's own page does publish Crew Chief / Referee / Umpire, but only "
        "for the current day, and its date-filtered form is disallowed by "
        "official.nba.com/robots.txt (Disallow: /*?*). stats.nba.com carries "
        "an apparently positional order but blocked this address after a "
        "single request. Prospective capture records positions from now on, so "
        "a future phase can test this family once enough seasons accumulate."
    ),
}

BASE_BUNDLES: Final[tuple[Bundle, ...]] = (
    Bundle("A", "Frozen Phase 3A3C control", CONTROL_FEATURES),
    Bundle("B", "A + crew whistle/environment tendencies",
           CONTROL_FEATURES + WHISTLE_FAMILY),
    Bundle("C", "A + expectation-adjusted home/visitor tendencies",
           CONTROL_FEATURES + HOME_FAMILY),
    Bundle("E", "A + crew experience", CONTROL_FEATURES + EXPERIENCE_FAMILY),
)

BUNDLES_BY_NAME: Final[dict[str, Bundle]] = {b.name: b for b in BASE_BUNDLES}


def build_bundle_f(helped: list[str]) -> Bundle | None:
    """Bundle F: the union of families that independently helped.

    Conditional by construction. If no family beat the control on development
    folds there is nothing to combine, and returning None is the honest result
    rather than assembling a bundle to fit an outcome already seen.
    """
    families: tuple[str, ...] = ()
    for name in sorted(helped):
        families = families + REFEREE_FAMILIES[name]
    if not families:
        return None
    return Bundle(
        "F", f"A + families that individually helped ({', '.join(sorted(helped))})",
        CONTROL_FEATURES + families,
    )
