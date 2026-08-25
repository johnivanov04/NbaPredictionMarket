"""Broad, deterministic categories for injury-report reason text.

**This is diagnostic only.** No model in Phase 3A3C consumes a reason-derived
feature, and the primary result does not depend on this file existing. The
point is to describe what the reason field contains, not to mine it.

The categories are deliberately coarse and matched by fixed substrings rather
than by anything learned. Reason text is free-form and the league's phrasing
drifts, so a fine-grained taxonomy would be a guess dressed up as data; a row
that does not clearly match is ``other``, never forced into a bucket.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final

INJURY: Final = "injury"
ILLNESS: Final = "illness"
REST: Final = "rest_management"
PERSONAL: Final = "personal"
ASSIGNMENT: Final = "assignment_g_league"
SUSPENSION: Final = "suspension_other"
NOT_GIVEN: Final = "not_given"
OTHER: Final = "other"

CATEGORIES: Final[tuple[str, ...]] = (
    INJURY, ILLNESS, REST, PERSONAL, ASSIGNMENT, SUSPENSION, NOT_GIVEN, OTHER,
)

#: Checked in order; the first match wins. Order matters where phrases overlap
#: -- "G League" outranks "Injury/Illness" because a two-way assignment that
#: also mentions a knock is still fundamentally an assignment.
_RULES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (ASSIGNMENT, ("g league", "on assignment", "two-way", "two- way")),
    (SUSPENSION, ("suspension", "suspended")),
    (PERSONAL, ("personal reasons", "personal", "not with team", "bereavement")),
    (REST, ("rest", "injury management", "return to competition",
            "load management", "reconditioning")),
    (ILLNESS, ("illness;", "; illness", "non-covid", "covid", "flu")),
    (INJURY, ("injury/illness", "soreness", "strain", "sprain", "surgery",
              "fracture", "contusion", "tendon", "tear")),
)


def classify(reason: str | None) -> str:
    """Map one reason string to a broad category.

    Unrecognised text is ``other`` rather than the nearest guess: the whole
    value of this being deterministic is that it never invents a reading.
    """
    if reason is None:
        return NOT_GIVEN
    text = " ".join(str(reason).split()).strip().casefold()
    if not text or text == "-":
        return NOT_GIVEN
    for category, needles in _RULES:
        if any(needle in text for needle in needles):
            return category
    return OTHER


def frequency_table(reasons: Iterable[str | None]) -> dict[str, Any]:
    """Category counts and shares, plus the commonest raw strings."""
    counts: dict[str, int] = dict.fromkeys(CATEGORIES, 0)
    raw: dict[str, int] = {}
    total = 0
    for reason in reasons:
        counts[classify(reason)] += 1
        total += 1
        key = " ".join(str(reason or "").split())
        raw[key] = raw.get(key, 0) + 1
    return {
        "total": total,
        "by_category": {
            category: {
                "count": count,
                "share": round(count / total, 6) if total else 0.0,
            }
            for category, count in counts.items()
        },
        "most_common_raw": [
            {"reason": text, "count": count}
            for text, count in sorted(raw.items(), key=lambda x: -x[1])[:15]
        ],
        "note": (
            "descriptive only; no reason-derived feature enters any Phase 3A3C "
            "model"
        ),
    }
