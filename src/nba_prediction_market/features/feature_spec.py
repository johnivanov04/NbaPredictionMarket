"""The exact set of columns a model may consume.

An **allowlist**, deliberately, rather than a denylist of known-bad columns: a
new leaky column added upstream is excluded by default instead of silently
becoming a feature.
"""

from __future__ import annotations

from typing import Final

#: The only columns permitted into a model matrix.
MODEL_FEATURES: Final[tuple[str, ...]] = (
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

#: The supervised target. Never a feature.
TARGET: Final = "home_win"

#: Columns that must never reach a model matrix, asserted in tests. Not the
#: mechanism of exclusion -- the allowlist is -- but a tripwire for mistakes.
FORBIDDEN_SUBSTRINGS: Final[tuple[str, ...]] = (
    "kalshi",
    "score",
    "home_win",
    "source_",
    "_corrected",
    "exclusion_",
    "probability",
    "midpoint",
    "team",
    "franchise",
    "date",
    "ist_stage",
    "phase",
)


def validate_feature_matrix(columns: list[str]) -> None:
    """Raise if a column outside the allowlist, or an obviously leaky one, appears."""
    extra = [c for c in columns if c not in MODEL_FEATURES]
    if extra:
        raise ValueError(
            f"Columns outside the model feature allowlist: {sorted(extra)}. "
            "Add them to MODEL_FEATURES deliberately if they belong."
        )
    for column in columns:
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in column and column in MODEL_FEATURES:
                continue  # allowlisted names may legitimately contain a substring
            if bad in column and column not in MODEL_FEATURES:
                raise ValueError(f"Column {column!r} looks leaky ({bad!r})")
