"""Leakage-safe sequential referee tendency state.

The contract is the whole point of this module: for a game G, every referee
feature is computed from games that **finished before G started**, and G's own
outcome enters the state only afterwards. That ordering is enforced by the API
-- ``features_for`` reads, ``update`` writes, and nothing does both -- and is
covered by tests that mutate a game's result and assert its own features do not
move.

League baselines are sequential for the same reason. A tendency measured
against a final-season league average would be scored against a number that did
not exist yet, which is leakage even though no individual game is misused.

Three things this module deliberately does **not** do:

* **No raw win records.** "Team X is 12-3 under referee Y" is sparse and
  confounded by which games an official is assigned. Tendencies here are
  league-relative, and the home-effect tendency is measured against a frozen
  pregame strength model rather than against nothing.
* **No position features.** Basketball-Reference lists officials
  alphabetically, so crew chief / referee / umpire cannot be recovered. The
  crew is unordered here rather than falsely labelled.
* **No unshrunk estimates.** An official with five games and one with five
  hundred do not get the same weight; see ``shrink``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import pstdev
from typing import Any, Final

#: Development-only shrinkage constants. Small and predetermined: this is the
#: prior weight in games, so 50 means "an official needs 50 games before their
#: own history outweighs the league baseline".
SHRINKAGE_K_GRID: Final[tuple[float, ...]] = (25.0, 50.0, 100.0)

DEFAULT_K: Final = 50.0


def shrink(observed: float, n: int, baseline: float, k: float) -> float:
    """Empirical-Bayes style shrinkage toward a league baseline.

    ``(n / (n + k)) * observed + (k / (n + k)) * baseline``

    With ``n = 0`` this returns the baseline exactly, which is the honest
    answer for an official who has never worked a game: we know nothing about
    them beyond the league.
    """
    if k <= 0:
        raise ValueError("k must be positive; k=0 would trust a single game fully")
    if n < 0:
        raise ValueError("n cannot be negative")
    weight = n / (n + k)
    return weight * observed + (1.0 - weight) * baseline


def shrunk_relative(observed: float, n: int, baseline: float, k: float) -> float:
    """Shrunk tendency expressed as a deviation from the league baseline.

    Algebraically ``(n / (n + k)) * (observed - baseline)``, which is what the
    model actually wants: zero means "indistinguishable from the league", and
    an official with no history is exactly zero rather than some arbitrary
    league-average constant that carries no signal.
    """
    return shrink(observed, n, baseline, k) - baseline


@dataclass(frozen=True)
class GameOutcome:
    """What one completed game contributes to referee and league state.

    ``expected_home_win_prob`` is a *pregame* estimate from the frozen MOV Elo
    built in an earlier phase. It is an input, never something this module
    fits.
    """

    total_personal_fouls: float
    total_free_throw_attempts: float
    total_points: float
    home_free_throw_attempts: float
    away_free_throw_attempts: float
    home_personal_fouls: float
    away_personal_fouls: float
    home_win: int
    expected_home_win_prob: float

    @property
    def home_minus_away_fta(self) -> float:
        return self.home_free_throw_attempts - self.away_free_throw_attempts

    @property
    def home_minus_away_pf(self) -> float:
        return self.home_personal_fouls - self.away_personal_fouls

    @property
    def home_win_residual(self) -> float:
        """Observed home result minus what the strength model expected.

        Positive means the home side won more often than a basketball-only
        model predicted for the games this official worked. It is *not* a claim
        about bias: assignment is not random, and this residual absorbs
        whatever the strength model misses about those particular fixtures.
        """
        return self.home_win - self.expected_home_win_prob


#: The quantities accumulated per official. Kept as an explicit tuple so a new
#: measure cannot be added without also being named in the feature allowlist.
MEASURES: Final[tuple[str, ...]] = (
    "total_personal_fouls",
    "total_free_throw_attempts",
    "total_points",
    "home_minus_away_fta",
    "home_minus_away_pf",
    "home_win_residual",
)


@dataclass
class Accumulator:
    """Running sums for one official, or for the league."""

    games: int = 0
    sums: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(MEASURES, 0.0)
    )

    def add(self, outcome: GameOutcome) -> None:
        self.games += 1
        for measure in MEASURES:
            self.sums[measure] += float(getattr(outcome, measure))

    def mean(self, measure: str) -> float:
        """Zero when nothing has been seen; callers gate on ``games``."""
        return self.sums[measure] / self.games if self.games else 0.0


#: The complete set of referee features. An **allowlist**, not a denylist: a
#: quantity absent from here cannot reach the model even if it is accumulated,
#: and the feature builder asserts the frame it produces matches this exactly.
FEATURE_ALLOWLIST: Final[tuple[str, ...]] = (
    # Family B -- whistle / game environment
    "ref_crew_pf_rel",
    "ref_crew_fta_rel",
    "ref_crew_points_rel",
    "ref_crew_pf_rel_dispersion",
    # Family C -- home/visitor and expectation-adjusted home effect
    "ref_crew_home_fta_diff_rel",
    "ref_crew_home_pf_diff_rel",
    "ref_crew_home_win_residual",
    # Family E -- experience
    "ref_crew_experience_mean",
    "ref_crew_experience_min",
)

#: Which features belong to which predetermined ablation family. Fixed before
#: any result was seen.
FEATURE_FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    "B_whistle_environment": (
        "ref_crew_pf_rel",
        "ref_crew_fta_rel",
        "ref_crew_points_rel",
        "ref_crew_pf_rel_dispersion",
    ),
    "C_home_expectation_adjusted": (
        "ref_crew_home_fta_diff_rel",
        "ref_crew_home_pf_diff_rel",
        "ref_crew_home_win_residual",
    ),
    "E_experience": (
        "ref_crew_experience_mean",
        "ref_crew_experience_min",
    ),
}

#: Maps a feature stem to the accumulated measure behind it.
_STEM_TO_MEASURE: Final[dict[str, str]] = {
    "pf_rel": "total_personal_fouls",
    "fta_rel": "total_free_throw_attempts",
    "points_rel": "total_points",
    "home_fta_diff_rel": "home_minus_away_fta",
    "home_pf_diff_rel": "home_minus_away_pf",
    "home_win_residual": "home_win_residual",
}


@dataclass
class RefereeTendencyState:
    """Sequential state for every official, plus the sequential league baseline.

    Usage is strictly read-then-write::

        features = state.features_for(crew)   # uses only completed games
        ...                                   # game G is played
        state.update(crew, outcome)           # G now informs the future

    Calling ``update`` before ``features_for`` for the same game would leak G
    into its own features. Nothing in this class prevents an out-of-order
    caller, so the feature builder does it in one place and a test pins it.
    """

    k: float = DEFAULT_K
    referees: dict[str, Accumulator] = field(default_factory=dict)
    league: Accumulator = field(default_factory=Accumulator)

    def experience(self, referee_id: str) -> int:
        """Games this official worked before now. Never a career total."""
        acc = self.referees.get(referee_id)
        return acc.games if acc else 0

    def _tendency(self, referee_id: str, measure: str) -> float:
        """One official's shrunk deviation from the league, in natural units."""
        if self.league.games == 0:
            # No baseline exists yet, so no deviation can be measured.
            return 0.0
        acc = self.referees.get(referee_id)
        n = acc.games if acc else 0
        observed = acc.mean(measure) if acc and n else 0.0
        return shrunk_relative(observed, n, self.league.mean(measure), self.k)

    def features_for(self, crew: list[str]) -> dict[str, float]:
        """Crew features from completed games only.

        An empty crew yields all-zero features rather than nulls: "we have no
        officials recorded" is a known state, and the accompanying coverage
        column records it so those games can be excluded explicitly.
        """
        if not crew:
            return dict.fromkeys(FEATURE_ALLOWLIST, 0.0)

        per_official = {
            stem: [self._tendency(r, measure) for r in crew]
            for stem, measure in _STEM_TO_MEASURE.items()
        }
        experience = [self.experience(r) for r in crew]

        features: dict[str, float] = {}
        for stem in ("pf_rel", "fta_rel", "points_rel"):
            features[f"ref_crew_{stem}"] = sum(per_official[stem]) / len(crew)
        for stem in ("home_fta_diff_rel", "home_pf_diff_rel", "home_win_residual"):
            features[f"ref_crew_{stem}"] = sum(per_official[stem]) / len(crew)
        # Dispersion says whether a crew agrees with itself. Population stdev,
        # so a one-official crew is 0 rather than undefined.
        features["ref_crew_pf_rel_dispersion"] = (
            pstdev(per_official["pf_rel"]) if len(crew) > 1 else 0.0
        )
        features["ref_crew_experience_mean"] = sum(experience) / len(crew)
        features["ref_crew_experience_min"] = float(min(experience))

        missing = set(FEATURE_ALLOWLIST) - set(features)
        extra = set(features) - set(FEATURE_ALLOWLIST)
        if missing or extra:
            raise AssertionError(
                f"feature set drifted from the allowlist: missing={missing}, "
                f"extra={extra}"
            )
        return features

    def update(self, crew: list[str], outcome: GameOutcome) -> None:
        """Fold a *completed* game into the league and each official's history."""
        self.league.add(outcome)
        for referee_id in crew:
            self.referees.setdefault(referee_id, Accumulator()).add(outcome)

    def snapshot(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "league_games": self.league.games,
            "distinct_referees": len(self.referees),
            "league_means": {
                m: round(self.league.mean(m), 4) for m in MEASURES
            },
            "experience_distribution": _quantiles(
                [a.games for a in self.referees.values()]
            ),
        }


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def q(p: float) -> float:
        idx = min(len(ordered) - 1, int(p * (len(ordered) - 1)))
        return float(ordered[idx])

    return {
        "n": len(ordered), "min": float(ordered[0]), "p25": q(0.25),
        "median": q(0.5), "p75": q(0.75), "max": float(ordered[-1]),
    }
