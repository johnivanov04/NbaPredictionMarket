"""What each official designation actually predicts about participation.

"Questionable" is a word, not a probability. To turn a designation into a
number we need to know how often a player so listed goes on to play, and how
much of his usual workload he carries when he does.

That mapping is *learned*, and learning it uses actual participation -- which
makes it exactly the kind of thing that leaks if handled carelessly. Two rules
keep it safe, and both are tested:

* **Training games only.** A fold's mapping is estimated from the seasons that
  fold trains on. A validation game can never contribute to the mapping applied
  to itself, and the 2025-26 holdout contributes to nothing.
* **Outcome, never input.** Participation is the target of this estimation. It
  is never a feature of the game it came from.

Estimates are shrunk toward the pooled rate so a rare designation with a
handful of observations cannot produce a confident multiplier.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

#: Observations below this count are dominated by the pooled prior.
SHRINKAGE_OBSERVATIONS: Final = 50.0
#: A status seen fewer times than this gets no mapping at all.
MIN_OBSERVATIONS: Final = 20


@dataclass(frozen=True)
class StatusEstimate:
    """Empirical behaviour of one designation, from training games only."""

    status: str
    observations: int
    played: int
    play_rate: float
    mean_minutes: float
    baseline_minutes: float
    minutes_ratio: float
    shrunk_minutes_ratio: float

    @property
    def standard_error(self) -> float:
        """Binomial standard error of the play rate."""
        if self.observations <= 0:
            return 0.0
        p = self.play_rate
        return (p * (1.0 - p) / self.observations) ** 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "observations": self.observations,
            "played": self.played,
            "play_rate": round(self.play_rate, 6),
            "play_rate_stderr": round(self.standard_error, 6),
            "mean_actual_minutes": round(self.mean_minutes, 4),
            "mean_baseline_minutes": round(self.baseline_minutes, 4),
            "minutes_ratio": round(self.minutes_ratio, 6),
            "shrunk_minutes_ratio": round(self.shrunk_minutes_ratio, 6),
        }


@dataclass(frozen=True)
class StatusCalibration:
    """A fold's complete status mapping."""

    estimates: dict[str, StatusEstimate]
    pooled_minutes_ratio: float
    training_seasons: tuple[int, ...]

    def multiplier(self, status: str) -> float | None:
        """Expected share of baseline minutes retained under this status.

        ``None`` for a status with too few observations -- callers must treat
        that as unknown rather than substituting a default, so a thin category
        cannot quietly acquire a confident weight.
        """
        estimate = self.estimates.get(status)
        if estimate is None or estimate.observations < MIN_OBSERVATIONS:
            return None
        return estimate.shrunk_minutes_ratio

    def as_mapping(self) -> dict[str, float]:
        """Every status with a usable multiplier."""
        return {
            status: value
            for status in self.estimates
            if (value := self.multiplier(status)) is not None
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_seasons": list(self.training_seasons),
            "pooled_minutes_ratio": round(self.pooled_minutes_ratio, 6),
            "shrinkage_observations": SHRINKAGE_OBSERVATIONS,
            "min_observations": MIN_OBSERVATIONS,
            "by_status": [e.to_dict() for e in self.estimates.values()],
        }


@dataclass(frozen=True)
class StatusObservation:
    """One designated player in one training game, with what followed."""

    season: int
    status: str
    baseline_minutes: float
    actual_minutes: float

    @property
    def played(self) -> bool:
        return self.actual_minutes > 0


def calibrate(
    observations: Iterable[StatusObservation], training_seasons: Iterable[int]
) -> StatusCalibration:
    """Estimate the status mapping from training observations alone.

    Observations from outside ``training_seasons`` are dropped rather than
    trusted, so a caller that accidentally passes a validation season cannot
    contaminate the fold.
    """
    allowed = tuple(sorted(set(training_seasons)))
    rows = [o for o in observations if o.season in allowed]

    total_actual = sum(o.actual_minutes for o in rows)
    total_baseline = sum(o.baseline_minutes for o in rows)
    pooled = total_actual / total_baseline if total_baseline > 0 else 1.0

    grouped: dict[str, list[StatusObservation]] = {}
    for row in rows:
        grouped.setdefault(row.status, []).append(row)

    estimates: dict[str, StatusEstimate] = {}
    for status, group in sorted(grouped.items()):
        n = len(group)
        played = sum(1 for o in group if o.played)
        actual = sum(o.actual_minutes for o in group)
        baseline = sum(o.baseline_minutes for o in group)
        ratio = actual / baseline if baseline > 0 else 0.0
        weight = n / (n + SHRINKAGE_OBSERVATIONS)
        estimates[status] = StatusEstimate(
            status=status,
            observations=n,
            played=played,
            play_rate=played / n if n else 0.0,
            mean_minutes=actual / n if n else 0.0,
            baseline_minutes=baseline / n if n else 0.0,
            minutes_ratio=ratio,
            shrunk_minutes_ratio=weight * ratio + (1.0 - weight) * pooled,
        )
    return StatusCalibration(estimates, pooled, allowed)


def status_ordering_from(calibration: StatusCalibration) -> list[str]:
    """Statuses ordered by learned availability, most available first.

    Used to check the documented ordering against what the data says rather
    than assuming the two agree.
    """
    scored = [
        (status, estimate.shrunk_minutes_ratio)
        for status, estimate in calibration.estimates.items()
        if estimate.observations >= MIN_OBSERVATIONS
    ]
    return [status for status, _ in sorted(scored, key=lambda x: -x[1])]


def check_ordering(
    calibration: StatusCalibration, documented: Mapping[str, int]
) -> dict[str, Any]:
    """Compare learned ordering against the documented one. Diagnostic only."""
    learned = status_ordering_from(calibration)
    expected = [s for s, _ in sorted(documented.items(), key=lambda x: x[1])
                if s in learned]
    return {
        "learned_order": learned,
        "documented_order": expected,
        "agrees": learned == expected,
    }
