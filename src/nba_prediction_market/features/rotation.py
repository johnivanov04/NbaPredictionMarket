"""Lagged rotation and player-quality features.

**This is not an injury feature.** BALLDONTLIE's injury feed has no history and
its lineup feed starts at tip-off, so neither can say who is available tonight.
What *can* be said, entirely from completed games, is how a team's minutes have
recently been distributed and whether that distribution has just changed. A
player who stopped appearing three games ago is visible here; a player returning
tonight is not, and must not be.

Definitions, all computed over a team's **prior** games only:

``minutes share``
    A player's minutes in a window divided by all minutes the team played in it.
``HHI``
    Herfindahl index of those shares, ``sum(share^2)``. Near ``1/N`` for an even
    N-man rotation, higher when minutes concentrate.
``overlap(A, B)``
    ``sum over players of min(share_A(p), share_B(p))`` -- the overlap
    coefficient of two minute-share distributions. It is 1.0 when two windows
    distribute minutes identically and 0.0 when they share nobody. Minutes-
    weighted by construction, so losing a starter costs far more than losing a
    fringe player.
``rotation disruption``
    Compares an older *baseline* window against a *recent* one. For each player,
    the shortfall ``max(0, baseline_minutes - recent_minutes)`` is summed,
    weighting large historical roles more simply by using minutes rather than
    counts.

Player quality is deliberately conservative: a per-minute plus-minus shrunk
toward zero by sample size, so a two-minute cameo cannot produce a huge rating.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

#: Windows over which rotation shape is summarised.
ROTATION_WINDOW: Final = 5
#: Baseline and recent windows for the disruption proxy, in games back from now.
DISRUPTION_BASELINE: Final = (10, 4)
DISRUPTION_RECENT: Final = (3, 1)
#: A player must average at least this many minutes in the baseline window to
#: count as a rotation regular whose absence is worth flagging.
HIGH_MINUTES_THRESHOLD: Final = 20.0
#: Games of prior appearances that shrink a player's quality estimate halfway to
#: the league baseline. Small samples are pulled hard toward zero.
QUALITY_SHRINKAGE_GAMES: Final = 20.0

ROTATION_FEATURES: Final[tuple[str, ...]] = (
    "recent_rotation_player_count",
    "recent_rotation_minutes_hhi",
    "top5_recent_minutes_share",
    "top8_recent_minutes_share",
    "last_game_vs_trailing5_minutes_overlap",
    "last3_vs_prior10_minutes_overlap",
    "expected_rotation_minutes_missing",
    "high_minutes_player_absence_count",
    "rotation_disruption_score",
    "expected_rotation_strength",
)


def minute_shares(games: Sequence[Mapping[Any, float]]) -> dict[Any, float]:
    """Player minute shares across a window of prior games."""
    totals: dict[Any, float] = {}
    for game in games:
        for player, minutes in game.items():
            if minutes and minutes > 0:
                totals[player] = totals.get(player, 0.0) + float(minutes)
    grand = sum(totals.values())
    if grand <= 0:
        return {}
    return {player: value / grand for player, value in totals.items()}


def herfindahl(shares: Mapping[Any, float]) -> float | None:
    """Concentration of minute shares. ``None`` when there is nothing to measure."""
    if not shares:
        return None
    return sum(share * share for share in shares.values())


def top_n_share(shares: Mapping[Any, float], n: int) -> float | None:
    if not shares:
        return None
    return sum(sorted(shares.values(), reverse=True)[:n])


def overlap(a: Mapping[Any, float], b: Mapping[Any, float]) -> float | None:
    """Overlap coefficient of two minute-share distributions, in ``[0, 1]``."""
    if not a or not b:
        return None
    return sum(min(a.get(player, 0.0), b.get(player, 0.0)) for player in set(a) | set(b))


def mean_minutes(games: Sequence[Mapping[Any, float]]) -> dict[Any, float]:
    """Average minutes per game for every player appearing in the window.

    A player absent from a game contributes zero minutes to that game, which is
    what makes a disappearance visible rather than invisible.
    """
    if not games:
        return {}
    totals: dict[Any, float] = {}
    for game in games:
        for player, minutes in game.items():
            if minutes and minutes > 0:
                totals[player] = totals.get(player, 0.0) + float(minutes)
    return {player: value / len(games) for player, value in totals.items()}


def _window(history: Sequence[Mapping[Any, float]], start_back: int, end_back: int
            ) -> list[Mapping[Any, float]]:
    """Games from ``start_back`` to ``end_back`` games ago, inclusive.

    ``history`` is chronological, so index ``-1`` is the most recent prior game.
    """
    if end_back < 1 or start_back < end_back:
        raise ValueError(f"invalid window ({start_back}, {end_back})")
    if len(history) < end_back:
        return []
    stop = len(history) - end_back + 1
    start = max(0, len(history) - start_back)
    return list(history[start:stop])


@dataclass
class PlayerQuality:
    """Shrunk per-minute plus-minus for one player, from prior games only."""

    minutes: float = 0.0
    plus_minus: float = 0.0
    games: int = 0

    def record(self, minutes: float | None, plus_minus: float | None) -> None:
        if minutes is None or minutes <= 0:
            return
        self.minutes += float(minutes)
        self.games += 1
        if plus_minus is not None:
            self.plus_minus += float(plus_minus)

    def rating(self, shrinkage_games: float = QUALITY_SHRINKAGE_GAMES) -> float:
        """Plus-minus per 36 minutes, shrunk toward zero by sample size.

        With no history the rating is exactly zero, so an unknown player is
        treated as league-average rather than as a wild outlier.
        """
        if self.minutes <= 0 or self.games == 0:
            return 0.0
        per_36 = 36.0 * self.plus_minus / self.minutes
        weight = self.games / (self.games + shrinkage_games)
        return per_36 * weight


@dataclass
class TeamRotationState:
    """A team's prior-game minute history within a season."""

    history: list[dict[Any, float]] = field(default_factory=list)
    quality: dict[Any, PlayerQuality] = field(default_factory=dict)

    def features(self) -> dict[str, float | None]:
        """Every rotation feature, from prior games only."""
        blank = dict.fromkeys(ROTATION_FEATURES)
        if not self.history:
            return blank

        recent = _window(self.history, ROTATION_WINDOW, 1)
        shares = minute_shares(recent)
        last_game = minute_shares(self.history[-1:])
        trailing5 = minute_shares(_window(self.history, ROTATION_WINDOW + 1, 2)) if len(
            self.history
        ) > 1 else {}
        last3 = minute_shares(_window(self.history, 3, 1))
        prior10 = minute_shares(_window(self.history, 13, 4)) if len(self.history) > 3 else {}

        baseline = mean_minutes(_window(self.history, *DISRUPTION_BASELINE))
        recent_minutes = mean_minutes(_window(self.history, *DISRUPTION_RECENT))

        missing = None
        absences = None
        disruption = None
        if baseline:
            shortfalls = {
                player: max(0.0, value - recent_minutes.get(player, 0.0))
                for player, value in baseline.items()
            }
            missing = sum(shortfalls.values())
            absences = sum(
                1
                for player, value in baseline.items()
                if value >= HIGH_MINUTES_THRESHOLD and recent_minutes.get(player, 0.0) == 0.0
            )
            total_baseline = sum(baseline.values())
            disruption = missing / total_baseline if total_baseline > 0 else None

        strength = None
        if shares:
            strength = sum(
                share * self.quality.get(player, PlayerQuality()).rating()
                for player, share in shares.items()
            )

        return {
            "recent_rotation_player_count": float(len(shares)) if shares else None,
            "recent_rotation_minutes_hhi": herfindahl(shares),
            "top5_recent_minutes_share": top_n_share(shares, 5),
            "top8_recent_minutes_share": top_n_share(shares, 8),
            "last_game_vs_trailing5_minutes_overlap": overlap(last_game, trailing5),
            "last3_vs_prior10_minutes_overlap": overlap(last3, prior10),
            "expected_rotation_minutes_missing": missing,
            "high_minutes_player_absence_count": (
                float(absences) if absences is not None else None
            ),
            "rotation_disruption_score": disruption,
            "expected_rotation_strength": strength,
        }

    def record(self, appearances: Mapping[Any, tuple[float | None, float | None]]) -> None:
        """Fold one completed game in: ``player -> (minutes, plus_minus)``."""
        minutes_only = {
            player: minutes
            for player, (minutes, _) in appearances.items()
            if minutes and minutes > 0
        }
        self.history.append(minutes_only)
        for player, (minutes, plus_minus) in appearances.items():
            self.quality.setdefault(player, PlayerQuality()).record(minutes, plus_minus)
