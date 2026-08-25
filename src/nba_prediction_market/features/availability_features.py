"""Team availability features from official T-30 injury-report states.

Availability only matters in proportion to who is affected: a 35-minute starter
listed OUT is not the same event as a two-way player listed OUT. Every feature
here is therefore a *role-weighted* aggregate, and the role weight comes from
prior games only.

Three rules hold throughout, and the tests pin all three:

* **No current-game information.** A player's role weight for game G is built
  from games strictly before G. Minutes played in G can never define the weight
  used to predict G.
* **No post-anchor reports.** States are selected by the as-of engine at
  ``tipoff - 30 minutes``; nothing later is visible.
* **Absence is not availability.** A player missing from a report is
  ``NOT_REPORTED``, never ``available``. Features are built from explicitly
  reported players, so no assumption about complete roster membership is needed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from nba_prediction_market.features.rotation import PlayerQuality, mean_minutes

#: Statuses the league prints, in ascending order of expected unavailability.
#: The order is documented rather than learned so that "downgrade" has a fixed
#: meaning even before any calibration exists.
STATUS_ORDER: Final[tuple[str, ...]] = (
    "available", "probable", "questionable", "doubtful", "out",
)
STATUS_RANK: Final[dict[str, int]] = {s: i for i, s in enumerate(STATUS_ORDER)}

#: A player absent from the report. Distinct from every reported status.
NOT_REPORTED: Final = "not_reported"
#: A status the league printed that is not in the known vocabulary.
UNKNOWN: Final = "unknown"

#: Games of prior history used for a player's expected-minutes role weight.
ROLE_WINDOW: Final = 10
#: Below this many prior appearances the role weight is shrunk toward zero,
#: so one big game does not make a call-up look like a starter.
ROLE_SHRINKAGE_GAMES: Final = 3.0

#: Raw count features, per team.
COUNT_FEATURES: Final[tuple[str, ...]] = tuple(
    f"{status}_count" for status in STATUS_ORDER
)
#: Role-weighted minute features, per team.
MINUTE_FEATURES: Final[tuple[str, ...]] = tuple(
    f"{status}_expected_minutes" for status in STATUS_ORDER
)


@dataclass
class PlayerRoleState:
    """One team's prior-game player minutes, for role weighting.

    Mirrors the Phase 3A3 rotation state: history is appended only *after* a
    game's features are emitted, so the current game is never in scope.
    """

    history: list[dict[Any, float]] = field(default_factory=list)
    quality: dict[Any, PlayerQuality] = field(default_factory=dict)
    appearances: dict[Any, int] = field(default_factory=dict)

    def record_game(
        self, minutes: Mapping[Any, float],
        plus_minus: Mapping[Any, float] | None = None,
    ) -> None:
        """Absorb a completed game. Call only after emitting that game's row."""
        self.history.append({k: float(v) for k, v in minutes.items() if v})
        for player, played in minutes.items():
            if played and played > 0:
                self.appearances[player] = self.appearances.get(player, 0) + 1
                state = self.quality.setdefault(player, PlayerQuality())
                state.record(played, (plus_minus or {}).get(player))

    def expected_minutes(self, player: Any) -> float | None:
        """Role weight: shrunk mean minutes over the recent window.

        ``None`` when the player has no prior history with this team at all --
        an unknown role is reported as unknown rather than invented as zero,
        because zero would silently claim the player does not matter.
        """
        if not self.history:
            return None
        window = self.history[-ROLE_WINDOW:]
        averages = mean_minutes(window)
        if player not in averages:
            return None
        games = self.appearances.get(player, 0)
        shrink = games / (games + ROLE_SHRINKAGE_GAMES)
        return averages[player] * shrink

    def player_quality(self, player: Any) -> float | None:
        """Lagged shrunk quality rating, or ``None`` with no history."""
        state = self.quality.get(player)
        return None if state is None else state.rating()


@dataclass(frozen=True)
class ReportedPlayer:
    """One player's reported state for one game, with a lagged role weight."""

    player_id: Any
    status: str
    expected_minutes: float | None
    quality: float | None


def team_count_features(players: Sequence[ReportedPlayer]) -> dict[str, float]:
    """Raw per-status counts. The deliberately weak availability baseline."""
    counts = dict.fromkeys(COUNT_FEATURES, 0.0)
    for player in players:
        key = f"{player.status}_count"
        if key in counts:
            counts[key] += 1.0
    return counts


def team_minute_features(players: Sequence[ReportedPlayer]) -> dict[str, float]:
    """Role-weighted minutes per status.

    A player whose role could not be established contributes nothing here.
    That keeps an unknown role from masquerading as a zero-minute player, and
    the count features above still record that he was designated.
    """
    totals = dict.fromkeys(MINUTE_FEATURES, 0.0)
    for player in players:
        key = f"{player.status}_expected_minutes"
        if key in totals and player.expected_minutes is not None:
            totals[key] += player.expected_minutes
    return totals


def expected_minutes_lost(
    players: Sequence[ReportedPlayer], availability: Mapping[str, float]
) -> float:
    """Team expected minutes lost, given a status -> play-rate mapping.

    ``availability[status]`` is the expected share of a player's baseline
    minutes he goes on to play given that status, learned from training games
    only. A status with no mapping contributes nothing rather than a guess.
    """
    total = 0.0
    for player in players:
        if player.expected_minutes is None:
            continue
        multiplier = availability.get(player.status)
        if multiplier is None:
            continue
        total += player.expected_minutes * (1.0 - multiplier)
    return total


def expected_quality_lost(
    players: Sequence[ReportedPlayer], availability: Mapping[str, float]
) -> float:
    """Expected minutes lost weighted by each player's lagged quality.

    Separate from :func:`expected_minutes_lost` on purpose: Phase 3A3 found a
    generic player-quality feature unhelpful, so quality is only ever tested as
    an interaction with an actual availability signal, as its own ablation.
    """
    total = 0.0
    for player in players:
        if player.expected_minutes is None or player.quality is None:
            continue
        multiplier = availability.get(player.status)
        if multiplier is None:
            continue
        total += player.expected_minutes * (1.0 - multiplier) * player.quality
    return total


def status_transitions(
    earlier: Mapping[Any, str] | None,
    later: Mapping[Any, str],
    weights: Mapping[Any, float | None],
) -> dict[str, float] | None:
    """Late-news movement between two anchors.

    Returns ``None`` when the earlier state does not exist. That is the point:
    a team with no earlier filing has *unknown* movement, which is not the same
    claim as "nothing changed", and collapsing the two would invent stability.

    Direction uses the documented :data:`STATUS_ORDER`, so a downgrade means the
    player moved toward unavailability.
    """
    if earlier is None:
        return None
    downgrades = upgrades = 0.0
    newly_out_minutes = 0.0
    for player, new_status in later.items():
        old_status = earlier.get(player)
        if old_status is None or old_status == new_status:
            continue
        old_rank = STATUS_RANK.get(old_status)
        new_rank = STATUS_RANK.get(new_status)
        if old_rank is None or new_rank is None:
            continue
        if new_rank > old_rank:
            downgrades += 1.0
            if new_status == "out":
                newly_out_minutes += weights.get(player) or 0.0
        elif new_rank < old_rank:
            upgrades += 1.0
    return {
        "late_downgrades": downgrades,
        "late_upgrades": upgrades,
        "newly_out_expected_minutes": newly_out_minutes,
    }


def diff(home: float | None, away: float | None) -> float | None:
    """Home-minus-away, or ``None`` if either side is unknown."""
    if home is None or away is None:
        return None
    return home - away
