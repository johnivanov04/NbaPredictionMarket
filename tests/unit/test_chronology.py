"""Chronology and rest-day policy: actual played time, never scheduled dates."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from nba_prediction_market.ingestion.chronology import (
    ORDERABLE_PRECISIONS,
    ORDERING_FIELD,
    SCHEDULED_DATE_FIELD,
    ChronologyError,
    rest_days,
    rest_days_for_sequence,
    team_game_sequence,
)
from nba_prediction_market.ingestion.source_corrections import (
    PRECISION_DATE_ONLY_VERIFIED,
    PRECISION_EXACT_DATETIME,
    PRECISION_MISSING,
)


def g(gid: int, tipoff: str | None, *, home: str = "BOS", away: str = "NYK",
      date: str = "2021-01-12") -> dict:
    return {
        "nba_game_id": gid,
        "home_team": home,
        "away_team": away,
        SCHEDULED_DATE_FIELD: date,
        ORDERING_FIELD: (
            datetime.fromisoformat(tipoff.replace("Z", "+00:00")) if tipoff else None
        ),
    }


def test_the_ordering_field_is_the_tipoff_not_the_scheduled_date() -> None:
    assert ORDERING_FIELD == "game_datetime_utc"
    assert SCHEDULED_DATE_FIELD == "date"
    assert ORDERING_FIELD != SCHEDULED_DATE_FIELD


def test_only_verified_precisions_are_orderable() -> None:
    assert PRECISION_EXACT_DATETIME in ORDERABLE_PRECISIONS
    assert PRECISION_DATE_ONLY_VERIFIED in ORDERABLE_PRECISIONS
    assert PRECISION_MISSING not in ORDERABLE_PRECISIONS


# --- rest days -------------------------------------------------------------


def test_back_to_back_is_about_one_day() -> None:
    rest = rest_days(
        datetime(2021, 1, 12, 0, 30, tzinfo=UTC), datetime(2021, 1, 13, 0, 30, tzinfo=UTC)
    )
    assert rest == pytest.approx(1.0)


@pytest.mark.parametrize("days", [1, 2, 3, 7, 30])
def test_rest_scales_with_actual_elapsed_time(days: int) -> None:
    start = datetime(2021, 1, 12, 0, 30, tzinfo=UTC)
    assert rest_days(start, start + timedelta(days=days)) == pytest.approx(float(days))


def test_rest_is_computed_across_timezones_as_real_elapsed_time() -> None:
    eastern = timezone(timedelta(hours=-5))
    previous = datetime(2021, 1, 11, 19, 30, tzinfo=eastern)   # 2021-01-12 00:30 UTC
    following = datetime(2021, 1, 13, 0, 30, tzinfo=UTC)
    assert rest_days(previous, following) == pytest.approx(1.0)


def test_naive_datetimes_are_refused() -> None:
    """A silent zone assumption would shift back-to-backs."""
    with pytest.raises(ChronologyError, match="timezone-aware"):
        rest_days(datetime(2021, 1, 12, 0, 30), datetime(2021, 1, 13, 0, 30, tzinfo=UTC))
    with pytest.raises(ChronologyError, match="timezone-aware"):
        rest_days(datetime(2021, 1, 12, 0, 30, tzinfo=UTC), datetime(2021, 1, 13, 0, 30))


def test_out_of_order_games_are_refused_not_silently_negated() -> None:
    with pytest.raises(ChronologyError, match="precedes"):
        rest_days(datetime(2021, 5, 7, 0, 0, tzinfo=UTC), datetime(2021, 1, 12, 0, 0, tzinfo=UTC))


# --- sequences -------------------------------------------------------------


def test_sequence_is_ordered_by_actual_tipoff_not_input_order() -> None:
    games = [
        g(3, "2021-01-16T00:30Z"),
        g(1, "2021-01-12T00:30Z"),
        g(2, "2021-01-14T00:30Z"),
    ]
    assert [x["nba_game_id"] for x in team_game_sequence(games, "BOS")] == [1, 2, 3]


def test_a_postponed_game_rests_from_its_actual_played_date() -> None:
    """The exact bug this policy prevents.

    A game *scheduled* 2021-01-12 but actually played 2021-05-07 must give a long
    rest before the May game -- not appear as a January back-to-back.
    """
    games = [
        g(1, "2021-01-11T00:30Z", date="2021-01-10"),
        # Scheduled in January, actually played in May.
        g(2, "2021-05-07T23:30Z", date="2021-01-12"),
        g(3, "2021-05-09T23:30Z", date="2021-05-09"),
    ]
    rests = dict(rest_days_for_sequence(games, "BOS"))

    assert rests[1] is None
    assert rests[2] == pytest.approx(116.958, abs=0.01), "rest spans to the played date"
    assert rests[3] == pytest.approx(2.0)


def test_ordering_by_scheduled_date_would_have_been_wrong() -> None:
    """Demonstrates why `date` is banned as a sort key."""
    games = [
        g(1, "2021-01-11T00:30Z", date="2021-01-10"),
        g(2, "2021-05-07T23:30Z", date="2021-01-12"),
        g(3, "2021-02-01T23:30Z", date="2021-02-01"),
    ]
    by_tipoff = [x["nba_game_id"] for x in team_game_sequence(games, "BOS")]
    by_scheduled = [x["nba_game_id"] for x in sorted(games, key=lambda x: x[SCHEDULED_DATE_FIELD])]

    assert by_tipoff == [1, 3, 2]
    assert by_scheduled == [1, 2, 3]
    assert by_tipoff != by_scheduled


def test_a_game_without_a_timestamp_cannot_be_sequenced() -> None:
    """Silently skipping it would change every rest value after it."""
    games = [g(1, "2021-01-12T00:30Z"), g(2, None)]
    with pytest.raises(ChronologyError, match="cannot be placed in a sequence"):
        team_game_sequence(games, "BOS")


def test_the_first_game_has_no_rest_value() -> None:
    rests = rest_days_for_sequence([g(1, "2021-01-12T00:30Z")], "BOS")
    assert rests == [(1, None)]


def test_only_the_requested_team_is_sequenced() -> None:
    games = [
        g(1, "2021-01-12T00:30Z", home="BOS", away="NYK"),
        g(2, "2021-01-13T00:30Z", home="LAL", away="GSW"),
        g(3, "2021-01-14T00:30Z", home="PHI", away="BOS"),
    ]
    assert [x["nba_game_id"] for x in team_game_sequence(games, "BOS")] == [1, 3]


def test_rest_counts_home_and_away_games_alike() -> None:
    games = [
        g(1, "2021-01-12T00:30Z", home="BOS", away="NYK"),
        g(2, "2021-01-13T00:30Z", home="PHI", away="BOS"),
    ]
    rests = dict(rest_days_for_sequence(games, "BOS"))
    assert rests[2] == pytest.approx(1.0)
