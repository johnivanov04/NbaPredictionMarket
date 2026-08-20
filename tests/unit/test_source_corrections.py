"""The correction layer: guarded, auditable, and never silent.

Raw BALLDONTLIE payloads are never modified; corrections apply only when the
trusted representation is built, and only when their guards match.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from nba_prediction_market.ingestion.source_corrections import (
    CORRECTIONS_BY_GAME_ID,
    EXCLUSION_UNRESOLVED_OUTCOME,
    EXCLUSION_UNUSABLE_CHRONOLOGY,
    FIELD_DATETIME,
    FIELD_SCORES,
    PRECISION_DATE_ONLY_VERIFIED,
    PRECISION_EXACT_DATETIME,
    PRECISION_MISSING,
    SOURCE_CORRECTIONS,
    CorrectionMismatchError,
    apply_corrections,
    check_guards,
    corrections_for,
    eligibility,
)


def record(**overrides: Any) -> dict[str, Any]:
    """A raw-shaped record as build_history_frame assembles it."""
    base = {
        "nba_game_id": 48851,
        "date": "2019-03-01",
        "away_team": "CHI",
        "home_team": "ATL",
        "source_away_score": 155,
        "source_home_score": 155,
        "away_score": 155,
        "home_score": 155,
        "home_win": None,
        "source_game_datetime_utc": datetime(2019, 3, 2, 0, 30, tzinfo=UTC),
        "game_datetime_utc": datetime(2019, 3, 2, 0, 30, tzinfo=UTC),
    }
    base.update(overrides)
    return base


def timestamp_record(**overrides: Any) -> dict[str, Any]:
    base = {
        "nba_game_id": 857680,
        "date": "2022-12-02",
        "away_team": "MIA",
        "home_team": "BOS",
        "source_away_score": 120,
        "source_home_score": 116,
        "away_score": 120,
        "home_score": 116,
        "home_win": False,
        "source_game_datetime_utc": None,
        "game_datetime_utc": None,
    }
    base.update(overrides)
    return base


# --- declaration integrity -------------------------------------------------


def test_seventeen_corrections_cover_seventeen_distinct_games() -> None:
    assert len(SOURCE_CORRECTIONS) == 17
    assert len(CORRECTIONS_BY_GAME_ID) == 17, "no game may need two corrections"


def test_thirteen_timestamps_and_four_scores() -> None:
    fields = [c.field for c in SOURCE_CORRECTIONS]
    assert fields.count(FIELD_DATETIME) == 13
    assert fields.count(FIELD_SCORES) == 4


@pytest.mark.parametrize("correction", SOURCE_CORRECTIONS, ids=lambda c: str(c.nba_game_id))
def test_every_correction_carries_full_provenance(correction) -> None:
    """A correction without a verifiable source is not auditable."""
    assert correction.source
    assert correction.source_url.startswith("https://")
    assert correction.note
    assert correction.reason
    assert correction.expects, "guards are mandatory"
    # Each was confirmed against a second, independent source.
    assert correction.corroborating_source
    assert correction.corroborating_url.startswith("https://")


@pytest.mark.parametrize("correction", SOURCE_CORRECTIONS, ids=lambda c: str(c.nba_game_id))
def test_every_correction_guards_on_game_identity(correction) -> None:
    """Date plus both teams, so a correction cannot land on another game."""
    assert {"date", "away_team", "home_team"} <= set(correction.expects)


@pytest.mark.parametrize("correction", SOURCE_CORRECTIONS, ids=lambda c: str(c.nba_game_id))
def test_corrections_serialise_for_the_report(correction) -> None:
    d = correction.to_dict()
    assert d["nba_game_id"] == correction.nba_game_id
    assert isinstance(d["source_url"], str)
    assert set(d) >= {"field", "original_value", "corrected_value", "reason", "source", "note"}


def test_lookup_is_tolerant_of_bad_ids() -> None:
    assert corrections_for(None) == ()
    assert corrections_for("nope") == ()
    assert corrections_for(999999999) == ()
    assert len(corrections_for(48851)) == 1


# --- applying a score correction ------------------------------------------


def test_quadruple_overtime_score_is_corrected_and_resolves_the_winner() -> None:
    """155-155 is the score through OT3; the actual final was CHI 168 - ATL 161."""
    row = record()
    outcome = apply_corrections(row)

    assert outcome.score_corrected is True
    assert row["away_score"] == 168
    assert row["home_score"] == 161
    assert row["home_win"] is False, "Chicago won as the away team"
    assert len(outcome.applied) == 1


@pytest.mark.parametrize(
    ("game_id", "day", "away", "home", "tied", "final_away", "final_home", "home_win"),
    [
        (28012, "2012-03-25", "UTA", "ATL", 123, 133, 139, True),
        (32587, "2015-12-18", "DET", "CHI", 127, 147, 144, False),
        (34714, "2017-01-29", "NYK", "ATL", 130, 139, 142, True),
        (48851, "2019-03-01", "CHI", "ATL", 155, 168, 161, False),
    ],
)
def test_all_four_overtime_corrections(
    game_id, day, away, home, tied, final_away, final_home, home_win
) -> None:
    row = record(
        nba_game_id=game_id, date=day, away_team=away, home_team=home,
        source_away_score=tied, source_home_score=tied,
        away_score=tied, home_score=tied, home_win=None,
    )
    apply_corrections(row)

    assert (row["away_score"], row["home_score"]) == (final_away, final_home)
    assert row["home_win"] is home_win
    assert row["home_score"] != row["away_score"], "the tie must be resolved"


def test_raw_source_scores_are_left_untouched() -> None:
    """Provenance: we must always be able to see what BALLDONTLIE returned."""
    row = record()
    apply_corrections(row)

    assert row["source_away_score"] == 155
    assert row["source_home_score"] == 155
    assert row["source_away_score"] == row["source_home_score"]


# --- applying a timestamp correction --------------------------------------


def test_missing_timestamp_is_recovered_as_aware_utc() -> None:
    row = timestamp_record()
    outcome = apply_corrections(row)

    assert outcome.datetime_corrected is True
    assert row["game_datetime_utc"] == datetime(2022, 12, 3, 0, 30, tzinfo=UTC)
    assert row["game_datetime_utc"].tzinfo is not None
    assert row["game_datetime_utc"].utcoffset().total_seconds() == 0
    assert outcome.chronology_precision == PRECISION_EXACT_DATETIME


def test_raw_source_timestamp_stays_null() -> None:
    row = timestamp_record()
    apply_corrections(row)
    assert row["source_game_datetime_utc"] is None


def test_every_timestamp_correction_is_timezone_aware() -> None:
    for correction in SOURCE_CORRECTIONS:
        if correction.field == FIELD_DATETIME:
            assert correction.corrected_value.tzinfo is not None
            assert correction.corrected_value.utcoffset().total_seconds() == 0


# --- guards ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    [
        ("date", "2019-03-02"),
        ("away_team", "MIL"),
        ("home_team", "BOS"),
        ("away_score", 999),
        ("home_score", 999),
    ],
)
def test_a_correction_refuses_a_record_that_does_not_match(field_name, wrong_value) -> None:
    """If upstream changed, applying the correction could corrupt a real result."""
    row = record(**{field_name: wrong_value})
    with pytest.raises(CorrectionMismatchError, match="expected"):
        apply_corrections(row)


def test_guard_failure_names_the_field_and_both_values() -> None:
    row = record(away_team="MIL")
    with pytest.raises(CorrectionMismatchError) as excinfo:
        check_guards(SOURCE_CORRECTIONS[-1], row)
    message = str(excinfo.value)
    assert "away_team" in message and "CHI" in message and "MIL" in message
    assert "re-verify" in message


def test_a_timestamp_correction_refuses_a_record_that_already_has_one() -> None:
    row = timestamp_record(game_datetime_utc=datetime(2022, 12, 3, 12, 0, tzinfo=UTC))
    with pytest.raises(CorrectionMismatchError):
        apply_corrections(row)


def test_an_uncorrected_record_passes_through_unchanged() -> None:
    row = {
        "nba_game_id": 111, "date": "2015-01-01", "away_team": "BOS", "home_team": "NYK",
        "away_score": 100, "home_score": 110, "home_win": True,
        "source_game_datetime_utc": datetime(2015, 1, 2, 0, 0, tzinfo=UTC),
        "game_datetime_utc": datetime(2015, 1, 2, 0, 0, tzinfo=UTC),
    }
    before = dict(row)
    outcome = apply_corrections(row)

    assert outcome.applied == ()
    assert outcome.datetime_corrected is False
    assert outcome.score_corrected is False
    assert {k: row[k] for k in before} == before


# --- eligibility -----------------------------------------------------------


def test_a_corrected_record_becomes_eligible() -> None:
    row = record()
    outcome = apply_corrections(row)
    row["chronology_precision"] = outcome.chronology_precision

    assert eligibility(row) == (True, None)


def test_an_unresolved_outcome_is_explicitly_ineligible() -> None:
    """Not dropped by dropna -- excluded with a stated reason."""
    row = {"home_win": None, "chronology_precision": PRECISION_EXACT_DATETIME}
    assert eligibility(row) == (False, EXCLUSION_UNRESOLVED_OUTCOME)


def test_missing_chronology_is_explicitly_ineligible() -> None:
    row = {"home_win": True, "chronology_precision": PRECISION_MISSING}
    assert eligibility(row) == (False, EXCLUSION_UNUSABLE_CHRONOLOGY)


def test_date_only_precision_is_still_eligible() -> None:
    row = {"home_win": True, "chronology_precision": PRECISION_DATE_ONLY_VERIFIED}
    assert eligibility(row) == (True, None)


def test_eligibility_never_guesses_when_precision_is_absent() -> None:
    assert eligibility({"home_win": True}) == (False, EXCLUSION_UNUSABLE_CHRONOLOGY)
