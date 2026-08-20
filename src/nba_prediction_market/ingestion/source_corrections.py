"""Independently verified corrections to BALLDONTLIE source data.

Raw API responses under ``data/raw/`` are never modified. Corrections are
declared here and applied only when building the trusted historical
representation, so "what the source returned" and "what we verified" stay
distinguishable in the processed data.

Two defects are corrected, both systematic rather than random:

**1. Quadruple-overtime games report an impossible tie.** BALLDONTLIE's schema
exposes ``ot1``/``ot2``/``ot3`` and **no ``ot4`` field**. A 4th overtime happens
only when the score is level after the 3rd, so the stored total is the score
through OT3 -- necessarily a tie -- and the deciding period is unrepresentable.
Game 48851 shows the mechanism exactly: OT1 16-16, OT2 7-7, OT3 8-8, total
155-155, actual final 168-161. All four 4OT games in 2006-07..2025-26 are
affected; the 457 games with one to three overtimes are unaffected.

**2. Missing tipoff timestamps.** Thirteen regular-season games have a null
``datetime``: two on 2009-01-22 and an entire eleven-game slate on 2022-12-02.
The dates and scores are correct; only the timestamp is absent.

Every correction was verified against ESPN, which is independent of
BALLDONTLIE. For each, the scores reported by both sources were compared and
matched before the timestamp was accepted, so a correction cannot be attached to
a different game than intended. Corrections additionally carry ``expects``
guards checked at apply time -- see :func:`apply_correction`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

logger = logging.getLogger(__name__)

# Chronology precision levels, ordered from best to worst.
PRECISION_EXACT_DATETIME: Final = "exact_datetime"
PRECISION_DATE_ONLY_VERIFIED: Final = "date_only_verified"
PRECISION_MISSING: Final = "missing"

FIELD_DATETIME: Final = "game_datetime_utc"
FIELD_SCORES: Final = "scores"

ESPN_API: Final = "ESPN public scoreboard API"
ESPN_WEB: Final = "ESPN scoreboard (web)"
LAND_OF_BASKETBALL: Final = "LandOfBasketball box score"


class CorrectionMismatchError(RuntimeError):
    """Raised when a correction's guards do not match the source record.

    This means the upstream data changed under a correction that was written
    against different values. Failing loudly is mandatory: silently applying it
    could attach a verified value to the wrong game.
    """


@dataclass(frozen=True)
class Correction:
    """One independently verified correction to a single source record."""

    nba_game_id: int
    field: str
    original_value: Any
    corrected_value: Any
    precision: str
    reason: str
    source: str
    source_url: str
    note: str
    #: Raw-record fields that must match before the correction is applied.
    #: Identity guards (date, teams) plus the value being replaced.
    expects: Mapping[str, Any] = field(default_factory=dict)
    #: A second, independent confirmation of the same value.
    corroborating_source: str | None = None
    corroborating_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nba_game_id": self.nba_game_id,
            "field": self.field,
            "original_value": _jsonable(self.original_value),
            "corrected_value": _jsonable(self.corrected_value),
            "precision": self.precision,
            "reason": self.reason,
            "source": self.source,
            "source_url": self.source_url,
            "note": self.note,
            "expects": {k: _jsonable(v) for k, v in self.expects.items()},
            "corroborating_source": self.corroborating_source,
            "corroborating_url": self.corroborating_url,
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    return value


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def _timestamp_correction(
    game_id: int,
    day: str,
    away: str,
    home: str,
    away_score: int,
    home_score: int,
    tipoff: str,
    url: str,
) -> Correction:
    """A recovered tipoff for a game whose date and scores already agree."""
    return Correction(
        nba_game_id=game_id,
        field=FIELD_DATETIME,
        original_value=None,
        corrected_value=_utc(tipoff),
        precision=PRECISION_EXACT_DATETIME,
        reason="balldontlie_null_datetime",
        source=ESPN_API,
        source_url=url,
        note=(
            f"ESPN reports tipoff {tipoff} for {away} at {home} on {day}. ESPN's "
            f"final score ({away} {away_score} - {home} {home_score}) matches "
            "BALLDONTLIE's exactly, confirming the same game before the timestamp "
            "was accepted."
        ),
        expects={
            "date": day,
            "away_team": away,
            "home_team": home,
            "away_score": away_score,
            "home_score": home_score,
            FIELD_DATETIME: None,
        },
        corroborating_source=ESPN_WEB,
        corroborating_url=f"https://www.espn.com/nba/scoreboard/_/date/{day.replace('-', '')}",
    )


_ESPN_20090122: Final = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=20090122"
)
_ESPN_20221202: Final = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=20221202"
)


def _score_correction(
    game_id: int,
    day: str,
    away: str,
    home: str,
    tied_at: int,
    final_away: int,
    final_home: int,
    espn_date: str,
    corroborating_url: str,
) -> Correction:
    """A 4OT final recovered after BALLDONTLIE truncated the game at OT3."""
    return Correction(
        nba_game_id=game_id,
        field=FIELD_SCORES,
        original_value=(tied_at, tied_at),
        corrected_value=(final_away, final_home),
        precision="verified_final_score",
        reason="balldontlie_missing_ot4_period",
        source=ESPN_API,
        source_url=(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
            f"scoreboard?dates={espn_date}"
        ),
        note=(
            f"{away} at {home} on {day} went to four overtimes. BALLDONTLIE has no "
            f"ot4 field, so its total stops at the end of OT3 ({tied_at}-{tied_at}, "
            f"necessarily tied). ESPN reports the actual final {away} {final_away} - "
            f"{home} {final_home} with periods=8 (4OT)."
        ),
        expects={
            "date": day,
            "away_team": away,
            "home_team": home,
            "away_score": tied_at,
            "home_score": tied_at,
        },
        corroborating_source=LAND_OF_BASKETBALL,
        corroborating_url=corroborating_url,
    )


#: Every declared correction. Adding one is a deliberate, reviewable edit.
SOURCE_CORRECTIONS: Final[tuple[Correction, ...]] = (
    # --- missing tipoff timestamps, 2009-01-22 (2 games) ---
    _timestamp_correction(
        23784, "2009-01-22", "BOS", "ORL", 90, 80, "2009-01-23T01:00Z", _ESPN_20090122
    ),
    _timestamp_correction(
        21849, "2009-01-22", "WAS", "LAL", 97, 117, "2009-01-23T03:30Z", _ESPN_20090122
    ),
    # --- missing tipoff timestamps, 2022-12-02 (full 11-game slate) ---
    _timestamp_correction(
        857686, "2022-12-02", "WAS", "CHA", 116, 117, "2022-12-03T00:00Z", _ESPN_20221202
    ),
    _timestamp_correction(
        857681, "2022-12-02", "DEN", "ATL", 109, 117, "2022-12-03T00:30Z", _ESPN_20221202
    ),
    _timestamp_correction(
        857680, "2022-12-02", "MIA", "BOS", 120, 116, "2022-12-03T00:30Z", _ESPN_20221202
    ),
    _timestamp_correction(
        857685, "2022-12-02", "TOR", "BKN", 105, 114, "2022-12-03T00:30Z", _ESPN_20221202
    ),
    _timestamp_correction(
        857689, "2022-12-02", "ORL", "CLE", 96, 107, "2022-12-03T00:30Z", _ESPN_20221202
    ),
    _timestamp_correction(
        857683, "2022-12-02", "LAL", "MIL", 133, 129, "2022-12-03T00:30Z", _ESPN_20221202
    ),
    _timestamp_correction(
        857688, "2022-12-02", "PHI", "MEM", 109, 117, "2022-12-03T01:00Z", _ESPN_20221202
    ),
    _timestamp_correction(
        857684, "2022-12-02", "NOP", "SAS", 117, 99, "2022-12-03T01:00Z", _ESPN_20221202
    ),
    _timestamp_correction(
        857682, "2022-12-02", "HOU", "PHX", 122, 121, "2022-12-03T02:00Z", _ESPN_20221202
    ),
    _timestamp_correction(
        857687, "2022-12-02", "IND", "UTA", 119, 139, "2022-12-03T02:00Z", _ESPN_20221202
    ),
    _timestamp_correction(
        857690, "2022-12-02", "CHI", "GSW", 111, 119, "2022-12-03T03:00Z", _ESPN_20221202
    ),
    # --- quadruple-overtime finals truncated at OT3 (4 games) ---
    _score_correction(
        28012, "2012-03-25", "UTA", "ATL", 123, 133, 139, "20120325",
        "https://www.landofbasketball.com/box_scores/2012/0325UTAATL.htm",
    ),
    _score_correction(
        32587, "2015-12-18", "DET", "CHI", 127, 147, 144, "20151218",
        "https://www.landofbasketball.com/box_scores/2015/1218DETCHI.htm",
    ),
    _score_correction(
        34714, "2017-01-29", "NYK", "ATL", 130, 139, 142, "20170129",
        "https://www.landofbasketball.com/box_scores/2017/0129NYKATL.htm",
    ),
    _score_correction(
        48851, "2019-03-01", "CHI", "ATL", 155, 168, 161, "20190301",
        "https://www.landofbasketball.com/box_scores/2019/0301CHIATL.htm",
    ),
)

CORRECTIONS_BY_GAME_ID: Final[dict[int, tuple[Correction, ...]]] = {}
for _correction in SOURCE_CORRECTIONS:
    CORRECTIONS_BY_GAME_ID.setdefault(_correction.nba_game_id, ())
    CORRECTIONS_BY_GAME_ID[_correction.nba_game_id] += (_correction,)


def corrections_for(nba_game_id: Any) -> tuple[Correction, ...]:
    """Declared corrections for one game id, in declaration order."""
    try:
        key = int(nba_game_id)
    except (TypeError, ValueError):
        return ()
    return CORRECTIONS_BY_GAME_ID.get(key, ())


def check_guards(correction: Correction, record: Mapping[str, Any]) -> None:
    """Verify a record matches everything the correction expects.

    Raises :class:`CorrectionMismatchError` on any mismatch. Never returns a
    partial result -- a correction either provably targets this record or it is
    refused.
    """
    for name, expected in correction.expects.items():
        actual = record.get(name)
        if isinstance(expected, str) and not isinstance(actual, str):
            actual = None if actual is None else str(actual)
        if actual != expected:
            raise CorrectionMismatchError(
                f"Correction for game {correction.nba_game_id} "
                f"({correction.field}) expected {name}={expected!r} but the source "
                f"record has {actual!r}. The upstream data changed; re-verify the "
                "correction against its source before proceeding."
            )


# --- application -----------------------------------------------------------

#: Modeling-eligibility exclusion reasons.
EXCLUSION_UNRESOLVED_OUTCOME: Final = "unresolved_outcome"
EXCLUSION_UNUSABLE_CHRONOLOGY: Final = "unusable_chronology"


@dataclass
class CorrectionOutcome:
    """What happened to one record when corrections were applied."""

    datetime_corrected: bool = False
    score_corrected: bool = False
    chronology_precision: str = PRECISION_MISSING
    applied: tuple[Correction, ...] = ()


def apply_corrections(record: dict[str, Any]) -> CorrectionOutcome:
    """Apply every declared correction for ``record``, in place.

    ``record`` must already carry the *source* values under
    ``source_game_datetime_utc`` / ``source_home_score`` / ``source_away_score``
    plus the working fields. Guards are checked first, so a correction written
    against different upstream values fails loudly rather than being applied to
    the wrong game.
    """
    outcome = CorrectionOutcome()
    for correction in corrections_for(record.get("nba_game_id")):
        check_guards(correction, record)

        if correction.field == FIELD_DATETIME:
            if record.get(FIELD_DATETIME) is not None:
                raise CorrectionMismatchError(
                    f"Correction for game {correction.nba_game_id} expected a null "
                    f"{FIELD_DATETIME} but one is present; re-verify it."
                )
            record[FIELD_DATETIME] = correction.corrected_value
            outcome.datetime_corrected = True
        elif correction.field == FIELD_SCORES:
            away, home = correction.corrected_value
            record["away_score"] = away
            record["home_score"] = home
            record["home_win"] = bool(home > away)
            outcome.score_corrected = True
        else:  # pragma: no cover - guarded by the declaration set
            raise CorrectionMismatchError(
                f"Unknown correction field {correction.field!r} for game "
                f"{correction.nba_game_id}"
            )
        outcome.applied += (correction,)

    outcome.chronology_precision = _precision_for(record, outcome)
    return outcome


def _precision_for(record: Mapping[str, Any], outcome: CorrectionOutcome) -> str:
    """Chronology precision after corrections."""
    if record.get(FIELD_DATETIME) is None:
        return PRECISION_MISSING
    for correction in outcome.applied:
        if correction.field == FIELD_DATETIME:
            return correction.precision
    return PRECISION_EXACT_DATETIME


def eligibility(record: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Whether a record may enter modelling, and why not if it may not.

    Two independent requirements, checked explicitly rather than by dropping
    nulls somewhere downstream:

    * a **trusted outcome** -- ``home_win`` must be resolved, and
    * **usable chronology** -- a timestamp precise enough to place the game in a
      team's sequence. ``date_only_verified`` is accepted; ``missing`` is not,
      because the scheduled ``date`` is not a safe substitute for postponed
      games.
    """
    if record.get("home_win") is None:
        return False, EXCLUSION_UNRESOLVED_OUTCOME
    if record.get("chronology_precision") not in {
        PRECISION_EXACT_DATETIME,
        PRECISION_DATE_ONLY_VERIFIED,
    }:
        return False, EXCLUSION_UNUSABLE_CHRONOLOGY
    return True, None
