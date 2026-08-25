"""Phase 4A3: forward schedule, Cup TBD handling, refresh, market identity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nba_prediction_market.capture.market_identity import (
    AMBIGUOUS,
    MATCHED,
    NOT_YET_LISTED,
    map_games,
    parse_event_ticker,
    summarise,
)
from nba_prediction_market.capture.schedule import (
    CHANGE_ADDED,
    CHANGE_DISAPPEARED,
    CHANGE_TIPOFF_MOVED,
    FULL_REGULAR_SEASON_GAMES,
    ScheduledGame,
    apply_refresh,
    assess_completeness,
    diff_schedules,
    upcoming,
)
from nba_prediction_market.ingestion.game_phase import (
    GAME_PHASES,
    PHASE_PRESEASON,
    PHASE_REGULAR_SEASON,
)

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
TIP = datetime(2026, 10, 20, 23, 0, tzinfo=UTC)

TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]


def _game(gid, away="PHI", home="NYK", tip=TIP, phase=PHASE_REGULAR_SEASON):
    return ScheduledGame(
        source_game_id=gid, season=2026, phase=phase, tipoff_utc=tip,
        home_team=home, away_team=away, source="test", first_seen_at_utc=NOW,
    )


def _round_robin(games_per_team: int) -> list[ScheduledGame]:
    """A synthetic season where every team plays exactly N games.

    Circle method: one team is fixed and the rest rotate, so each round pairs
    all 30 teams exactly once and N rounds give N games per team.
    """
    games, gid = [], 0
    rotating = TEAMS[1:]
    for rnd in range(games_per_team):
        order = [TEAMS[0], *rotating]
        for i in range(len(order) // 2):
            away, home = order[i], order[-1 - i]
            gid += 1
            games.append(_game(f"g{gid}", away=away, home=home,
                               tip=TIP + timedelta(days=rnd)))
        rotating = [rotating[-1], *rotating[:-1]]
    return games


class TestPreseasonClassification:
    def test_preseason_is_a_declared_phase(self):
        assert PHASE_PRESEASON in GAME_PHASES
        assert PHASE_PRESEASON == "preseason"

    def test_preseason_is_captured_but_counts_toward_nothing(self):
        pre = _game("p1", phase=PHASE_PRESEASON)
        assert pre.is_capturable
        assert not pre.counts_toward_research

    def test_a_regular_season_game_counts_toward_research(self):
        assert _game("r1").counts_toward_research

    def test_preseason_is_excluded_from_research_totals(self):
        games = [_game("r1"), _game("p1", phase=PHASE_PRESEASON)]
        assert sum(g.counts_toward_research for g in games) == 1

    def test_preseason_does_not_count_toward_season_completeness(self):
        # Preseason games must not make an incomplete regular season look full.
        season = _round_robin(80) + [
            _game(f"p{i}", phase=PHASE_PRESEASON) for i in range(60)
        ]
        report = assess_completeness(season, 2026)
        assert report.games == len(_round_robin(80))
        assert report.status == "incomplete_by_design"


class TestCupTbdHandling:
    def test_eighty_games_per_team_is_incomplete_by_design(self):
        report = assess_completeness(_round_robin(80), 2026)
        assert report.min_games_per_team == report.max_games_per_team == 80
        assert report.status == "incomplete_by_design"
        assert report.is_awaiting_cup_assignment
        assert "Emirates NBA Cup" in report.to_dict()["explanation"]

    def test_the_shortfall_is_thirty_games(self):
        report = assess_completeness(_round_robin(80), 2026)
        assert report.unassigned_games == 30
        assert report.expected_games == FULL_REGULAR_SEASON_GAMES

    def test_a_full_season_is_complete(self):
        report = assess_completeness(_round_robin(82), 2026)
        assert report.status == "complete"
        assert report.unassigned_games == 0

    def test_an_uneven_shortfall_is_not_the_cup_pattern(self):
        # A genuinely broken ingest leaves teams on different counts, which
        # must not be excused as "waiting for the Cup".
        games = _round_robin(80)[:-5]
        report = assess_completeness(games, 2026)
        assert report.status == "unexpectedly_incomplete"
        assert "investigate" in report.to_dict()["explanation"]

    def test_incomplete_by_design_is_not_reported_as_corrupt(self):
        assert assess_completeness(_round_robin(80), 2026).status != (
            "unexpectedly_incomplete"
        )


class TestScheduleRefresh:
    def test_a_refresh_is_idempotent(self):
        stored = [_game("a"), _game("b")]
        assert diff_schedules(stored, stored, detected_at_utc=NOW) == []
        assert len(apply_refresh(stored, stored)) == 2

    def test_newly_assigned_cup_games_are_added(self):
        stored = _round_robin(80)
        cup = [_game(f"cup{i}", tip=TIP + timedelta(days=100 + i)) for i in range(30)]
        changes = diff_schedules(stored, stored + cup, detected_at_utc=NOW)
        assert len(changes) == 30
        assert {c.change for c in changes} == {CHANGE_ADDED}
        merged = apply_refresh(stored, stored + cup)
        assert len(merged) == len(stored) + 30

    def test_a_moved_tipoff_is_recorded_not_silently_overwritten(self):
        stored = [_game("a", tip=TIP)]
        moved = [_game("a", tip=TIP + timedelta(hours=3))]
        changes = diff_schedules(stored, moved, detected_at_utc=NOW)
        assert len(changes) == 1
        assert changes[0].change == CHANGE_TIPOFF_MOVED
        payload = changes[0].to_dict()
        assert payload["previous_tipoff_utc"] == TIP
        assert payload["moved_hours"] == pytest.approx(3.0)

    def test_a_moved_tipoff_is_applied(self):
        merged = apply_refresh([_game("a", tip=TIP)],
                               [_game("a", tip=TIP + timedelta(hours=3))])
        assert merged[0].tipoff_utc == TIP + timedelta(hours=3)

    def test_first_seen_survives_a_refresh(self):
        # When a game first appeared is provenance worth keeping.
        earlier = NOW - timedelta(days=30)
        stored = [ScheduledGame("a", 2026, PHASE_REGULAR_SEASON, TIP, "NYK", "PHI",
                                "test", earlier)]
        merged = apply_refresh(stored, [_game("a", tip=TIP + timedelta(hours=1))])
        assert merged[0].first_seen_at_utc == earlier

    def test_a_game_vanishing_from_the_source_is_reported_not_deleted(self):
        stored = [_game("a"), _game("b")]
        changes = diff_schedules(stored, [_game("a")], detected_at_utc=NOW)
        assert [c.change for c in changes] == [CHANGE_DISAPPEARED]
        assert len(apply_refresh(stored, [_game("a")])) == 2

    def test_refresh_output_is_ordered_by_tipoff(self):
        merged = apply_refresh(
            [], [_game("b", tip=TIP + timedelta(days=2)), _game("a", tip=TIP)]
        )
        assert [g.source_game_id for g in merged] == ["a", "b"]


class TestAnchors:
    def test_all_four_anchors_precede_tipoff(self):
        anchors = _game("a").anchors()
        assert set(anchors) == {"T-6h", "T-3h", "T-1h", "T-30m"}
        for value in anchors.values():
            assert value < TIP

    def test_anchor_offsets_are_exact(self):
        anchors = _game("a").anchors()
        assert anchors["T-6h"] == TIP - timedelta(hours=6)
        assert anchors["T-30m"] == TIP - timedelta(minutes=30)

    def test_upcoming_respects_the_horizon_and_ordering(self):
        games = [
            _game("soon", tip=NOW + timedelta(hours=5)),
            _game("later", tip=NOW + timedelta(hours=40)),
            _game("past", tip=NOW - timedelta(hours=1)),
        ]
        result = upcoming(games, NOW, 36.0)
        assert [g.source_game_id for g in result] == ["soon"]

    def test_upcoming_includes_preseason(self):
        games = [_game("p", tip=NOW + timedelta(hours=2), phase=PHASE_PRESEASON)]
        assert len(upcoming(games, NOW, 36.0)) == 1

    def test_a_naive_now_is_refused(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            upcoming([], datetime(2026, 10, 1, 12, 0), 36.0)


class TestMarketIdentity:
    def test_team_codes_are_exactly_three_characters(self):
        # A greedy two-to-four match splits OKCSAS as OKCS + AS and resolves
        # neither; Phase 1 established the width is exactly three.
        assert parse_event_ticker("KXNBAGAME-26OCT20OKCSAS") == (
            "2026-10-20", "OKC", "SAS"
        )
        assert parse_event_ticker("KXNBAGAME-26OCT20PHINYK") == (
            "2026-10-20", "PHI", "NYK"
        )

    def test_a_malformed_ticker_yields_nothing_rather_than_a_guess(self):
        assert parse_event_ticker("KXNBAGAME-BADTICKER") is None
        assert parse_event_ticker("SOMETHINGELSE-26OCT20OKCSAS") is None

    def test_an_unknown_team_code_is_refused(self):
        assert parse_event_ticker("KXNBAGAME-26OCT20ZZZYYY") is None

    def test_a_listed_game_matches(self):
        game = _game("a", away="PHI", home="NYK", tip=TIP)
        mappings = map_games([game], ["KXNBAGAME-26OCT20PHINYK"], now=NOW)
        assert mappings[0].status == MATCHED
        assert mappings[0].event_ticker == "KXNBAGAME-26OCT20PHINYK"

    def test_a_game_months_away_with_no_market_is_informational(self):
        far = _game("a", tip=NOW + timedelta(days=90))
        mapping = map_games([far], [], now=NOW)[0]
        assert mapping.status == NOT_YET_LISTED
        assert mapping.severity(NOW) == "INFO"

    def test_a_missing_market_escalates_near_the_anchor(self):
        soon = _game("a", tip=NOW + timedelta(hours=8))
        assert map_games([soon], [], now=NOW)[0].severity(NOW) == "WARNING"
        urgent = _game("b", tip=NOW + timedelta(hours=3))
        assert map_games([urgent], [], now=NOW)[0].severity(NOW) == "CRITICAL"

    def test_two_tickers_for_one_game_is_ambiguous_not_arbitrary(self):
        # Attaching a quote from the wrong game is the one error that would
        # silently corrupt the dataset.
        game = _game("a", away="PHI", home="NYK", tip=TIP)
        mappings = map_games(
            [game], ["KXNBAGAME-26OCT20PHINYK", "KXNBAGAME-26OCT20PHINYK"], now=NOW
        )
        assert mappings[0].status == AMBIGUOUS
        assert mappings[0].event_ticker is None

    def test_the_summary_separates_urgent_from_normal_absence(self):
        games = [
            _game("far", tip=NOW + timedelta(days=90)),
            _game("near", tip=NOW + timedelta(hours=3)),
        ]
        summary = summarise(map_games(games, [], now=NOW), NOW)
        assert summary["by_status"][NOT_YET_LISTED] == 2
        assert len(summary["urgent_missing_identity"]) == 1
        assert "entirely normal" in summary["note"]


class TestSeasonMetadata:
    def test_the_2026_27_season_is_declared(self):
        from nba_prediction_market.ingestion.season_metadata import SEASON_METADATA

        info = SEASON_METADATA[2026]
        assert info.regular_season_start.isoformat() == "2026-10-20"
        assert info.regular_season_end.isoformat() == "2027-04-11"
        assert info.expected_regular_season_games == FULL_REGULAR_SEASON_GAMES

    def test_the_declaration_records_that_it_is_forward_looking(self):
        from nba_prediction_market.ingestion.season_metadata import SEASON_METADATA

        assert "80 games per team" in SEASON_METADATA[2026].notes
        assert "provisional" in SEASON_METADATA[2026].notes
