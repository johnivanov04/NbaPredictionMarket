"""Phase 4A0: fee correctness, execution discipline, research guardrails."""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from nba_prediction_market.models.kalshi_fees import (
    FEE_SCHEDULES,
    FeeScheduleUnavailableError,
    maker_fee,
    schedule_for,
    taker_fee,
    taker_fee_per_contract,
)
from nba_prediction_market.pipelines.build_market_edge_audit import (
    CONTRACT_SIZES,
    DISAGREEMENT_BINS,
    EDGE_THRESHOLDS,
    PRIMARY_SIZE,
    SIDES,
    block_bootstrap_mean,
    simulate_threshold,
)

IN_SEASON = date(2026, 1, 15)


class TestFeeSchedule:
    def test_the_schedule_is_chosen_by_trade_date(self):
        assert schedule_for(date(2025, 11, 1)).stated_effective.endswith("Oct 1, 2025")
        assert schedule_for(date(2026, 3, 1)).stated_effective.endswith("Feb 5, 2026")

    def test_a_date_outside_every_window_raises_rather_than_guessing(self):
        with pytest.raises(FeeScheduleUnavailableError, match="no published"):
            schedule_for(date(2019, 1, 1))

    def test_both_published_versions_carry_the_same_general_formula(self):
        multipliers = {s.taker_multiplier for s in FEE_SCHEDULES}
        assert multipliers == {0.07}
        assert {s.maker_multiplier for s in FEE_SCHEDULES} == {0.0175}

    def test_no_nba_product_specific_exception_is_claimed(self):
        for schedule in FEE_SCHEDULES:
            assert schedule.to_dict()["product_specific_exception_for_nba"] is False

    def test_every_schedule_cites_its_source(self):
        for schedule in FEE_SCHEDULES:
            assert schedule.source_url.startswith("https://web.archive.org/")
            assert "kalshi-fee-schedule.pdf" in schedule.source_url


class TestFeeArithmetic:
    def test_the_published_formula_is_reproduced_exactly(self):
        # 0.07 x 100 x 0.5 x 0.5 = $1.75 exactly; binary floating point lands
        # a hair above and would ceiling to $1.76.
        assert taker_fee(0.50, 100, IN_SEASON) == pytest.approx(1.75)

    def test_fees_round_up_on_the_whole_order_not_per_contract(self):
        one = taker_fee(0.50, 1, IN_SEASON)
        hundred = taker_fee(0.50, 100, IN_SEASON)
        assert one == pytest.approx(0.02)          # 1.75c rounds up to 2c
        assert hundred < 100 * one                  # rounding amortises

    def test_per_contract_cost_falls_with_size(self):
        costs = [taker_fee_per_contract(0.50, c, IN_SEASON) for c in CONTRACT_SIZES]
        assert costs == sorted(costs, reverse=True)

    def test_the_primary_size_is_rounding_stable(self):
        # At 100 contracts the fee equals the exact formula, so the headline
        # view is not distorted by the rounding step.
        exact = 0.07 * PRIMARY_SIZE * 0.5 * 0.5
        assert taker_fee(0.50, PRIMARY_SIZE, IN_SEASON) == pytest.approx(exact)

    def test_the_fee_curve_peaks_at_fifty_cents(self):
        peak = taker_fee(0.50, 100, IN_SEASON)
        for price in (0.1, 0.25, 0.75, 0.9):
            assert taker_fee(price, 100, IN_SEASON) < peak

    def test_maker_fees_are_materially_cheaper_than_taker_fees(self):
        # The published multipliers are 0.07 and 0.0175, a factor of four, but
        # the ratio is not exactly four after each is rounded up to a cent.
        taker = taker_fee(0.50, 100, IN_SEASON)
        maker = maker_fee(0.50, 100, IN_SEASON)
        assert maker < taker
        assert maker == pytest.approx(taker / 4.0, abs=0.01)

    def test_invalid_inputs_are_refused(self):
        with pytest.raises(ValueError, match="contracts must be positive"):
            taker_fee(0.5, 0, IN_SEASON)
        with pytest.raises(ValueError, match="probability in dollars"):
            taker_fee(50.0, 1, IN_SEASON)


class TestExecutionDiscipline:
    def test_entries_use_the_ask_never_the_midpoint(self):
        assert {s.price_column for s in SIDES} == {"home_yes_ask", "away_yes_ask"}
        assert not [s for s in SIDES if "midpoint" in s.price_column]

    def test_both_sides_are_buyable_yes_contracts(self):
        assert {s.name for s in SIDES} == {"HOME", "AWAY"}
        assert {s.win_column for s in SIDES} == {"home_win", "away_win"}


class TestPredeclaredResearchDesign:
    def test_thresholds_are_fixed_and_ascending(self):
        assert EDGE_THRESHOLDS == (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
        assert list(EDGE_THRESHOLDS) == sorted(EDGE_THRESHOLDS)

    def test_disagreement_bins_tile_the_range_without_gaps(self):
        for (_, high, _), (low, _, _) in pairwise(DISAGREEMENT_BINS):
            assert high == low

    def test_bins_are_symmetric_about_zero(self):
        labels = [b[2] for b in DISAGREEMENT_BINS]
        assert labels[0] == "< -10%" and labels[-1] == "> +10%"
        assert len(labels) == 9


class TestSettlementSimulation:
    def _frame(self, ask: float, won: int, n: int = 50) -> pd.DataFrame:
        return pd.DataFrame({
            "best_net_edge": [0.05] * n,
            "best_ask": [ask] * n,
            "best_side": ["HOME"] * n,
            "best_side_won": [won] * n,
            "best_fee_per_contract": [0.0175] * n,
            "trade_date": [date(2026, 1, 10 + i % 5) for i in range(n)],
        })

    def test_a_losing_contract_costs_the_ask_plus_the_fee(self):
        result = simulate_threshold(self._frame(0.40, 0), 0.0, 100)
        per_trade = -(0.40 + 0.0175) * 100
        assert result["mean_pnl_per_trade"] == pytest.approx(per_trade)

    def test_a_winning_contract_pays_one_dollar_less_cost(self):
        result = simulate_threshold(self._frame(0.40, 1), 0.0, 100)
        per_trade = (1.0 - 0.40 - 0.0175) * 100
        assert result["mean_pnl_per_trade"] == pytest.approx(per_trade)

    def test_no_trade_is_forced_when_the_threshold_is_not_cleared(self):
        frame = self._frame(0.40, 1)
        frame["best_net_edge"] = -0.01
        assert simulate_threshold(frame, 0.0, 100)["trades"] == 0

    def test_coverage_is_reported_against_eligible_games(self):
        frame = self._frame(0.40, 1, n=50)
        frame.loc[frame.index[:20], "best_net_edge"] = -0.01
        result = simulate_threshold(frame, 0.0, 100)
        assert result["trades"] == 30
        assert result["coverage"] == pytest.approx(30 / 50)

    def test_fees_are_reported_separately_from_gross(self):
        result = simulate_threshold(self._frame(0.40, 1), 0.0, 100)
        assert result["net_pnl"] == pytest.approx(
            result["gross_settlement_pnl"] - result["fees_paid"]
        )


class TestClusterAwareUncertainty:
    def test_resampling_is_by_day_not_by_game(self):
        values = np.array([1.0] * 50 + [-1.0] * 50)
        blocks = np.array(["d1"] * 50 + ["d2"] * 50)
        result = block_bootstrap_mean(values, blocks, n_resamples=500)
        assert result["n_blocks"] == 2
        assert result["method"] == "block bootstrap over game dates"

    def test_within_block_correlation_widens_the_interval(self):
        # The reason to resample days rather than games: outcomes on the same
        # night are correlated. Here every game in a day shares its day's value,
        # so game-level resampling would see 200 "independent" points when
        # there are really only 20. Block resampling recovers the honest width.
        rng = np.random.default_rng(0)
        per_day = rng.normal(size=20)
        values = np.repeat(per_day, 10)
        days = np.repeat(np.arange(20), 10)

        blocked = block_bootstrap_mean(values, days, n_resamples=2000, seed=3)
        as_if_independent = block_bootstrap_mean(
            values, np.arange(values.size), n_resamples=2000, seed=3
        )
        blocked_width = blocked["ci_high"] - blocked["ci_low"]
        naive_width = as_if_independent["ci_high"] - as_if_independent["ci_low"]
        assert blocked_width > naive_width

    def test_an_empty_series_reports_nothing_rather_than_zero(self):
        assert block_bootstrap_mean(np.array([]), np.array([]))["n"] == 0

    def test_the_interval_is_reproducible(self):
        values = np.random.default_rng(1).normal(size=100)
        blocks = np.repeat(np.arange(20), 5)
        first = block_bootstrap_mean(values, blocks, n_resamples=500, seed=7)
        second = block_bootstrap_mean(values, blocks, n_resamples=500, seed=7)
        assert first["ci_low"] == second["ci_low"]
        assert first["ci_high"] == second["ci_high"]
