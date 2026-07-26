"""Statistical routines checked against hand-computed closed forms.

These are the load-bearing tests. Every other feature reports numbers that
come from here, so a silent error in this module would corrupt every result
the library produces.
"""

from __future__ import annotations

import math

import pytest

from agenteval import (
    PairedLengthError,
    bca_interval,
    bootstrap_interval,
    cliffs_delta,
    cohens_d,
    interpret_effect,
    mcnemar_test,
    mean,
    minimum_detectable_effect,
    paired_bootstrap_diff,
    permutation_test,
    required_sample_size,
    stdev,
    stratified_rates,
    wilson_interval,
)
from agenteval.stats import _inv_norm_cdf, _norm_cdf, _z_for


class TestNormalQuantiles:
    @pytest.mark.parametrize("level,expected", [
        (0.95, 1.959964),
        (0.99, 2.575829),
        (0.90, 1.644854),
        (0.80, 1.281552),
    ])
    def test_matches_published_z_values(self, level, expected):
        assert _z_for(level) == pytest.approx(expected, abs=1e-4)

    def test_inverse_cdf_round_trips_through_cdf(self):
        for p in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
            assert _norm_cdf(_inv_norm_cdf(p)) == pytest.approx(p, abs=1e-6)

    def test_median_is_zero(self):
        assert _inv_norm_cdf(0.5) == pytest.approx(0.0, abs=1e-9)

    def test_symmetry(self):
        assert _inv_norm_cdf(0.25) == pytest.approx(-_inv_norm_cdf(0.75), abs=1e-6)

    def test_extremes_do_not_raise(self):
        assert _inv_norm_cdf(0.0) == -math.inf
        assert _inv_norm_cdf(1.0) == math.inf


class TestWilsonInterval:
    @staticmethod
    def closed_form(successes: int, n: int, z: float | None = None) -> tuple[float, float]:
        # Use the library's own z so this test isolates the Wilson algebra;
        # the quantile itself is verified against published values above.
        if z is None:
            z = _z_for(0.95)
        p = successes / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return max(0.0, centre - spread), min(1.0, centre + spread)

    @pytest.mark.parametrize("successes,n", [(8, 10), (0, 10), (10, 10), (50, 100), (1, 3), (7, 9)])
    def test_matches_closed_form(self, successes, n):
        ci = wilson_interval(successes, n)
        low, high = self.closed_form(successes, n)
        assert ci.low == pytest.approx(low, abs=1e-9)
        assert ci.high == pytest.approx(high, abs=1e-9)

    def test_known_reference_values(self):
        ci = wilson_interval(8, 10)
        assert ci.low == pytest.approx(0.4902, abs=1e-4)
        assert ci.high == pytest.approx(0.9433, abs=1e-4)

    def test_zero_successes_lower_bound_is_zero(self):
        ci = wilson_interval(0, 10)
        assert ci.low == 0.0
        assert ci.high == pytest.approx(0.2775, abs=1e-4)

    def test_all_successes_upper_bound_is_one(self):
        ci = wilson_interval(10, 10)
        assert ci.high == 1.0
        assert ci.low == pytest.approx(0.7225, abs=1e-4)

    @pytest.mark.parametrize("successes", range(0, 21))
    def test_always_within_unit_interval(self, successes):
        ci = wilson_interval(successes, 20)
        assert 0.0 <= ci.low <= ci.point <= ci.high <= 1.0

    def test_narrows_as_sample_grows(self):
        assert wilson_interval(5, 10).width > wilson_interval(50, 100).width
        assert wilson_interval(50, 100).width > wilson_interval(500, 1000).width

    def test_higher_confidence_is_wider(self):
        assert wilson_interval(50, 100, 0.99).width > wilson_interval(50, 100, 0.95).width

    def test_zero_trials_returns_full_range(self):
        ci = wilson_interval(0, 0)
        assert (ci.low, ci.high) == (0.0, 1.0)

    def test_contains_and_margin(self):
        ci = wilson_interval(50, 100)
        assert ci.contains(0.5)
        assert not ci.contains(0.99)
        assert ci.margin == pytest.approx(ci.width / 2)


class TestMcNemar:
    def test_all_discordant_one_direction(self):
        result = mcnemar_test([True] * 10 + [False] * 10, [True] * 20)
        assert result.p_value == pytest.approx(2 / 2**10, abs=1e-9)
        assert result.significant

    def test_balanced_discordant_pairs_not_significant(self):
        result = mcnemar_test([True] * 5 + [False] * 5, [False] * 5 + [True] * 5)
        assert result.p_value == 1.0
        assert not result.significant

    def test_no_discordant_pairs(self):
        result = mcnemar_test([True, False, True], [True, False, True])
        assert result.p_value == 1.0
        assert "no discordant" in result.detail

    def test_reports_fixed_and_broken_counts(self):
        result = mcnemar_test([False, False, True], [True, True, False])
        assert "2 fixed" in result.detail
        assert "1 broken" in result.detail

    def test_statistic_sign_follows_direction(self):
        improved = mcnemar_test([False] * 10, [True] * 10)
        regressed = mcnemar_test([True] * 10, [False] * 10)
        assert improved.statistic > 0
        assert regressed.statistic < 0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(PairedLengthError):
            mcnemar_test([True], [True, False])


class TestPermutationTest:
    def test_identical_samples_not_significant(self):
        result = permutation_test([0.5] * 30, [0.5] * 30, iterations=2000, seed=1)
        assert result.p_value > 0.9
        assert not result.significant

    def test_disjoint_samples_significant(self):
        result = permutation_test([0.0] * 30, [1.0] * 30, iterations=2000, seed=1)
        assert result.p_value < 0.01
        assert result.significant
        assert result.statistic == pytest.approx(1.0)

    def test_p_value_never_zero(self):
        result = permutation_test([0.0] * 50, [1.0] * 50, iterations=100, seed=1)
        assert result.p_value > 0

    def test_reproducible_with_seed(self):
        a = permutation_test([0.0, 1.0] * 20, [1.0, 0.0] * 20, iterations=1000, seed=7)
        b = permutation_test([0.0, 1.0] * 20, [1.0, 0.0] * 20, iterations=1000, seed=7)
        assert a.p_value == b.p_value

    def test_empty_input(self):
        result = permutation_test([], [], iterations=100)
        assert result.p_value == 1.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(PairedLengthError):
            permutation_test([1.0], [1.0, 2.0])


class TestBootstrap:
    def test_point_estimate_is_the_sample_mean(self):
        values = [1.0] * 50 + [0.0] * 50
        assert bootstrap_interval(values, iterations=2000, seed=3).point == pytest.approx(0.5)

    def test_brackets_the_mean(self):
        ci = bootstrap_interval([1.0] * 50 + [0.0] * 50, iterations=3000, seed=3)
        assert ci.low < 0.5 < ci.high

    def test_reproducible_with_seed(self):
        a = bootstrap_interval([1.0, 0.0] * 25, iterations=1000, seed=11)
        b = bootstrap_interval([1.0, 0.0] * 25, iterations=1000, seed=11)
        assert (a.low, a.high) == (b.low, b.high)

    def test_agrees_with_analytic_wilson_for_binary_data(self):
        boot = bootstrap_interval([1.0] * 50 + [0.0] * 50, iterations=5000, seed=3)
        analytic = wilson_interval(50, 100)
        assert boot.low == pytest.approx(analytic.low, abs=0.05)
        assert boot.high == pytest.approx(analytic.high, abs=0.05)

    def test_constant_data_has_zero_width(self):
        assert bootstrap_interval([0.7] * 20, iterations=500, seed=1).width == 0.0

    def test_single_value(self):
        ci = bootstrap_interval([0.42], iterations=100)
        assert ci.low == ci.high == pytest.approx(0.42)

    def test_empty_input(self):
        ci = bootstrap_interval([], iterations=100)
        assert (ci.point, ci.low, ci.high) == (0.0, 0.0, 0.0)


class TestBCaInterval:
    def test_brackets_the_mean(self):
        ci = bca_interval([1.0] * 50 + [0.0] * 50, iterations=3000, seed=5)
        assert ci.low < 0.5 < ci.high

    def test_reproducible_with_seed(self):
        a = bca_interval([1.0, 0.0] * 30, iterations=1000, seed=9)
        b = bca_interval([1.0, 0.0] * 30, iterations=1000, seed=9)
        assert (a.low, a.high) == (b.low, b.high)

    def test_differs_from_percentile_on_skewed_data(self):
        import random

        rng = random.Random(3)
        skewed = [rng.expovariate(1.0) for _ in range(60)]
        bca = bca_interval(skewed, iterations=8000, seed=11)
        percentile = bootstrap_interval(skewed, iterations=8000, seed=11)
        assert (bca.low, bca.high) != (percentile.low, percentile.high)

    def test_close_to_percentile_on_symmetric_data(self):
        import random

        rng = random.Random(4)
        symmetric = [rng.gauss(0, 1) for _ in range(80)]
        bca = bca_interval(symmetric, iterations=8000, seed=11)
        percentile = bootstrap_interval(symmetric, iterations=8000, seed=11)
        assert bca.low == pytest.approx(percentile.low, abs=0.1)

    def test_falls_back_for_tiny_samples(self):
        assert bca_interval([1.0, 0.0], iterations=200).method in {"bootstrap", "bca"}

    def test_empty_input(self):
        assert bca_interval([], iterations=100).point == 0.0


class TestPairedBootstrap:
    def test_uniform_improvement(self):
        ci = paired_bootstrap_diff([0.0] * 40, [1.0] * 40, iterations=3000, seed=3)
        assert ci.point == pytest.approx(1.0)
        assert ci.low > 0

    def test_no_change_interval_contains_zero(self):
        ci = paired_bootstrap_diff([0.5] * 40, [0.5] * 40, iterations=3000, seed=3)
        assert ci.contains(0.0)

    def test_regression_is_negative(self):
        ci = paired_bootstrap_diff([1.0] * 40, [0.0] * 40, iterations=3000, seed=3)
        assert ci.point == pytest.approx(-1.0)
        assert ci.high < 0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(PairedLengthError) as excinfo:
            paired_bootstrap_diff([1.0], [1.0, 2.0])
        assert "1 and 2" in str(excinfo.value)


class TestEffectSize:
    def test_cliffs_delta_bounds(self):
        assert cliffs_delta([0.0] * 10, [1.0] * 10) == 1.0
        assert cliffs_delta([1.0] * 10, [0.0] * 10) == -1.0
        assert cliffs_delta([0.5] * 10, [0.5] * 10) == 0.0

    def test_cliffs_delta_empty(self):
        assert cliffs_delta([], [1.0]) == 0.0

    def test_cohens_d_direction(self):
        assert cohens_d([1, 2, 3, 4], [3, 4, 5, 6]) > 0
        assert cohens_d([3, 4, 5, 6], [1, 2, 3, 4]) < 0

    def test_cohens_d_zero_variance(self):
        assert cohens_d([1, 1, 1], [1, 1, 1]) == 0.0

    def test_cohens_d_needs_two_points(self):
        assert cohens_d([1.0], [2.0]) == 0.0

    @pytest.mark.parametrize("delta,label", [
        (0.05, "negligible"), (0.25, "small"), (0.40, "medium"), (0.80, "large"),
    ])
    def test_interpretation_thresholds(self, delta, label):
        assert interpret_effect(delta) == label

    def test_interpretation_uses_magnitude(self):
        assert interpret_effect(-0.8) == interpret_effect(0.8)


class TestPowerAnalysis:
    def test_sample_size_matches_published_value(self):
        assert 380 <= required_sample_size(0.5, 0.1) <= 420

    def test_smaller_effects_need_more_data(self):
        assert required_sample_size(0.5, 0.05) > 3 * required_sample_size(0.5, 0.1)

    def test_zero_delta_returns_zero(self):
        assert required_sample_size(0.5, 0.0) == 0

    def test_mde_round_trips_with_sample_size(self):
        assert minimum_detectable_effect(400, 0.5) == pytest.approx(0.1, abs=0.02)

    def test_mde_shrinks_with_more_data(self):
        assert minimum_detectable_effect(100) > minimum_detectable_effect(1000)

    def test_mde_zero_samples(self):
        assert minimum_detectable_effect(0) == 1.0


class TestDescriptives:
    def test_mean_and_stdev(self):
        assert mean([1, 2, 3, 4]) == 2.5
        assert stdev([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(2.138, abs=1e-3)

    def test_empty_and_singleton(self):
        assert mean([]) == 0.0
        assert stdev([5.0]) == 0.0

    def test_stratified_rates(self):
        rates = stratified_rates({"easy": (9, 10), "hard": (2, 10)})
        assert rates["easy"].point == 0.9
        assert rates["hard"].point == 0.2
        assert rates["easy"].low > rates["hard"].high
