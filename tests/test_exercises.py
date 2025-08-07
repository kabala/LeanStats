from __future__ import annotations

import math

import numpy as np

from src.intervals.solutions import (
    z_critical_value,
    margin_of_error_proportion,
    sample_size_for_proportion,
    ci_wald_proportion,
    ci_wilson_proportion,
    ci_exact_proportion,
    interval_width,
    sample_size_with_fpc,
    sample_size_with_design_effect,
    ci_mean_known_sigma,
    ci_mean_unknown_sigma,
    margin_of_error_mean,
    sample_size_for_mean,
    ci_diff_proportions,
    sample_size_for_diff_proportions,
    ci_agresti_coull,
    ci_poisson_rate,
    max_required_n_over_p_grid,
    ci_logit_proportion,
    recommended_ci_method,
    ci_bootstrap_proportion,
    sample_size_consistency,
)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def test_1_z_crit():
    assert approx(z_critical_value(0.95), 1.959963984540054)
    assert approx(z_critical_value(0.99), 2.5758293035489004)


def test_2_me_proportion():
    me = margin_of_error_proportion(0.5, 100, 0.95)
    assert 0.09 < me < 0.11


def test_3_n_for_proportion():
    n = sample_size_for_proportion(0.03, 0.5, 0.95)
    assert 1000 < n < 1100


def test_4_5_6_proportion_intervals():
    p_hat = 0.52
    n = 500
    x = int(round(p_hat * n))
    wald = ci_wald_proportion(p_hat, n, 0.95)
    wilson = ci_wilson_proportion(p_hat, n, 0.95)
    exact_ci = ci_exact_proportion(x, n, 0.95)
    # Sanity: all are within [0,1] and Wilson is typically more stable than Wald
    for a, b in [wald, wilson, exact_ci]:
        assert 0 <= a <= b <= 1
    assert interval_width(wilson) <= interval_width(wald) + 0.02


def test_8_fpc():
    n_inf = sample_size_for_proportion(0.03, 0.5, 0.95)
    n_f = sample_size_with_fpc(n_inf, 10_000)
    assert n_f < n_inf


def test_9_deff():
    n_inf = 1000
    assert sample_size_with_design_effect(n_inf, 1.5) == 1500


def test_10_11_mean():
    z_ic = ci_mean_known_sigma(10.0, 2.0, 100, 0.95)
    t_ic = ci_mean_unknown_sigma(10.0, 2.5, 25, 0.95)
    assert z_ic[0] < 10 < z_ic[1]
    assert t_ic[0] < 10 < t_ic[1]


def test_12_me_mean():
    me_z = margin_of_error_mean(2.0, 100, 0.95, True)
    me_t = margin_of_error_mean(2.5, 25, 0.95, False)
    assert me_z > 0
    assert me_t > me_z * 0.9  # should be similar order of magnitude


def test_13_n_mean():
    n = sample_size_for_mean(0.5, 2.0, 0.95)
    assert n > 0


def test_14_15_diff_proportions():
    ic = ci_diff_proportions(52, 100, 45, 100, 0.95)
    assert ic[0] <= (0.52 - 0.45) <= ic[1]
    n = sample_size_for_diff_proportions(0.06, 0.5, 0.5, 0.95)
    assert n > 0


def test_16_agresti_coull_ordering():
    ic = ci_agresti_coull(0.52, 500, 0.95)
    assert 0 <= ic[0] <= ic[1] <= 1


def test_17_poisson():
    lo, hi = ci_poisson_rate(12.0, 0.95)
    assert 0 <= lo <= hi


def test_18_max_over_p():
    pts = np.linspace(0.05, 0.95, 19)
    val = max_required_n_over_p_grid(pts, 0.03, 0.95)
    assert val > 0


def test_19_logit():
    lo, hi = ci_logit_proportion(0.52, 500, 0.95)
    assert 0 <= lo <= hi <= 1


def test_20_recommended_method():
    assert recommended_ci_method(50, 0.02) == "exact"
    assert recommended_ci_method(50, 0.08) in {"wilson", "exact"}
    assert recommended_ci_method(500, 0.5) == "normal"


def test_21_bootstrap():
    rng = np.random.default_rng(123)
    data = rng.binomial(1, 0.52, size=300)
    lo, hi = ci_bootstrap_proportion(data, 0.95, 500, 123)
    assert 0.3 < lo < hi < 0.7


def test_22_consistency():
    n_m, n_s = sample_size_consistency(0.03, 0.5, 0.95)
    # should be close
    assert abs(n_m - n_s) / n_s < 0.02

