"""
Guided exercises: Confidence intervals and margin of error

How to use:
- Run this file with Python to see partial outputs and hints.
- Or open it in your editor and run it block by block.

Each exercise references functions in src/intervals/solutions.py.
"""

from __future__ import annotations

import numpy as np

from src.intervals import (
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


def header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    # 1
    header("1) z critical value for a given confidence level")
    # Hint: z_{alpha/2} = norm.ppf(1 - alpha/2)
    print("z 95%:", z_critical_value(0.95))
    print("z 99%:", z_critical_value(0.99))

    # 2
    header("2) Margin of error for a proportion (normal)")
    # Hint: SE = sqrt(p(1-p)/n)
    print("ME (p=0.5, n=100, 95%):", margin_of_error_proportion(0.5, 100, 0.95))

    # 3
    header("3) Sample size for a proportion given ME")
    # Hint: solve n from ME = z * sqrt(p(1-p)/n)
    print("n (ME=0.03, p=0.5, 95%):", sample_size_for_proportion(0.03, 0.5, 0.95))

    # 4
    header("4) Wald (normal) CI for a proportion")
    print("Wald CI (p=0.52, n=500):", ci_wald_proportion(0.52, 500, 0.95))

    # 5
    header("5) Wilson CI for a proportion")
    print("Wilson CI (p=0.52, n=500):", ci_wilson_proportion(0.52, 500, 0.95))

    # 6
    header("6) Exact CI (Clopper–Pearson)")
    print("Exact CI (x=260, n=500):", ci_exact_proportion(260, 500, 0.95))

    # 7
    header("7) Compare CI widths across methods")
    wald = ci_wald_proportion(0.52, 500, 0.95)
    wilson = ci_wilson_proportion(0.52, 500, 0.95)
    exact = ci_exact_proportion(260, 500, 0.95)
    print("Width Wald/Wilson/Exact:", interval_width(wald), interval_width(wilson), interval_width(exact))

    # 8
    header("8) Sample size with FPC")
    n_inf = sample_size_for_proportion(0.03, 0.5, 0.95)
    print("n without FPC:", n_inf)
    print("n with FPC (N=10000):", sample_size_with_fpc(n_inf, 10_000))

    # 9
    header("9) Adjust by design effect (DEFF)")
    print("n with DEFF=1.5:", sample_size_with_design_effect(n_inf, 1.5))

    # 10
    header("10) CI for mean with known sigma (z)")
    print("Mean z CI:", ci_mean_known_sigma(10.0, 2.0, 100, 0.95))

    # 11
    header("11) CI for mean with unknown sigma (t)")
    print("Mean t CI:", ci_mean_unknown_sigma(10.0, 2.5, 25, 0.95))

    # 12
    header("12) Margin of error for the mean")
    print("ME mean (z):", margin_of_error_mean(2.0, 100, 0.95, known_sigma=True))
    print("ME mean (t):", margin_of_error_mean(2.5, 25, 0.95, known_sigma=False))

    # 13
    header("13) Sample size for the mean")
    print("n for mean (ME=0.5, sigma=2):", sample_size_for_mean(0.5, 2.0, 0.95))

    # 14
    header("14) CI for difference of proportions")
    print("Diff proportion CI:", ci_diff_proportions(52, 100, 45, 100, 0.95))

    # 15
    header("15) Sample size for difference of proportions")
    print("n per group (ME=0.06, p1=0.5, p2=0.5):", sample_size_for_diff_proportions(0.06, 0.5, 0.5, 0.95))

    # 16
    header("16) Agresti–Coull CI")
    print("AC CI (p=0.52, n=500):", ci_agresti_coull(0.52, 500, 0.95))

    # 17
    header("17) CI for a Poisson rate")
    print("Poisson CI (lambda_hat=12):", ci_poisson_rate(12.0, 0.95))

    # 18
    header("18) Maximum n at p=0.5 (numerical)")
    pts = np.linspace(0.05, 0.95, 19)
    print("max n over grid:", max_required_n_over_p_grid(pts, me=0.03, confidence=0.95))

    # 19
    header("19) Logit-transformed CI")
    print("Logit CI (p=0.52, n=500):", ci_logit_proportion(0.52, 500, 0.95))

    # 20
    header("20) Rule-of-thumb CI method selection")
    for ph in [0.02, 0.08, 0.5]:
        print(ph, "->", recommended_ci_method(50, ph))

    # 21
    header("21) Bootstrap CI for a proportion")
    rng = np.random.default_rng(123)
    data = rng.binomial(1, 0.52, size=500)
    print("Bootstrap CI:", ci_bootstrap_proportion(data, 0.95, 1000, 123))

    # 22
    header("22) Consistency: statsmodels vs manual")
    print("n manual vs statsmodels:", sample_size_consistency(0.03, 0.5, 0.95))


if __name__ == "__main__":
    main()


