"""
Real-world style confidence interval & margin of error exercises (20)

Solve each scenario using helper functions from src/intervals/solutions.py.
Convert this script to a Jupyter notebook later if you prefer (VS Code: Export / Convert).

Legend:
- Each problem has a description cell (# %% [markdown]) followed by a starter code cell.
- Replace TODOs with your code; keep variable names intuitive.
- Keep prints compact (1–3 lines per part).

NOTE: If you run this file directly from the project root (recommended) imports will work.
      To allow running it from within the notebooks/ folder (or via VS Code cell runner),
      we programmatically add the project root to sys.path below.
"""

from __future__ import annotations

# %%
import sys
import pathlib
import numpy as np

# Add project root (parent of this file's directory) to sys.path for 'src' package.
try:  # __file__ may be undefined in some interactive contexts
    _THIS = pathlib.Path(__file__).resolve()
    _ROOT = _THIS.parents[1]  # project root (contains 'src')
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
except NameError:
    pass

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


# %%
# ## 1. University Residency Poll
# 232 students sampled; 43% out-of-state. Published 95% CI: (0.3663, 0.4937).
# Tasks:
#   a) Derive margin of error from the CI.
#   b) 95% conservative (p=0.5) sample size for ME <= 0.04.
#   c) 98% conservative sample size for ME <= 0.03.
# Hints: margin_of_error = interval_width(ci)/2; sample_size_for_proportion.
# --- Your work below ---
ci = (0.3663, 0.4937)
# me_1a = interval_width(ci)/2
# n_1b = sample_size_for_proportion(0.04, 0.5, 0.95)
# n_1c = sample_size_for_proportion(0.03, 0.5, 0.98)
# print("1a ME:", me_1a)
# print("1b n 95% ME<=0.04 (conservative):", n_1b)
# print("1c n 98% ME<=0.03 (conservative):", n_1c)

# %%
# ## 2. Customer Satisfaction Survey
# 78 "very satisfied" out of 180 customers.
# Tasks:
#   a) Compute Wald, Wilson, Agresti–Coull, Exact 95% CIs and their widths.
#   b) Which method would you report? Use recommended_ci_method for guidance.
# Hints: ci_wald_proportion, ci_wilson_proportion, ci_agresti_coull, ci_exact_proportion, interval_width.
# --- Your work below ---
x = 78
n = 180
p_hat = x / n

# wald_2 = ci_wald_proportion(p_hat, n)
# wilson_2 = ci_wilson_proportion(p_hat, n)
# ac_2 = ci_agresti_coull(p_hat, n)
# exact_2 = ci_exact_proportion(x, n)
# print("2a Wald:", wald_2, "width", interval_width(wald_2))
# print("2a Wilson:", wilson_2, "width", interval_width(wilson_2))
# print("2a Agresti–Coull:", ac_2, "width", interval_width(ac_2))
# print("2a Exact:", exact_2, "width", interval_width(exact_2))
# print("2b recommended method:", recommended_ci_method(n, p_hat))

print(p_hat)



# %%
# ## 3. Email Campaign A/B Test
# A: 410 / 2000 opens, B: 465 / 2050 opens.
# Tasks:
#   a) 95% Wilson CI for each.
#   b) 95% CI for difference (A - B).
#   c) Interpretation: does CI for difference include 0?
# Hints: ci_wilson_proportion, ci_diff_proportions.
# --- Your work below ---
xa, na = 410, 2000
xb, nb = 465, 2050
# pa = xa/na; pb = xb/nb
# wilson_a = ci_wilson_proportion(pa, na)
# wilson_b = ci_wilson_proportion(pb, nb)
# diff_3 = ci_diff_proportions(xa, na, xb, nb)
# print("3a Wilson A:", wilson_a)
# print("3a Wilson B:", wilson_b)
# print("3b Diff CI (A-B):", diff_3)

# %%
# ## 4. Manufacturing Defect Rate
# 18 defectives in 1200.
# Tasks:
#   a) 95% Wilson CI.
#   b) Should Wald be avoided? (recommended_ci_method)
#   c) Required n for ME=0.005 (use current p-hat).
# --- Your work below ---
x4, n4 = 18, 1200
# p4 = x4/n4
# wilson_4 = ci_wilson_proportion(p4, n4)
# rec_4 = recommended_ci_method(n4, p4)
# n_target_4 = sample_size_for_proportion(0.005, p4)
# print("4a Wilson:", wilson_4)
# print("4b recommended method:", rec_4)
# print("4c required n:", n_target_4)

# %%
# ## 5. Finite Population Voter List
# Need 95% ME<=0.03, N=25000, p=0.5.
# Tasks:
#   a) n ignoring FPC.
#   b) n with FPC.
# --- Your work below ---
# n_inf_5 = sample_size_for_proportion(0.03, 0.5)
# n_fpc_5 = sample_size_with_fpc(n_inf_5, 25_000)
# print("5a n infinite:", n_inf_5)
# print("5b n with FPC:", n_fpc_5)

# %%
# ## 6. Cluster Sampling Design Effect
# Use n from Ex.5 before FPC; DEFF=1.7.
# Tasks:
#   a) Inflate SRS n by DEFF.
#   b) Apply FPC after DEFF; compare with applying DEFF after FPC.
# --- Your work below ---
# from math import ceil
# n_deff_6 = sample_size_with_design_effect(n_inf_5, 1.7)
# n_deff_then_fpc = sample_size_with_fpc(n_deff_6, 25_000)
# n_fpc_then_deff = sample_size_with_design_effect(n_fpc_5, 1.7)
# print("6a n after DEFF:", n_deff_6)
# print("6b (DEFF->FPC):", n_deff_then_fpc, "(FPC->DEFF):", n_fpc_then_deff)

# %%
# ## 7. Call Center Mean Handling Time
# σ=2.4 known; n=64; mean=11.2.
# Tasks:
#   a) 95% CI (known sigma, z).
#   b) Achieved ME.
#   c) n needed for ME=0.3.


# %%
# ## 8. Startup Feature Usage (Unknown σ)
# n=20; xbar=14.5; s=5.2.
# Tasks:
#   a) 95% CI (t-based).
#   b) 99% CI; note width change.
# --- Your work below ---
# ci8_95 = ci_mean_unknown_sigma(14.5, 5.2, 20, 0.95)
# ci8_99 = ci_mean_unknown_sigma(14.5, 5.2, 20, 0.99)
# print("8a 95%:", ci8_95)
# print("8b 99%:", ci8_99)

# %%
# ## 9. Two-Sided Proportion Stability Check
# Expect p1=0.55, p2=0.50; want ME<=0.04 (difference) at 95%.
# Tasks:
#   a) n per group with expected p's.
#   b) n per group if both p=0.50.
# --- Your work below ---
# n9_expected = sample_size_for_diff_proportions(0.04, 0.55, 0.50)
# n9_equal = sample_size_for_diff_proportions(0.04, 0.50, 0.50)
# print("9a n per group expected p's:", n9_expected)
# print("9b n per group if both 0.50:", n9_equal)

# %%
# ## 10. Rare Event (Escalations)
# 9 escalations in 2400 tickets.
# Tasks:
#   a) Exact 95% CI.
#   b) Wilson 95% CI.
#   c) Which is wider? Interpret.
# --- Your work below ---
x10, n10 = 9, 2400
# p10 = x10/n10
# exact_10 = ci_exact_proportion(x10, n10)
# wilson_10 = ci_wilson_proportion(p10, n10)
# print("10 Exact:", exact_10)
# print("10 Wilson:", wilson_10)

# %%
# ## 11. Machine Failures (Poisson)
# λ̂=15 failures.
# Tasks:
#   a) 95% CI for rate.
#   b) 99% CI; compare expansion factor vs 95%.
# --- Your work below ---
# ci11_95 = ci_poisson_rate(15, 0.95)
# ci11_99 = ci_poisson_rate(15, 0.99)
# print("11 95%:", ci11_95)
# print("11 99%:", ci11_99)

# %%
# ## 12. Max Sample Size Over p Grid
# ME=0.025, 95%, p grid 0.05..0.95.
# Tasks:
#   a) Compute required n over grid; report max.
#   b) Identify p where max occurs.
# --- Your work below ---
# p_grid_12 = np.linspace(0.05, 0.95, 19)
# ns_12 = [sample_size_for_proportion(0.025, p) for p in p_grid_12]
# max_n_12 = max(ns_12)
# p_at_max_12 = p_grid_12[int(np.argmax(ns_12))]
# print("12 max n:", max_n_12, "at p=", p_at_max_12)

# %%
# ## 13. Logistic vs Wald CI
# p̂=0.62, n=140.
# Tasks:
#   a) Wald CI.
#   b) Logit-transformed CI.
#   c) Note asymmetry and bounds.
# --- Your work below ---
# wald_13 = ci_wald_proportion(0.62, 140)
# logit_13 = ci_logit_proportion(0.62, 140)
# print("13 Wald:", wald_13)
# print("13 Logit:", logit_13)

# %%
# ## 14. Bootstrap CI for Conversion
# Simulate n=600 Bernoulli(p=0.47) data; bootstrap 1000 resamples.
# Tasks:
#   a) Bootstrap percentile 95% CI.
#   b) Wilson CI.
#   c) Compare closeness.
# --- Your work below ---
# rng14 = np.random.default_rng(2024)
# data14 = rng14.binomial(1, 0.47, size=600)
# boot_14 = ci_bootstrap_proportion(data14, 0.95, 1000, seed=2024)
# wilson_14 = ci_wilson_proportion(data14.mean(), 600)
# print("14 Bootstrap:", boot_14)
# print("14 Wilson:", wilson_14)

# %%
# ## 15. Method Recommendation Edge Cases
# n=40; p-hat values: 0.02,0.05,0.10,0.50,0.90,0.98.
# Tasks:
#   a) Determine recommended method for each.
#   b) Summarize pattern.
# --- Your work below ---
# ps_15 = [0.02,0.05,0.10,0.50,0.90,0.98]
# methods_15 = {p: recommended_ci_method(40,p) for p in ps_15}
# print("15 methods:", methods_15)

# %%
# ## 16. Consistency Check statsmodels
# ME=0.035, p=0.5, 95%.
# Tasks:
#   a) Manual sample size vs statsmodels.
#   b) Difference.
# --- Your work below ---
# manual_16, sm_16 = sample_size_consistency(0.035, 0.5, 0.95)
# print("16 manual vs statsmodels:", manual_16, sm_16, "diff:", sm_16 - manual_16)

# %%
# ## 17. Re-Estimating After Pilot
# Pilot p̂=0.38, n=150; target ME=0.025 (95%).
# Tasks:
#   a) Required n using p̂.
#   b) Required n using p=0.5.
#   c) Percent increase.
# --- Your work below ---
# p17 = 0.38
# n_required_17 = sample_size_for_proportion(0.025, p17)
# n_conservative_17 = sample_size_for_proportion(0.025, 0.5)
# percent_increase_17 = (n_conservative_17 - n_required_17)/n_required_17 * 100
# print("17 n using p-hat:", n_required_17)
# print("17 conservative n:", n_conservative_17)
# print("17 % increase:", percent_increase_17)

# %%
# ## 18. Hospital Readmission
# 37 readmissions of 210.
# Tasks:
#   a) Wilson 95% CI.
#   b) Margin of error.
#   c) n for future ME=0.04.
# --- Your work below ---
# x18, n18 = 37, 210
# p18 = x18/n18
# wilson_18 = ci_wilson_proportion(p18, n18)
# me_18 = interval_width(wilson_18)/2
# n_plan_18 = sample_size_for_proportion(0.04, p18)
# print("18 Wilson:", wilson_18)
# print("18 ME:", me_18)
# print("18 n for future ME 0.04:", n_plan_18)

# %%
# ## 19. Year-over-Year Satisfaction
# Y1: 420/800, Y2: 470/850.
# Tasks:
#   a) Wilson CI each year.
#   b) CI for difference (Y2 - Y1).
#   c) Interpret sign / inclusion of 0.
# --- Your work below ---
# x1_19, n1_19 = 420, 800
# x2_19, n2_19 = 470, 850
# p1_19, p2_19 = x1_19/n1_19, x2_19/n2_19
# wilson1_19 = ci_wilson_proportion(p1_19, n1_19)
# wilson2_19 = ci_wilson_proportion(p2_19, n2_19)
# diff_19 = ci_diff_proportions(x2_19, n2_19, x1_19, n1_19)
# print("19 Wilson Y1:", wilson1_19)
# print("19 Wilson Y2:", wilson2_19)
# print("19 Diff (Y2-Y1):", diff_19)

# %%
# ## 20. Content Platform Precision Drive
# Current: p̂=0.57, n=900. Need ME<=0.015 (95%).
# Tasks:
#   a) Total n required (use p̂).
#   b) Additional observations needed.
#   c) Repeat with p=0.5; compare additional.
# --- Your work below ---
# current_p20, current_n20 = 0.57, 900
# n_goal_20 = sample_size_for_proportion(0.015, current_p20)
# additional_20 = n_goal_20 - current_n20
# n_goal_cons_20 = sample_size_for_proportion(0.015, 0.5)
# additional_cons_20 = n_goal_cons_20 - current_n20
# print("20 total n (p-hat):", n_goal_20, "additional:", additional_20)
# print("20 total n (p=0.5):", n_goal_cons_20, "additional:", additional_cons_20)

if __name__ == "__main__":
    # Optionally run everything at once
    pass
