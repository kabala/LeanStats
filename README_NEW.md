# LeanStats – Real‑World Confidence Interval & Margin of Error Practice

This extended module adds 20 applied exercises grounded in realistic data collection scenarios (polls, quality control, clinical trials, web A/B tests, call centers, industrial processes, epidemiology). Each exercise can be solved with (or partly guided by) functions already implemented in `src/intervals/solutions.py`. Your task in each problem is to (1) identify the right statistical quantity, (2) choose the matching helper function(s), and (3) compute the answer in the new interactive notebook script `notebooks/real_world_intervals.py` (a Python “notebook script” using `# %%` cell markers).

## Available Functions (Quick Map)
(See `src/intervals/solutions.py` for details.)

| Purpose | Key Functions |
|---------|---------------|
| Critical value & basics | `z_critical_value`, `margin_of_error_proportion`, `interval_width` |
| One proportion CIs | `ci_wald_proportion`, `ci_wilson_proportion`, `ci_exact_proportion`, `ci_agresti_coull`, `ci_logit_proportion`, `ci_bootstrap_proportion` |
| Planning sample size (one p) | `sample_size_for_proportion`, `sample_size_with_fpc`, `sample_size_with_design_effect`, `max_required_n_over_p_grid` |
| Means | `ci_mean_known_sigma`, `ci_mean_unknown_sigma`, `margin_of_error_mean`, `sample_size_for_mean` |
| Two proportions | `ci_diff_proportions`, `sample_size_for_diff_proportions` |
| Rates / Counts | `ci_poisson_rate` |
| Method guidance | `recommended_ci_method`, `sample_size_consistency` |

## How To Use the Interactive Script
1. Open `notebooks/real_world_intervals.py` in VS Code.
2. Ensure Python & Jupyter extensions are installed.
3. Run cell-by-cell: each exercise has a `# %%` block with the statement and a second block where you can write/verify code.
4. Replace `TODO` code with your own solutions; keep outputs visible for self‑checking.
5. (Optional) Convert to a `.ipynb` via the VS Code “Export” / “Convert to Jupyter Notebook” button once satisfied.

## Real‑World Style Exercises (20)
Each exercise gives: scenario → questions → expected approach hints (not formulas) so you must map to functions.

### 1. University Residency Poll
A university poll sampled 232 undergrads; 43% are out-of-state. The published 95% CI is (0.3663, 0.4937).
- a) Derive the margin of error from the interval.
- b) Plan a 95% conservative (p=0.5) sample with ME ≤ 4%.
- c) Plan a 98% conservative sample with ME ≤ 3%.
Use basic CI width & `sample_size_for_proportion`.

### 2. Customer Satisfaction Survey
In a pilot survey of 180 customers, 78 report being “very satisfied.”
- a) Compute Wald, Wilson, Agresti–Coull, and Exact 95% CIs. Compare widths.
- b) Which method would you report and why? (Use `recommended_ci_method` as a guide.)

### 3. Email Campaign A/B Test
Variant A: 410 opens out of 2,000 emails. Variant B: 465 opens out of 2,050.
- a) 95% CI for each proportion (Wilson).
- b) 95% CI for difference (A − B).
- c) Interpret whether there is evidence of a difference.

### 4. Manufacturing Defect Rate
A factory samples 1,200 units; 18 are defective.
- a) 95% Wilson CI for the defect proportion.
- b) Should normal/Wald be avoided here? Use rule method.
- c) Target future ME = 0.005 at 95% with current p̂; required n?

### 5. Finite Population Voter List
You need a 95% ME ≤ 3% estimate of support in a finite population of N=25,000 registered voters (conservative p=0.5).
- a) n ignoring FPC.
- b) Adjusted n with FPC.

### 6. Cluster Sampling Design Effect
From Exercise 5 sample size (pre-FPC), a design effect DEFF=1.7 is anticipated.
- a) Inflate the SRS n by DEFF.
- b) Then apply FPC sequentially; comment on order impact (explore both orders quickly).

### 7. Call Center Mean Handling Time
Assume historical σ=2.4 minutes known. A sample of n=64 calls has mean 11.2 minutes.
- a) 95% CI for mean (z-based).
- b) What ME was achieved?
- c) If you wanted ME=0.3, what n would have been needed?

### 8. Startup Feature Usage (Unknown σ)
A sample of 20 users logs average session length x̄=14.5 min with sample SD s=5.2.
- a) 95% CI using t.
- b) Recompute 99% CI; comment on width change.

### 9. Two-Sided Proportion Stability Check
Plan an A/B test expecting p1≈0.55, p2≈0.50 and wanting ME (half-width) ≤ 0.04 on the difference at 95%.
- a) Required n per group.
- b) If actual p’s are both 0.50, recompute n; is it higher or lower?

### 10. Rare Event (Support Tickets Escalation)
In 2,400 tickets, 9 required escalation.
- a) Compute Exact 95% CI.
- b) Compute Wilson 95% CI.
- c) Which is more conservative? Interpret.

### 11. Incidence Rate (Poisson) – Machine Failures
A machine records 15 failures per monitoring period (treat as Poisson λ̂ = 15).
- a) 95% CI for the rate.
- b) 99% CI — compare expansion factor.

### 12. Maximizing Sample Size Over p Grid
For ME=0.025 at 95%, evaluate required n over p∈{0.05,0.10,…,0.95}.
- a) Use the helper to find max n.
- b) Verify it occurs near p=0.5.

### 13. Logistic (Logit) CI vs Wald
For p̂=0.62, n=140:
- a) 95% Wald CI.
- b) 95% Logit-transformed CI.
- c) Discuss asymmetry and bounds.

### 14. Bootstrap CI for Conversion Rate
Synthetic data: Bernoulli with p=0.47, n=600 (simulate once with a seed). 
- a) Bootstrap 95% percentile CI (1,000 resamples).
- b) Compare to Wilson CI; are they close?

### 15. Method Recommendation Edge Cases
For n=40, examine p̂ in {0.02,0.05,0.10,0.50,0.90,0.98}.
- a) For each p̂, record recommended method.
- b) Briefly explain pattern (small counts effect).

### 16. Consistency Check With statsmodels
For ME=0.035, p=0.5, confidence=95%:
- a) Compare manual vs statsmodels sample size.
- b) Report absolute difference.

### 17. Re-Estimating After Pilot
Pilot: n=150 users, p̂=0.38. You now target ME=0.025 (95%).
- a) Compute required n using p̂.
- b) Overwrite with conservative p=0.5; percent increase?

### 18. Hospital Readmission Proportion
Sample: 210 discharged patients; 37 readmitted within 30 days.
- a) Wilson 95% CI.
- b) Margin of error from that CI.
- c) If planning ME=0.04 next year using current p̂, required n?

### 19. Comparing Satisfaction Levels Year-over-Year
Year 1: 420 satisfied out of 800. Year 2: 470 satisfied out of 850.
- a) 95% CI for each year (Wilson).
- b) 95% CI for difference (Y2 − Y1).
- c) Does interval include 0? Interpret.

### 20. Content Platform – Drive Toward Precision
Current estimate: p̂=0.57 from n=900. You want ME≤0.015 (95%).
- a) How many additional observations beyond current n?
- b) If you instead treat p as 0.5, how many additional? (Is planning conservative?)

## Suggested Workflow
For each exercise:
1. Read scenario; jot the target parameter (p, mean μ, difference, rate, etc.).
2. Pick the correct helper function(s).
3. Compute, store results in variables, print succinctly.
4. Optionally add assertions (sanity bounds) to self‑validate.

## Running Tests
Existing tests cover original exercises. You can add new tests in `tests/` if desired for these applied problems.

## Next Ideas
- Add functions for proportion difference power analysis.
- Add Bayesian credible intervals for proportions.
- Extend to variance and proportion ratio intervals.

Enjoy building intuition by mapping realistic narratives to statistical mechanics.
