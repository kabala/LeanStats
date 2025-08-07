from __future__ import annotations

import math
from typing import Tuple, Sequence

import numpy as np
from scipy.stats import norm, t, beta


# 1. Critical z value for a confidence level
def z_critical_value(confidence: float) -> float:
    alpha = 1 - confidence
    return norm.ppf(1 - alpha / 2)


# 2. Margin of error for a single proportion (normal approximation)
def margin_of_error_proportion(p: float, n: int, confidence: float = 0.95) -> float:
    z = z_critical_value(confidence)
    se = math.sqrt(p * (1 - p) / n)
    return z * se


# 3. Required sample size for a proportion given target ME
def sample_size_for_proportion(me: float, p: float = 0.5, confidence: float = 0.95) -> float:
    z = z_critical_value(confidence)
    return (z**2) * p * (1 - p) / (me**2)


# 4. Wald (normal) CI for a proportion
def ci_wald_proportion(p_hat: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    me = margin_of_error_proportion(p_hat, n, confidence)
    return max(0.0, p_hat - me), min(1.0, p_hat + me)


# 5. Wilson CI for a proportion
def ci_wilson_proportion(p_hat: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    z = z_critical_value(confidence)
    denom = 1 + z**2 / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    half_width = (z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


# 6. Exact (Clopper–Pearson) CI for a proportion
def ci_exact_proportion(x: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    alpha = 1 - confidence
    if x == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2, x, n - x + 1)
    if x == n:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha / 2, x + 1, n - x)
    return lower, upper


# 7. Interval width helper
def interval_width(interval: Tuple[float, float]) -> float:
    a, b = interval
    return float(b - a)


# 8. Finite population correction (FPC) for required n
def sample_size_with_fpc(n_infinite: float, population_size: int) -> float:
    return n_infinite / (1 + (n_infinite - 1) / population_size)


# 9. Design effect adjustment
def sample_size_with_design_effect(n_srs: float, design_effect: float) -> float:
    return n_srs * design_effect


# 10. z-based CI for a mean with known sigma
def ci_mean_known_sigma(xbar: float, sigma: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    z = z_critical_value(confidence)
    me = z * sigma / math.sqrt(n)
    return xbar - me, xbar + me


# 11. t-based CI for a mean with unknown sigma
def ci_mean_unknown_sigma(xbar: float, s: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    alpha = 1 - confidence
    tval = t.ppf(1 - alpha / 2, df=n - 1)
    me = tval * s / math.sqrt(n)
    return xbar - me, xbar + me


# 12. Margin of error for the mean
def margin_of_error_mean(sd: float, n: int, confidence: float = 0.95, known_sigma: bool = True) -> float:
    if known_sigma:
        z = z_critical_value(confidence)
        return z * sd / math.sqrt(n)
    alpha = 1 - confidence
    tval = t.ppf(1 - alpha / 2, df=n - 1)
    return tval * sd / math.sqrt(n)


# 13. Sample size for mean (z, known sigma)
def sample_size_for_mean(me: float, sigma: float, confidence: float = 0.95) -> float:
    z = z_critical_value(confidence)
    return (z * sigma / me) ** 2


# 14. CI for difference of proportions (normal approx)
def ci_diff_proportions(x1: int, n1: int, x2: int, n2: int, confidence: float = 0.95) -> Tuple[float, float]:
    p1 = x1 / n1
    p2 = x2 / n2
    diff = p1 - p2
    z = z_critical_value(confidence)
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    me = z * se
    return diff - me, diff + me


# 15. Sample size for difference of proportions (equal n)
def sample_size_for_diff_proportions(me: float, p1: float, p2: float, confidence: float = 0.95) -> float:
    z = z_critical_value(confidence)
    variance_sum = p1 * (1 - p1) + p2 * (1 - p2)
    return (z**2) * variance_sum / (me**2)


# 16. Agresti–Coull CI for a proportion
def ci_agresti_coull(p_hat: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    z = z_critical_value(confidence)
    n_tilde = n + z**2
    p_tilde = (p_hat * n + (z**2) / 2) / n_tilde
    half = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return max(0.0, p_tilde - half), min(1.0, p_tilde + half)


# 17. CI for a Poisson rate (exact via chi-square quantiles)
def ci_poisson_rate(lambda_hat: float, confidence: float = 0.95) -> Tuple[float, float]:
    if lambda_hat < 0:
        raise ValueError("lambda must be non-negative")
    from scipy.stats import chi2
    alpha = 1 - confidence
    k_int = int(round(lambda_hat))
    if k_int == 0:
        lower = 0.0
    else:
        lower = 0.5 * chi2.ppf(alpha / 2, df=2 * k_int)
    upper = 0.5 * chi2.ppf(1 - alpha / 2, df=2 * (k_int + 1))
    return float(lower), float(upper)


# 18. Numerical check: maximum required n over a grid of p values
def max_required_n_over_p_grid(p_grid: Sequence[float], me: float, confidence: float = 0.95) -> float:
    values = [sample_size_for_proportion(me, p, confidence) for p in p_grid]
    return float(max(values))


# 19. Logit-transformed CI for a proportion (approximate)
def ci_logit_proportion(p_hat: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    eps = 1e-9
    p = min(max(p_hat, eps), 1 - eps)
    logit = math.log(p / (1 - p))
    se = math.sqrt(1 / (n * p * (1 - p)))
    z = z_critical_value(confidence)
    lower_logit = logit - z * se
    upper_logit = logit + z * se
    def inv_logit(x: float) -> float:
        return 1 / (1 + math.exp(-x))
    return inv_logit(lower_logit), inv_logit(upper_logit)


# 20. Rule-of-thumb: recommended CI method for proportions
def recommended_ci_method(n: int, p_hat: float) -> str:
    np_ = n * p_hat
    nq_ = n * (1 - p_hat)
    if np_ < 5 or nq_ < 5:
        return "exact"
    if np_ < 10 or nq_ < 10:
        return "wilson"
    return "normal"


# 21. Bootstrap percentile CI for a proportion
def ci_bootstrap_proportion(data: Sequence[int], confidence: float = 0.95, n_boot: int = 2000, seed: int = 123) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boots.append(sample.mean())
    alpha = 1 - confidence
    lower = float(np.quantile(boots, alpha / 2))
    upper = float(np.quantile(boots, 1 - alpha / 2))
    return lower, upper


# 22. Consistency: manual sample size vs statsmodels helper
def sample_size_consistency(me: float, p: float, confidence: float = 0.95) -> Tuple[float, float]:
    from statsmodels.stats.proportion import samplesize_confint_proportion
    alpha = 1 - confidence
    n_manual = sample_size_for_proportion(me, p, confidence)
    # Compatibility: older/newer statsmodels use half_width vs half_length
    try:
        n_sm = samplesize_confint_proportion(p, half_width=me, alpha=alpha)
    except TypeError:
        n_sm = samplesize_confint_proportion(p, half_length=me, alpha=alpha)
    return float(n_manual), float(n_sm)


