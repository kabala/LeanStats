## LeanStats — Confidence intervals and margin of error

This repository helps you master confidence intervals and margin of error, with a short primer, a “Python notebook script” containing guided exercises, and automated tests to verify your solutions.

### Short primer
- **Confidence interval (CI)**: an estimated range that, with confidence \(1-\alpha\), contains the population parameter. For a proportion \(p\), a 95% normal-approx CI is \([\hat p - z_{\alpha/2}\,SE,\ \hat p + z_{\alpha/2}\,SE]\), where \(SE=\sqrt{\hat p(1-\hat p)/n}\).
- **Margin of error (ME)**: half the CI width. For proportions under the normal approximation, \(ME = z_{\alpha/2}\,\sqrt{\hat p(1-\hat p)/n}\). For sample size planning: \(n = z_{\alpha/2}^2\, p(1-p)/ME^2\) (maximized at \(p=0.5\)).

### Structure
- `confidence_interval.ipynb`: Original notebook comparing methods.
- `notebooks/intervals_exercises.py`: Python notebook script with 22 exercises and hints.
- `src/intervals/solutions.py`: Implemented functions that solve the exercises (study or re-implement them).
- `tests/`: Automated tests to validate the exercises with `pytest`.
- `requirements.txt`: Minimal dependencies.

### Exercise plan (22)
1. Compute the critical value \(z_{\alpha/2}\) for a given confidence level.
2. Margin of error for a single proportion (normal approximation).
3. Sample size for a proportion given ME and confidence.
4. Wald (normal) CI for a proportion.
5. Wilson CI for a proportion.
6. Exact CI (Clopper–Pearson) for a proportion.
7. Compare CI widths across methods (normal, Wilson, exact).
8. Sample size with finite population correction (FPC).
9. Adjust sample size by design effect (DEFF).
10. CI for a mean with known \(\sigma\) (z-based).
11. CI for a mean with unknown \(\sigma\) (t-based).
12. Margin of error for the mean.
13. Sample size for the mean.
14. CI for difference in proportions (two independent samples).
15. Sample size for difference in proportions (equal n).
16. Agresti–Coull CI for a proportion.
17. CI for a Poisson rate.
18. Show that required sample size for a proportion is maximized at \(p=0.5\) (numerical check).
19. Logit-transformed CI for a proportion (approximate).
20. Rule-of-thumb method selection for proportion CI based on \(n\) and \(\hat p\).
21. Bootstrap CI for a proportion (percentile, fixed seed).
22. Consistency check: manual sample size vs `statsmodels`.

Each exercise in `notebooks/ejercicios_intervalos.py` includes a statement, hint(s), and an example call to the corresponding function in `src/intervalos/soluciones.py`.

### Requirements
```bash
pip install -r requirements.txt
```

If you use a conda environment named `ds`, prefix commands with:
```bash
conda run -n ds <your command>
```

### How to use
1. Explore the functions in `src/intervalos/soluciones.py` and the prompts in `notebooks/ejercicios_intervalos.py`.
2. Run the “Python notebook script” directly as a script or execute it in your editor block by block.
3. Run the tests to check your answers:
```bash
pytest -q
```

If you prefer to practice by coding from scratch, try re-implementing the functions in `src/intervalos/soluciones.py` and verify with the tests.
