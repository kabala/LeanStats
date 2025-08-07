from __future__ import annotations

# This module has been replaced by src/intervals/solutions.py (English names).
# Kept only to avoid import errors if referenced accidentally. Please use the
# new module and function names in English.


# 2. Margen de error para proporción (aprox. normal)
def margen_error_proporcion(p: float, n: int, confidence: float = 0.95) -> float:
    z = z_crit(confidence)
    se = math.sqrt(p * (1 - p) / n)
    return z * se


# 3. Tamaño de muestra para proporción dado ME
def n_para_proporcion(me: float, p: float = 0.5, confidence: float = 0.95) -> float:
    z = z_crit(confidence)
    return (z**2) * p * (1 - p) / (me**2)


# 4. IC normal (Wald) para proporción
def ic_wald_proporcion(p_hat: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    me = margen_error_proporcion(p_hat, n, confidence)
    return max(0.0, p_hat - me), min(1.0, p_hat + me)


# 5. IC de Wilson para proporción
def ic_wilson_proporcion(p_hat: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    z = z_crit(confidence)
    denom = 1 + z**2 / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    half_width = (z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * n)) / n)) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


# 6. IC exacto (Clopper–Pearson) para proporción
def ic_exacto_proporcion(x: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
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


# 7. Comparar anchos de IC
def ancho_intervalo(intervalo: Tuple[float, float]) -> float:
    a, b = intervalo
    return float(b - a)


# 8. Corrección por población finita (FPC) para n requerido
def n_con_fpc(n_infinito: float, N: int) -> float:
    # n_f = n0 / (1 + (n0 - 1)/N)
    return n_infinito / (1 + (n_infinito - 1) / N)


# 9. Ajuste por efecto de diseño
def n_con_deff(n_srs: float, deff: float) -> float:
    return n_srs * deff


# 10. IC para media con sigma conocida (z)
def ic_media_sigma_conocida(xbar: float, sigma: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    z = z_crit(confidence)
    me = z * sigma / math.sqrt(n)
    return xbar - me, xbar + me


# 11. IC para media con sigma desconocida (t)
def ic_media_sigma_desconocida(xbar: float, s: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    alpha = 1 - confidence
    tval = t.ppf(1 - alpha / 2, df=n - 1)
    me = tval * s / math.sqrt(n)
    return xbar - me, xbar + me


# 12. Margen de error para la media
def margen_error_media(sigma_o_s: float, n: int, confidence: float = 0.95, conocido: bool = True) -> float:
    if conocido:
        z = z_crit(confidence)
        return z * sigma_o_s / math.sqrt(n)
    alpha = 1 - confidence
    tval = t.ppf(1 - alpha / 2, df=n - 1)
    return tval * sigma_o_s / math.sqrt(n)


# 13. Tamaño de muestra para media (z, sigma conocida)
def n_para_media(me: float, sigma: float, confidence: float = 0.95) -> float:
    z = z_crit(confidence)
    return (z * sigma / me) ** 2


# 14. IC para diferencia de proporciones (aprox. normal)
def ic_diff_proporciones(x1: int, n1: int, x2: int, n2: int, confidence: float = 0.95) -> Tuple[float, float]:
    p1 = x1 / n1
    p2 = x2 / n2
    diff = p1 - p2
    z = z_crit(confidence)
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    me = z * se
    return diff - me, diff + me


# 15. Tamaño de muestra para diferencia de proporciones (n iguales)
def n_para_diff_proporciones(me: float, p1: float, p2: float, confidence: float = 0.95) -> float:
    z = z_crit(confidence)
    # SE = sqrt(p1(1-p1)/n + p2(1-p2)/n) = sqrt((v1+v2)/n)
    v = p1 * (1 - p1) + p2 * (1 - p2)
    return (z**2) * v / (me**2)


# 16. IC Agresti–Coull para proporción
def ic_agresti_coull(p_hat: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    z = z_crit(confidence)
    n_tilde = n + z**2
    p_tilde = (p_hat * n + (z**2) / 2) / n_tilde
    half = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return max(0.0, p_tilde - half), min(1.0, p_tilde + half)


# 17. IC para tasa de Poisson (exacto)
def ic_poisson(lam_hat: float, confidence: float = 0.95) -> Tuple[float, float]:
    # lam_hat = k / t, pero aquí asumimos t=1; para generalizar, escalar por t
    # Usamos cuantiles chi2 equivalentes a través de la relación con gamma
    alpha = 1 - confidence
    k = lam_hat  # si lam_hat es entero cuenta, usamos k=int(lam_hat)
    if lam_hat < 0:
        raise ValueError("lambda no puede ser negativo")
    # Aproximación mediante normal si no es entera; alternativa: usar intervalos exactos basados en chi2
    # Implementación exacta cuando k es entero no negativo
    k_int = int(round(lam_hat))
    lower = 0.0 if k_int == 0 else 0.5 * np.square(t.ppf(alpha / 2, df=2 * k_int))  # placeholder, corregimos abajo
    # Reemplazar por cuantiles chi2 usando scipy.stats: chi2.ppf
    from scipy.stats import chi2

    if k_int == 0:
        lower = 0.0
    else:
        lower = 0.5 * chi2.ppf(alpha / 2, df=2 * k_int)
    upper = 0.5 * chi2.ppf(1 - alpha / 2, df=2 * (k_int + 1))
    return lower, upper


# 18. Demostración numérica de máximo en p=0.5
def n_maximo_en_p(pts: Sequence[float], me: float, confidence: float = 0.95) -> float:
    valores = [n_para_proporcion(me, p, confidence) for p in pts]
    return float(max(valores))


# 19. IC por transformación logit
def ic_logit_proporcion(p_hat: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    # Evitar 0/1 extremos
    eps = 1e-9
    p = min(max(p_hat, eps), 1 - eps)
    logit = math.log(p / (1 - p))
    se = math.sqrt(1 / (n * p * (1 - p)))
    z = z_crit(confidence)
    lower_logit = logit - z * se
    upper_logit = logit + z * se
    def inv_logit(x: float) -> float:
        return 1 / (1 + math.exp(-x))
    return inv_logit(lower_logit), inv_logit(upper_logit)


# 20. Regla de selección de método para IC de proporciones
def metodo_ic_recomendado(n: int, p_hat: float) -> str:
    # Reglas simples: si n*p_hat < 5 o n*(1-p_hat) < 5 => exacto; si 5-10 => Wilson/Agresti; si grande => normal
    np_ = n * p_hat
    nq_ = n * (1 - p_hat)
    if np_ < 5 or nq_ < 5:
        return "exacto"
    if np_ < 10 or nq_ < 10:
        return "wilson"
    return "normal"


# 21. IC bootstrap (percentil) para proporción
def ic_bootstrap_proporcion(datos: Sequence[int], confidence: float = 0.95, n_boot: int = 2000, seed: int = 123) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    datos = np.asarray(datos)
    n = len(datos)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(datos, size=n, replace=True)
        boots.append(sample.mean())
    alpha = 1 - confidence
    lower = float(np.quantile(boots, alpha / 2))
    upper = float(np.quantile(boots, 1 - alpha / 2))
    return lower, upper


# 22. Consistencia statsmodels vs manual (devolver ambas n)
def consistencia_tamano_muestral(me: float, p: float, confidence: float = 0.95) -> Tuple[float, float]:
    from statsmodels.stats.proportion import samplesize_confint_proportion
    alpha = 1 - confidence
    n_manual = n_para_proporcion(me, p, confidence)
    # Compatibilidad: algunas versiones usan 'half_width', otras 'half_length'
    try:
        n_sm = samplesize_confint_proportion(p, half_width=me, alpha=alpha)
    except TypeError:
        n_sm = samplesize_confint_proportion(p, half_length=me, alpha=alpha)
    return float(n_manual), float(n_sm)

