"""Проверка стационарности и преобразования ряда к стационарному виду."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from scipy.special import boxcox1p as _boxcox1p, inv_boxcox1p as _inv_boxcox1p


def boxcox1p(x, lam: float):
    """Преобразование Бокса–Кокса для (1 + y): ((1+y)^λ − 1)/λ, при λ = 0 — log(1 + y).

    Сдвиг на 1 позволяет обрабатывать нулевые продажи.
    """
    out = _boxcox1p(np.asarray(x, dtype=float), lam)
    return pd.Series(out, index=x.index, name=getattr(x, "name", None)) if isinstance(x, pd.Series) else out


def inv_boxcox1p(x, lam: float):
    out = _inv_boxcox1p(np.asarray(x, dtype=float), lam)
    return pd.Series(out, index=x.index, name=getattr(x, "name", None)) if isinstance(x, pd.Series) else out


def guerrero_lambda(s: pd.Series, block: int, robust: bool = False,
                    grid: np.ndarray = np.linspace(-1, 2, 301)) -> float:
    """Метод Герреро (1993) выбора λ Бокса–Кокса.

    Ряд режется на блоки длины ``block``; ищется λ, минимизирующая коэффициент вариации
    отношения σ_i / μ_i^(1−λ) по блокам (т.е. делающая разброс независимым от уровня).
    ``robust=True`` — вместо (std, mean) используются (IQR, median): устойчиво к выбросам.
    """
    x = s.dropna() + 1
    groups = [g.values for _, g in x.groupby(np.arange(len(x)) // block) if len(g) >= block // 2]
    if robust:
        spread = np.array([np.quantile(g, 0.75) - np.quantile(g, 0.25) for g in groups])
        level = np.array([np.median(g) for g in groups])
    else:
        spread = np.array([g.std(ddof=1) for g in groups])
        level = np.array([g.mean() for g in groups])
    cvs = []
    for lam in grid:
        r = spread / level ** (1 - lam)
        cvs.append(r.std(ddof=1) / r.mean())
    return float(grid[int(np.argmin(cvs))])


def variance_stabilization_table(s: pd.Series, lambdas, freq: str = "MS", n_splits: int = 4) -> pd.DataFrame:
    """Насколько каждое λ Бокса–Кокса делает разброс независимым от уровня.

    * Spearman ρ(среднее, СКО) по месяцам: цель — ρ ≈ 0 (при ρ > 0 разброс растёт с уровнем,
      т.е. преобразование слишком слабое, при ρ < 0 — «перекоррекция»);
    * Левене по ``n_splits`` последовательным отрезкам и по кварталам: H0 — дисперсии равны.
    """
    rows = {}
    for lam in lambdas:
        x = s if lam is None else boxcox1p(s, lam)
        g = x.groupby(pd.Grouper(freq=freq))
        rho, rho_p = stats.spearmanr(g.mean(), g.std())
        parts = np.array_split(x.dropna().values, n_splits)
        quarters = [q.dropna().values for _, q in x.groupby(pd.Grouper(freq="QS")) if q.notna().sum() > 5]
        label = "без преобразования (λ=1)" if lam is None else ("log1p (λ=0)" if lam == 0 else f"λ={lam:.2f}")
        rows[label] = {
            "λ": 1.0 if lam is None else lam,
            "ρ(среднее, СКО) по месяцам": rho, "p-value ρ": rho_p,
            f"Levene p ({n_splits} отрезка)": stats.levene(*parts, center="median").pvalue,
            "Levene p (кварталы)": stats.levene(*quarters, center="median").pvalue,
            "СКО по отрезкам": ", ".join(f"{p.std(ddof=1):.2f}" for p in parts),
        }
    return pd.DataFrame(rows).T


def adf_test(s: pd.Series, regression: str = "c", autolag: str = "AIC") -> dict:
    """Расширенный тест Дики–Фуллера.

    H0: ряд имеет единичный корень (нестационарен). H1: ряд стационарен
    (при regression='ct' — стационарен относительно детерминированного тренда).
    p < alpha → отвергаем H0 → свидетельство В ПОЛЬЗУ стационарности.
    """
    stat, p, lags, nobs, crit, _ = adfuller(s.dropna(), regression=regression, autolag=autolag)
    return {"test": f"ADF ({regression})", "statistic": stat, "p-value": p, "lags": lags, "nobs": nobs,
            "crit 5%": crit["5%"]}


def kpss_test(s: pd.Series, regression: str = "c", nlags: str | int = "auto") -> dict:
    """Тест Квятковского–Филлипса–Шмидта–Шина.

    H0: ряд стационарен (regression='c' — относительно уровня, 'ct' — относительно тренда).
    p < alpha → отвергаем H0 → свидетельство ПРОТИВ стационарности.
    statsmodels интерполирует p-value по таблице в диапазоне [0.01, 0.1]; за его пределами
    p-value — граница («≤0.01» или «≥0.1»).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        stat, p, lags, crit = kpss(s.dropna(), regression=regression, nlags=nlags)
    return {"test": f"KPSS ({regression})", "statistic": stat, "p-value": p, "lags": lags,
            "nobs": int(s.notna().sum()), "crit 5%": crit["5%"]}


def zivot_andrews_test(s: pd.Series, regression: str = "c") -> dict:
    """Тест Зивота–Эндрюса: H0 — единичный корень; H1 — стационарность с одним структурным сдвигом.

    Полезен, когда ADF не отвергает H0 из-за сдвига уровня (ADF теряет мощность при сдвигах).
    """
    stat, p, crit, lags, bp = zivot_andrews(s.dropna(), regression=regression, autolag="AIC")
    bp_date = s.dropna().index[bp] if bp is not None else None
    return {"test": f"Zivot–Andrews ({regression})", "statistic": stat, "p-value": p, "lags": lags,
            "nobs": int(s.notna().sum()), "crit 5%": crit["5%"], "break": bp_date}


def ljung_box(s: pd.Series, lags: list[int] | int = 10) -> pd.DataFrame:
    """Тест Льюнга–Бокса: H0 — автокорреляции до лага k равны нулю (белый шум)."""
    return acorr_ljungbox(s.dropna(), lags=lags, return_df=True)


def combine_verdict(adf_p: float, kpss_p: float, alpha: float = 0.05) -> str:
    """Совместная интерпретация ADF и KPSS (у тестов противоположные нулевые гипотезы)."""
    adf_stationary = adf_p < alpha
    kpss_stationary = kpss_p >= alpha
    if adf_stationary and kpss_stationary:
        return "стационарен (оба теста согласны)"
    if not adf_stationary and not kpss_stationary:
        return "нестационарен (оба теста согласны)"
    if adf_stationary and not kpss_stationary:
        return "противоречие: ADF — стационарен, KPSS — нет → вероятна разностная стационарность / сдвиг уровня"
    return "противоречие: ADF — единичный корень, KPSS — стационарен → низкая мощность / трендовая стационарность"


@dataclass
class StationarityTester:
    """Набор тестов стационарности для одного или нескольких рядов.

    Для каждого ряда выполняются ADF и KPSS с одной и той же спецификацией детерминированной
    части (``regression``: 'c' — константа, 'ct' — константа + линейный тренд), что делает
    результаты сопоставимыми, и выносится совместный вердикт.
    """

    regression: str = "c"
    alpha: float = 0.05
    results_: pd.DataFrame | None = field(default=None, init=False)

    def test(self, s: pd.Series) -> dict:
        a = adf_test(s, self.regression)
        k = kpss_test(s, self.regression)
        return {
            "n": a["nobs"] + a["lags"] + 1,
            "ADF stat": a["statistic"], "ADF p-value": a["p-value"], "ADF lags": a["lags"],
            "KPSS stat": k["statistic"], "KPSS p-value": k["p-value"], "KPSS lags": k["lags"],
            "вывод": combine_verdict(a["p-value"], k["p-value"], self.alpha),
        }

    def run(self, series: dict[str, pd.Series]) -> pd.DataFrame:
        self.results_ = pd.DataFrame({name: self.test(s) for name, s in series.items()}).T
        return self.results_


def rolling_moments_test(s: pd.Series, n_splits: int = 4) -> pd.DataFrame:
    """Сравнение среднего и дисперсии по равным последовательным отрезкам ряда.

    Краскел–Уоллис (H0: одинаковое положение) и Левене (H0: равные дисперсии) —
    непараметрическая проверка постоянства моментов, дополняющая ADF/KPSS.
    """
    parts = np.array_split(s.dropna().values, n_splits)
    kw = stats.kruskal(*parts)
    lev = stats.levene(*parts, center="median")
    table = pd.DataFrame({"part": range(1, n_splits + 1),
                          "mean": [p.mean() for p in parts], "std": [p.std(ddof=1) for p in parts]})
    table.attrs["kruskal_p"] = kw.pvalue
    table.attrs["levene_p"] = lev.pvalue
    return table


# --------------------------------------------------------------------------------------
# Преобразования
# --------------------------------------------------------------------------------------
class SeriesTransformer:
    """Последовательность обратимых преобразований ряда.

    Поддерживаемые шаги:
        * ``("log1p",)``        — log(1 + y): стабилизация дисперсии, допускает нули;
        * ``("boxcox", λ)``     — Бокс–Кокс на (1 + y) с заданной λ (λ = 0 ⇔ log1p);
        * ``("diff", lag)``     — разность порядка lag (1 — тренд/уровень, period — сезонность);
        * ``("detrend",)``      — вычитание линейного тренда МНК (для трендово-стационарного ряда).

    Пример: ``SeriesTransformer([("log1p",), ("diff", 6)]).fit_transform(y)``.
    Метод :meth:`inverse_transform` восстанавливает ряд в исходной шкале (нужно для прогнозов).
    """

    def __init__(self, steps: list[tuple]):
        self.steps = steps
        self._state: list[dict] = []

    @property
    def name(self) -> str:
        parts = []
        for st in self.steps:
            if st[0] == "log1p":
                parts.append("log1p")
            elif st[0] == "boxcox":
                parts.append(f"BoxCox(λ={st[1]:.2f})")
            elif st[0] == "diff":
                parts.append(f"Δ{st[1]}" if st[1] > 1 else "Δ")
            elif st[0] == "detrend":
                parts.append("−лин.тренд")
        return " → ".join(parts) if parts else "исходный"

    def fit_transform(self, s: pd.Series) -> pd.Series:
        self._state = []
        x = s.astype(float).copy()
        for st in self.steps:
            kind = st[0]
            if kind == "log1p":
                self._state.append({"kind": kind})
                x = np.log1p(x)
            elif kind == "boxcox":
                lam = st[1]
                self._lam = lam
                self._state.append({"kind": kind, "lam": lam})
                x = boxcox1p(x, lam)
            elif kind == "diff":
                lag = st[1]
                self._state.append({"kind": kind, "lag": lag, "head": x.iloc[:lag].copy()})
                x = x.diff(lag).iloc[lag:]
            elif kind == "detrend":
                t = np.arange(len(x))
                coef = np.polyfit(t, x.values, 1)
                self._state.append({"kind": kind, "coef": coef, "t0": 0})
                x = x - np.polyval(coef, t)
            else:
                raise ValueError(f"неизвестный шаг {kind}")
        x.name = self.name
        return x

    def inverse_transform(self, x: pd.Series) -> pd.Series:
        y = x.copy()
        for st in reversed(self._state):
            if st["kind"] == "diff":
                lag, head = st["lag"], st["head"]
                vals = list(head.values)
                for i, v in enumerate(y.values):
                    vals.append(v + vals[i])
                y = pd.Series(vals, index=head.index.append(y.index))
            elif st["kind"] == "log1p":
                y = np.expm1(y)
            elif st["kind"] == "detrend":
                y = y + np.polyval(st["coef"], np.arange(len(y)))
            elif st["kind"] == "boxcox":
                y = inv_boxcox1p(y, st["lam"])
        return y


def overdifferencing_check(original: pd.Series, transformed: dict[str, pd.Series]) -> pd.DataFrame:
    """Признаки переразностирования: ACF(1) около −0.5 и рост дисперсии после разности.

    Если ряд трендово-стационарен, взятие разности вносит необратимую MA-компоненту
    (единичный корень в MA-части), что проявляется как ACF(1) → −0.5 и Var(Δy) > Var(y).
    """
    from statsmodels.tsa.stattools import acf

    rows = {}
    base_var = original.var()
    for name, s in transformed.items():
        x = s.dropna()
        rows[name] = {"ACF(1)": acf(x, nlags=1, fft=True)[1], "дисперсия": x.var(),
                      "дисперсия / дисперсия исходного": x.var() / base_var}
    return pd.DataFrame(rows).T


def suggest_differencing(s: pd.Series, max_d: int = 2, alpha: float = 0.05, regression: str = "c") -> int:
    """Минимальный порядок обычного дифференцирования, при котором KPSS не отвергает стационарность
    (логика ``ndiffs`` из forecast/pmdarima)."""
    x = s.dropna()
    for d in range(max_d + 1):
        if kpss_test(x, regression)["p-value"] >= alpha:
            return d
        x = x.diff().dropna()
    return max_d
