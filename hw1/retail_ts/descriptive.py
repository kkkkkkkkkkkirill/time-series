"""Описательные статистики и дополнительные характеристики ряда продаж."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

WEEKDAY_ORDER = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def descriptive_stats(series: dict[str, pd.Series] | pd.Series) -> pd.DataFrame:
    """Основные описательные статистики для одного или нескольких рядов."""
    if isinstance(series, pd.Series):
        series = {series.name or "series": series}
    rows = {}
    for name, s in series.items():
        s = s.dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        rows[name] = {
            "n": len(s),
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "CV (std/mean)": s.std() / s.mean(),
            "min": s.min(),
            "Q1": q1,
            "Q3": q3,
            "IQR": q3 - q1,
            "max": s.max(),
            "skewness": stats.skew(s),
            "excess kurtosis": stats.kurtosis(s),
            "zero share, %": (s == 0).mean() * 100,
            "sum": s.sum(),
        }
    return pd.DataFrame(rows).T


def intermittency(s: pd.Series) -> pd.Series:
    """Классификация спроса Syntetos–Boylan (2005) по ADI и CV².

    ADI — средний интервал между ненулевыми продажами; CV² — квадрат коэф. вариации
    ненулевых продаж. Пороги: ADI = 1.32, CV² = 0.49.
    """
    nz = s[s > 0]
    adi = len(s) / len(nz)
    cv2 = (nz.std() / nz.mean()) ** 2
    if adi < 1.32:
        kind = "smooth" if cv2 < 0.49 else "erratic"
    else:
        kind = "intermittent" if cv2 < 0.49 else "lumpy"
    return pd.Series({"ADI": adi, "CV²": cv2, "класс спроса": kind})


def rolling_stats(s: pd.Series, window: int) -> pd.DataFrame:
    """Скользящие среднее, стандартное отклонение и коэффициент вариации."""
    r = s.rolling(window, min_periods=window)
    out = pd.DataFrame({"value": s, "rolling_mean": r.mean(), "rolling_std": r.std()})
    out["rolling_cv"] = out["rolling_std"] / out["rolling_mean"]
    return out


def level_spread_by_period(s: pd.Series, freq: str = "MS") -> pd.DataFrame:
    """Уровень и разброс внутри каждого периода (например, месяца) для дневного ряда."""
    g = s.groupby(pd.Grouper(freq=freq))
    out = pd.DataFrame({"mean": g.mean(), "median": g.median(), "std": g.std(), "n": g.size()})
    out["cv"] = out["std"] / out["mean"]
    return out


def mean_variance_relation(s: pd.Series, freq: str = "MS") -> pd.Series:
    """Связь уровня и разброса по периодам (например, месяцам) для дневного ряда.

    * корреляция Спирмена mean ↔ std;
    * наклон b в log(std) ~ log(mean) и его робастный аналог log(IQR) ~ log(median).
      b ≈ 0 → аддитивная структура (λ ≈ 1), b ≈ 1 → разброс ∝ уровню (λ ≈ 0, log),
      b ≈ 0.5 → sqrt (λ ≈ 0.5). Оценка λ Бокса–Кокса ≈ 1 − b.
    """
    t = level_spread_by_period(s, freq).dropna()
    g = s.groupby(pd.Grouper(freq=freq))
    iqr = g.quantile(0.75) - g.quantile(0.25)
    med = g.median()
    ok = (iqr > 0) & (med > 0)
    rho, p = stats.spearmanr(t["mean"], t["std"])
    slope = np.polyfit(np.log(t["mean"]), np.log(t["std"]), 1)[0]
    rslope = np.polyfit(np.log(med[ok]), np.log(iqr[ok]), 1)[0]
    return pd.Series({"Spearman rho(mean, std)": rho, "p-value": p,
                      "наклон log std ~ log mean": slope, "λ ≈ 1 − наклон": 1 - slope,
                      "робастный наклон log IQR ~ log median": rslope, "λ ≈ 1 − робастный наклон": 1 - rslope})


def variance_homogeneity(s: pd.Series, freq: str = "QS") -> pd.Series:
    """Тесты Левене (Brown–Forsythe, медианный) и Флигнера: H0 — дисперсии по периодам равны."""
    groups = [g.dropna().values for _, g in s.groupby(pd.Grouper(freq=freq)) if len(g.dropna()) > 5]
    lev = stats.levene(*groups, center="median")
    fl = stats.fligner(*groups)
    return pd.Series({"Levene (median) stat": lev.statistic, "Levene p-value": lev.pvalue,
                      "Fligner stat": fl.statistic, "Fligner p-value": fl.pvalue, "число групп": len(groups)})


def weekday_profile(s: pd.Series) -> pd.DataFrame:
    """Средние продажи, медиана, разброс и индекс дня недели (среднее дня / общее среднее)."""
    df = pd.DataFrame({"value": s, "weekday": s.index.day_name()})
    g = df.groupby("weekday")["value"]
    out = pd.DataFrame({"mean": g.mean(), "median": g.median(), "std": g.std(), "n": g.size()})
    out = out.reindex([d for d in WEEKDAY_ORDER if d in out.index])
    out["index (mean / overall mean)"] = out["mean"] / s.mean()
    return out


def kruskal_by_group(s: pd.Series, groups: pd.Series | np.ndarray) -> pd.Series:
    """Критерий Краскела–Уоллиса: H0 — распределения значений одинаковы во всех группах.

    Непараметрический аналог ANOVA — подходит для скошенных продаж.
    """
    df = pd.DataFrame({"v": s.values, "g": np.asarray(groups)})
    samples = [g["v"].values for _, g in df.groupby("g")]
    h, p = stats.kruskal(*samples)
    n = len(df)
    k = len(samples)
    eps2 = (h - k + 1) / (n - k)  # размер эффекта epsilon²
    return pd.Series({"H": h, "p-value": p, "число групп": k, "epsilon² (размер эффекта)": eps2})


def price_profile(transactions: pd.DataFrame) -> pd.DataFrame:
    """Цены и объёмы по типам покупателей (оптовые клиенты с ID vs розница без ID)."""
    g = transactions.groupby("CustomerType")
    out = pd.DataFrame(
        {
            "строк": g.size(),
            "единиц": g["Quantity"].sum(),
            "доля единиц, %": g["Quantity"].sum() / transactions["Quantity"].sum() * 100,
            "медианная цена": g["UnitPrice"].median(),
            "средняя цена (взвеш.)": g["Revenue"].sum() / g["Quantity"].sum(),
            "медиана единиц в строке": g["Quantity"].median(),
        }
    )
    return out


def concentration(transactions: pd.DataFrame, by: str = "CustomerID", top: int = 10) -> pd.Series:
    """Концентрация продаж: доля единиц, приходящаяся на top-N клиентов/стран."""
    units = transactions.dropna(subset=[by]).groupby(by)["Quantity"].sum().sort_values(ascending=False)
    total = transactions["Quantity"].sum()
    return pd.Series(
        {
            f"уникальных ({by})": units.size,
            f"доля top-1, %": units.iloc[:1].sum() / total * 100,
            f"доля top-{top}, %": units.iloc[:top].sum() / total * 100,
        }
    )


def order_size_distribution(transactions: pd.DataFrame, bins=(0, 10, 50, 100, 500, np.inf)) -> pd.DataFrame:
    """Распределение размера строки заказа: сколько строк и какая доля единиц в каждой корзине."""
    cat = pd.cut(transactions["Quantity"], bins=bins, right=True)
    g = transactions.groupby(cat, observed=False)["Quantity"]
    out = pd.DataFrame({"строк": g.size(), "единиц": g.sum()})
    out["доля строк, %"] = out["строк"] / out["строк"].sum() * 100
    out["доля единиц, %"] = out["единиц"] / out["единиц"].sum() * 100
    out.index = out.index.astype(str)
    return out
