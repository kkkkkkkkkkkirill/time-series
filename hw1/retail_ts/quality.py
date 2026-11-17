"""Диагностика качества ряда: пропуски и выбросы."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

MAD_SCALE = 1.4826  # MAD * 1.4826 ≈ σ для нормального распределения


# --------------------------------------------------------------------------------------
# Пропуски
# --------------------------------------------------------------------------------------
def _runs(mask: pd.Series) -> pd.DataFrame:
    """Непрерывные отрезки, на которых mask == True."""
    mask = mask.astype(bool)
    if not mask.any():
        return pd.DataFrame(columns=["start", "end", "length"])
    groups = (mask != mask.shift()).cumsum()
    runs = (
        mask[mask]
        .groupby(groups[mask])
        .apply(lambda g: pd.Series({"start": g.index.min(), "end": g.index.max(), "length": len(g)}))
        .unstack()
        .reset_index(drop=True)
    )
    return runs


@dataclass
class MissingDiagnostics:
    """Диагностика пропусков дневного ряда, построенного :meth:`SalesSeriesBuilder.daily`.

    Разделяет три ситуации, которые нельзя смешивать:
        * ``closed``   — магазин не работал: наблюдения нет по объективной причине;
        * ``no_sales`` — магазин работал, товар не купили: истинный ноль;
        * ``NaN в исходных данных`` — отсутствие значения в рабочий день (ошибка данных).
    """

    daily: pd.DataFrame
    closed_days: pd.DataFrame

    def status_counts(self) -> pd.DataFrame:
        counts = self.daily["status"].value_counts().rename("дней")
        share = (counts / counts.sum() * 100).round(2).rename("доля, %")
        return pd.concat([counts, share], axis=1)

    def unexplained_missing(self) -> pd.DataFrame:
        """Рабочие дни, где значение отсутствует (должно быть пусто)."""
        d = self.daily
        return d[(d["status"] != "closed") & d["value"].isna()]

    def no_sales_days(self) -> pd.DataFrame:
        return self.daily[self.daily["status"] == "no_sales"]

    def closed_runs(self) -> pd.DataFrame:
        """Отрезки подряд идущих нерабочих дней (кроме одиночных суббот) с причинами."""
        mask = self.daily["status"] == "closed"
        runs = _runs(mask)
        if runs.empty:
            return runs
        runs["reasons"] = [
            "; ".join(sorted(set(self.closed_days.loc[r.start:r.end, "reason"])))
            for r in runs.itertuples()
        ]
        return runs[runs["length"] > 1].reset_index(drop=True)

    def calendar_gaps(self, index: pd.DatetimeIndex, freq: str) -> int:
        """Число пропущенных меток в регулярной сетке заданной частоты."""
        full = pd.date_range(index.min(), index.max(), freq=freq)
        return len(full.difference(index))


def impute_closed_days(grid_series: pd.Series, method: str = "time") -> tuple[pd.Series, pd.Index]:
    """Заполняет NaN нерабочих периодов (праздники на сетке Вс–Пт, закрытая неделя) интерполяцией.

    Используется ТОЛЬКО для методов, требующих регулярного ряда без пропусков
    (STL, ACF, тесты стационарности). Заполненные даты возвращаются отдельно,
    чтобы их можно было явно отметить на графиках.
    """
    filled_idx = grid_series.index[grid_series.isna()]
    filled = grid_series.interpolate(method=method, limit_direction="both")
    return filled, filled_idx


# --------------------------------------------------------------------------------------
# Выбросы
# --------------------------------------------------------------------------------------
def iqr_outliers(s: pd.Series, k: float = 1.5) -> pd.DataFrame:
    """Правило Тьюки: x < Q1 - k·IQR или x > Q3 + k·IQR (не учитывает тренд и сезонность)."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return pd.DataFrame({"value": s, "lower": lo, "upper": hi, "is_outlier": (s < lo) | (s > hi)})


def hampel_outliers(s: pd.Series, window: int = 13, n_sigma: float = 3.0) -> pd.DataFrame:
    """Фильтр Хампеля: отклонение от скользящей медианы больше n_sigma робастных σ (MAD).

    Учитывает локальный уровень ряда (тренд), устойчив к самим выбросам.
    """
    med = s.rolling(window, center=True, min_periods=window // 2 + 1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=window // 2 + 1).median()
    sigma = MAD_SCALE * mad
    lo, hi = med - n_sigma * sigma, med + n_sigma * sigma
    return pd.DataFrame(
        {"value": s, "median": med, "lower": lo, "upper": hi, "is_outlier": (s - med).abs() > n_sigma * sigma}
    )


def stl_residual_outliers(s: pd.Series, period: int, z: float = 3.5, log: bool = True) -> pd.DataFrame:
    """Выбросы по остаткам робастной STL: |robust z| > z.

    Остаток очищен и от тренда, и от сезонности, поэтому «обычный» высокий четверг
    не будет ошибочно помечен как выброс. Логарифм (log1p) стабилизирует дисперсию.
    """
    x = np.log1p(s) if log else s
    res = STL(x, period=period, robust=True).fit()
    resid = res.resid
    med = resid.median()
    robust_z = (resid - med) / (MAD_SCALE * (resid - med).abs().median())
    return pd.DataFrame({"value": s, "resid": resid, "robust_z": robust_z, "is_outlier": robust_z.abs() > z})


class OutlierDetector:
    """Запускает несколько методов поиска выбросов и строит консенсус.

    Итоговый флаг ``consensus`` = выброс найден минимум ``min_votes`` методами.
    """

    def __init__(self, period: int, hampel_window: int = 13, n_sigma: float = 3.0,
                 iqr_k: float = 1.5, stl_z: float = 3.5, min_votes: int = 2):
        self.period = period
        self.hampel_window = hampel_window
        self.n_sigma = n_sigma
        self.iqr_k = iqr_k
        self.stl_z = stl_z
        self.min_votes = min_votes
        self.details_: dict[str, pd.DataFrame] = {}

    def fit(self, s: pd.Series) -> "OutlierDetector":
        self.series_ = s
        self.details_ = {
            "IQR": iqr_outliers(s, self.iqr_k),
            "Hampel": hampel_outliers(s, self.hampel_window, self.n_sigma),
            "STL-resid": stl_residual_outliers(s, self.period, self.stl_z),
        }
        flags = pd.DataFrame({k: v["is_outlier"] for k, v in self.details_.items()})
        flags["votes"] = flags.sum(axis=1)
        flags["consensus"] = flags["votes"] >= self.min_votes
        flags["value"] = s
        self.flags_ = flags
        return self

    def summary(self) -> pd.DataFrame:
        f = self.flags_
        rows = {m: int(f[m].sum()) for m in self.details_}
        rows["консенсус (≥%d методов)" % self.min_votes] = int(f["consensus"].sum())
        out = pd.Series(rows, name="число выбросов").to_frame()
        out["доля, %"] = (out["число выбросов"] / len(f) * 100).round(2)
        return out

    @property
    def outliers(self) -> pd.DataFrame:
        return self.flags_[self.flags_["consensus"]]

    def upper_bound(self) -> pd.Series:
        """Локальная верхняя граница Хампеля — используется для опционального ограничения."""
        return self.details_["Hampel"]["upper"]


def explain_outliers(dates: pd.Index, transactions: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Для каждой выбросной даты — какие заказы сформировали продажи (крупнейшие строки)."""
    rows = []
    for d in dates:
        day = transactions[transactions["Date"] == d]
        total = day["Quantity"].sum()
        big = day.nlargest(top, "Quantity")
        rows.append(
            {
                "date": d.date(),
                "units_total": int(total),
                "n_lines": len(day),
                "largest_order_units": int(big["Quantity"].iat[0]) if len(big) else 0,
                "largest_share_%": round(big["Quantity"].iat[0] / total * 100, 1) if total else 0,
                "largest_customer": big["CustomerID"].iat[0] if len(big) else None,
                "largest_country": big["Country"].iat[0] if len(big) else None,
            }
        )
    return pd.DataFrame(rows)


def cap_outliers(s: pd.Series, flags: pd.Series, upper: pd.Series) -> pd.Series:
    """Ограничивает (winsorize) только отмеченные значения локальной верхней границей."""
    capped = s.copy()
    mask = flags & (s > upper)
    capped[mask] = upper[mask]
    return capped
