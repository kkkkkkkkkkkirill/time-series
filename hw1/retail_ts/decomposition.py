"""Декомпозиция ряда и статистическая проверка сезонности."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf

from .descriptive import kruskal_by_group
from .stationarity import boxcox1p, inv_boxcox1p


def min_obs_for_seasonality(period: int, cycles: int = 2) -> int:
    """Минимум наблюдений, чтобы сезонность периода ``period`` вообще была идентифицируема."""
    return period * cycles


def seasonality_identifiable(n: int, period: int, cycles: int = 2) -> bool:
    return n >= min_obs_for_seasonality(period, cycles)


@dataclass
class STLDecomposer:
    """Обёртка над STL (Seasonal-Trend decomposition using LOESS, Cleveland et al., 1990).

    Parameters
    ----------
    period   : сезонный период в шагах ряда.
    lam      : λ Бокса–Кокса для (1 + y), применяемая перед декомпозицией (None — без преобразования,
               0 — log1p, т.е. мультипликативная модель Y = T·S·R). Преобразование нужно, если
               разброс растёт вместе с уровнем: в преобразованной шкале модель становится аддитивной.
    robust   : робастная версия (выбросы получают малые веса и не искажают тренд/сезонность).
    seasonal : длина окна сглаживания сезонной компоненты (нечётная ≥ 7).
    """

    period: int
    lam: float | None = None
    robust: bool = True
    seasonal: int = 7
    trend: int | None = None
    result_: pd.DataFrame | None = field(default=None, init=False)

    def fit(self, s: pd.Series) -> "STLDecomposer":
        x = s if self.lam is None else boxcox1p(s, self.lam)
        res = STL(x, period=self.period, seasonal=self.seasonal, trend=self.trend, robust=self.robust).fit()
        self.result_ = pd.DataFrame(
            {"observed": res.observed, "trend": res.trend, "seasonal": res.seasonal,
             "resid": res.resid, "weights": res.weights}
        )
        return self

    def strength(self) -> pd.Series:
        """Сила тренда и сезонности (Wang, Smith & Hyndman, 2006).

        F_T = max(0, 1 − Var(R)/Var(T+R)),  F_S = max(0, 1 − Var(R)/Var(S+R)).
        Значения близки к 1 → компонента выражена, близки к 0 → практически отсутствует.
        Ориентир Hyndman & Athanasopoulos: F_S > 0.64 — сильная сезонность, < 0.4 — слабая.
        """
        r = self.result_
        ft = max(0.0, 1 - r["resid"].var() / (r["trend"] + r["resid"]).var())
        fs = max(0.0, 1 - r["resid"].var() / (r["seasonal"] + r["resid"]).var())
        return pd.Series({"сила тренда F_T": ft, "сила сезонности F_S": fs})

    def seasonal_profile(self) -> pd.Series:
        """Средняя сезонная компонента по позициям цикла (по меткам дня недели)."""
        r = self.result_
        key = r.index.day_name() if isinstance(r.index, pd.DatetimeIndex) else np.arange(len(r)) % self.period
        return r["seasonal"].groupby(key).mean()

    def seasonal_index(self) -> pd.Series:
        """Сезонный индекс в исходной шкале: во сколько раз продажи в данной позиции цикла
        отличаются от «среднего дня» при среднем уровне тренда.

        index_k = inv(T̄ + S_k) / inv(T̄), где inv — обратное преобразование Бокса–Кокса.
        """
        prof = self.seasonal_profile()
        if self.lam is None:
            base = self.result_["trend"].mean()
            return (base + prof) / base
        base = self.result_["trend"].mean()
        return (inv_boxcox1p(base + prof, self.lam) + 1) / (inv_boxcox1p(base, self.lam) + 1)

    def seasonal_amplitude_over_time(self, freq: str = "MS") -> pd.Series:
        """Размах сезонной компоненты (max − min) по периодам: стабильна ли сезонность."""
        s = self.result_["seasonal"]
        return s.groupby(pd.Grouper(freq=freq)).agg(lambda x: x.max() - x.min())


def classical_decomposition(s: pd.Series, period: int, model: str = "additive") -> pd.DataFrame:
    """Классическая декомпозиция скользящим средним (для сравнения с STL)."""
    res = seasonal_decompose(s, model=model, period=period, extrapolate_trend="freq")
    return pd.DataFrame({"observed": res.observed, "trend": res.trend, "seasonal": res.seasonal,
                         "resid": res.resid})


def periodogram_peaks(s: pd.Series, top: int = 5, detrend: str = "linear") -> pd.DataFrame:
    """Пики периодограммы: доминирующие периоды (в шагах ряда)."""
    x = s.dropna().values.astype(float)
    freqs, power = signal.periodogram(x, detrend=detrend, scaling="spectrum")
    mask = freqs > 0
    freqs, power = freqs[mask], power[mask]
    idx = np.argsort(power)[::-1][:top]
    return pd.DataFrame({"frequency": freqs[idx], "period (steps)": 1 / freqs[idx],
                         "power": power[idx], "share of power, %": power[idx] / power.sum() * 100})


class SeasonalityTester:
    """Комплексная проверка гипотезы о сезонности заданного периода.

    Сезонность считается установленной, только если согласуются несколько независимых
    признаков, а не один визуальный паттерн:
        1. ACF на лаге = period значимо отличается от нуля (за пределами 95% интервала Бартлетта);
        2. Краскел–Уоллис по позициям цикла: распределения отличаются (p < alpha);
        3. Сила сезонности F_S по STL превышает порог и статистически значима по
           перестановочному тесту (перемешивание детрендированного ряда разрушает сезонность);
        4. Периодограмма имеет пик на period (или его гармониках).
    """

    def __init__(self, period: int, alpha: float = 0.05, n_permutations: int = 199,
                 lam: float | None = 0.0, random_state: int = 42, stl_kwargs: dict | None = None):
        self.period = period
        self.stl_kwargs = stl_kwargs or {}
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.lam = lam
        self.rng = np.random.default_rng(random_state)

    def _fs(self, x: pd.Series) -> float:
        return STLDecomposer(self.period, **self.stl_kwargs).fit(x).strength()["сила сезонности F_S"]

    def permutation_test(self, s: pd.Series) -> tuple[float, float, np.ndarray]:
        x = s if self.lam is None else boxcox1p(s, self.lam)
        observed = self._fs(x)
        trend = STLDecomposer(self.period, **self.stl_kwargs).fit(x).result_["trend"]
        detr = (x - trend).values
        null = np.empty(self.n_permutations)
        for i in range(self.n_permutations):
            perm = pd.Series(self.rng.permutation(detr), index=x.index) + trend
            null[i] = self._fs(perm)
        p = (1 + (null >= observed).sum()) / (1 + self.n_permutations)
        return observed, p, null

    def run(self, s: pd.Series) -> pd.DataFrame:
        n = len(s)
        x = s if self.lam is None else boxcox1p(s, self.lam)
        acf_vals, confint = acf(x, nlags=self.period * 3, alpha=self.alpha, fft=True)
        lag = self.period
        half_width = (confint[lag, 1] - confint[lag, 0]) / 2
        acf_sig = abs(acf_vals[lag]) > half_width

        positions = np.arange(n) % self.period
        kw = kruskal_by_group(x - x.rolling(self.period, center=True, min_periods=1).mean(), positions)

        fs, p_perm, self.null_ = self.permutation_test(s)

        peaks = periodogram_peaks(x, top=3)
        harmonics = [self.period / k for k in (1, 2, 3)]
        peak_match = any(np.isclose(peaks["period (steps)"], h, rtol=0.05).any() for h in harmonics)

        rows = [
            ("ACF на лаге %d" % lag, f"{acf_vals[lag]:.3f} (95% граница ±{half_width:.3f})", acf_sig),
            ("Краскел–Уоллис по позициям цикла", f"H={kw['H']:.1f}, p={kw['p-value']:.2e}, ε²={kw['epsilon² (размер эффекта)']:.2f}",
             kw["p-value"] < self.alpha),
            ("Перестановочный тест для F_S (STL)", f"F_S={fs:.3f}; p={p_perm:.3f} (H0: сезонности нет)",
             p_perm < self.alpha),
            ("Пик периодограммы (top-3 периода)", ", ".join(f"{v:.2f}" for v in peaks["period (steps)"]), peak_match),
        ]
        out = pd.DataFrame(rows, columns=["признак", "результат", "подтверждает сезонность"])
        self.fs_ = fs
        self.p_perm_ = p_perm
        self.verdict_ = bool(out["подтверждает сезонность"].all())
        self.magnitude_ = self.magnitude(fs)
        return out

    @staticmethod
    def magnitude(fs: float) -> str:
        """Качественная сила сезонности по F_S (ориентиры Hyndman & Athanasopoulos)."""
        if fs >= 0.64:
            return "сильная"
        if fs >= 0.4:
            return "умеренная"
        if fs >= 0.15:
            return "слабая/умеренная: значима, но шум доминирует"
        return "очень слабая"


def mann_kendall(s: pd.Series) -> pd.Series:
    """Тест Манна–Кендалла на монотонный тренд (через τ Кендалла с временем) + наклон Сена.

    H0: монотонного тренда нет. Непараметрический, работает и на коротких рядах (n ≥ 8).
    """
    from scipy import stats

    x = s.dropna()
    t = np.arange(len(x))
    tau, p = stats.kendalltau(t, x.values)
    slope = stats.theilslopes(x.values, t)[0]
    return pd.Series({"n": len(x), "Kendall tau": tau, "p-value": p, "наклон Сена (за шаг)": slope,
                      "наклон, % от среднего за шаг": slope / x.mean() * 100})
