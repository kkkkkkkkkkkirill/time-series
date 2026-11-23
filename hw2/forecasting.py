"""Подготовка ряда из ДЗ № 1, модели и оценка прогнозов во времени."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

HW2 = Path(__file__).resolve().parent
PRODUCT = "85099B"
INPUT = HW2 / "input" / "series_85099B.csv"


@dataclass(frozen=True)
class Experiment:
    label: str
    unit: str
    horizons: tuple[int, ...]
    test_size: int
    naive_lag: int
    lags: tuple[int, ...]
    window: int


EXPERIMENTS = {
    "D": Experiment("День", "шт. за день", (1, 7, 31, 61), 100, 6, (1, 2, 6, 12), 6),
    "W": Experiment("Неделя", "шт. на 6 рабочих дней", (1, 4, 8), 17, 1, (1, 2, 4), 4),
    "M": Experiment("Месяц", "шт. на 26 рабочих дней", (1, 2), 4, 1, (1, 2), 2),
}


def rebuild_series() -> pd.DataFrame:
    """Переиспользует загрузчик, очистку и агрегацию из ДЗ № 1."""
    hw1 = HW2.parent / "hw1"
    sys.path.insert(0, str(hw1))
    from retail_ts.data import (
        SalesSeriesBuilder,
        StoreCalendar,
        TransactionCleaner,
        load_online_retail,
    )

    raw = load_online_retail(hw1 / "data")
    calendar = StoreCalendar(raw)
    transactions = TransactionCleaner(PRODUCT).transform(raw)
    builder = SalesSeriesBuilder(transactions, calendar)
    daily = builder.daily_grid()
    if calendar.last_timestamp.hour < 17:
        daily = daily.loc[daily.index < calendar.end]
    weekly = builder.weekly().query("not is_partial")
    monthly = builder.monthly().query("not is_partial")
    series = {
        "D": daily,
        "W": builder.calendar_adjusted(weekly, standard_days=6),
        "M": builder.calendar_adjusted(monthly, standard_days=26),
    }
    return pd.concat(
        [
            pd.DataFrame({"frequency": freq, "period": row.index, "actual": row.to_numpy()})
            for freq, row in series.items()
        ],
        ignore_index=True,
    )


def load_series(rebuild: bool = False) -> dict[str, pd.Series]:
    """По умолчанию читает зафиксированный результат подготовки из ДЗ № 1."""
    data = rebuild_series() if rebuild else pd.read_csv(INPUT, parse_dates=["period"])
    result = {}
    for freq in EXPERIMENTS:
        subset = data.loc[data["frequency"] == freq].set_index("period")["actual"]
        index = pd.DatetimeIndex(subset.index)
        if index.empty or not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError(f"{freq}: индекс должен быть непустым, уникальным и возрастающим")
        if freq == "D":
            expected = pd.date_range(index[0], index[-1], freq="D")
            expected = expected[expected.dayofweek != 5]
        else:
            expected = pd.date_range(index[0], index[-1], freq={"W": "W-SAT", "M": "MS"}[freq])
        if not index.equals(expected):
            raise ValueError(f"{freq}: нарушена сетка периодов")
        observed = subset.dropna()
        if not np.isfinite(observed).all() or (observed < 0).any():
            raise ValueError(f"{freq}: некорректное значение продаж")
        result[freq] = subset
    return result


def train_values(history: pd.Series, lag: int) -> np.ndarray:
    """Заполняет закрытые периоды только по предыдущим значениям."""
    values = history.to_numpy(dtype=float, copy=True)
    for i in np.flatnonzero(np.isnan(values)):
        previous = i - lag if i >= lag else i - 1
        if previous < 0:
            raise ValueError("Первое обучающее наблюдение отсутствует")
        values[i] = values[previous]
    return values


def naive(history: pd.Series, future: pd.DatetimeIndex, spec: Experiment) -> np.ndarray:
    """Повторяет последнее наблюдение или последний цикл Вс–Пт."""
    values = train_values(history, spec.naive_lag)
    return np.resize(values[-spec.naive_lag :], len(future))


def ets(history: pd.Series, future: pd.DatetimeIndex, spec: Experiment) -> np.ndarray:
    """Оценивает модель Холта с затухающим трендом и дневной сезонностью."""
    daily = spec.naive_lag > 1
    values = train_values(history, spec.naive_lag)
    scale = max(1.0, float(values.mean()))
    model = ExponentialSmoothing(
        values / scale,
        trend="add",
        damped_trend=True,
        seasonal="add" if daily else None,
        seasonal_periods=spec.naive_lag if daily else None,
        initialization_method="estimated",
    )
    fitted = model.fit(damping_trend=0.90)
    if not fitted.mle_retvals.success:
        raise RuntimeError(f"ETS: не сошлась оптимизация на срезе {history.index[-1]:%Y-%m-%d}")
    return np.asarray(fitted.forecast(len(future))) * scale


def features(values: np.ndarray, pos: int, date: pd.Timestamp, spec: Experiment) -> list[float]:
    """Использует только лаги и календарную метку прогнозируемого периода."""
    row = [values[pos - lag] for lag in spec.lags]
    row.append(float(np.mean(values[pos - spec.window : pos])))
    if spec.naive_lag > 1:
        row.extend(float(date.dayofweek == weekday) for weekday in (6, 0, 1, 2, 3, 4))
    return row


def boosting(history: pd.Series, future: pd.DatetimeIndex, spec: Experiment) -> np.ndarray:
    """Обучает одношаговый градиентный бустинг и прогнозирует рекурсивно."""
    observed = history.to_numpy(dtype=float)
    values = train_values(history, spec.naive_lag)
    warmup = max(max(spec.lags), spec.window)
    positions = [i for i in range(warmup, len(values)) if np.isfinite(observed[i])]
    train_x = [features(values, i, history.index[i], spec) for i in positions]
    model = GradientBoostingRegressor(
        loss="huber",
        n_estimators=80,
        learning_rate=0.05,
        max_depth=1 if spec.label == "Месяц" else 2,
        min_samples_leaf=2 if spec.label == "Месяц" else 4,
        random_state=42,
    )
    model.fit(train_x, observed[positions])
    extended = np.concatenate((values, np.empty(len(future))))
    for i, date in enumerate(future, start=len(values)):
        step = model.predict([features(extended, i, date, spec)])[0]
        extended[i] = max(0.0, step)
    return extended[len(values) :]


Forecaster = Callable[[pd.Series, pd.DatetimeIndex, Experiment], np.ndarray]
MODELS: dict[str, Forecaster] = {"Naive": naive, "ETS": ets, "Gradient Boosting": boosting}


def initial_size(series: pd.Series, spec: Experiment) -> int:
    n_train = len(series) - spec.test_size
    max_lag = max(max(spec.lags), spec.window, spec.naive_lag)
    if n_train <= max_lag + 2 or spec.test_size < max(spec.horizons):
        raise ValueError(f"{spec.label}: недостаточно данных для обучения или теста")
    return n_train


def describe_splits(series_by_freq: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for freq, spec in EXPERIMENTS.items():
        series = series_by_freq[freq]
        cut = initial_size(series, spec)
        common = series.iloc[cut + max(spec.horizons) - 1 :].dropna()
        rows.append(
            {
                "frequency": freq,
                "periods": len(series),
                "observed": int(series.notna().sum()),
                "initial_train": cut,
                "initial_train_last_period": series.index[cut - 1],
                "test_periods": spec.test_size,
                "test_first_period": series.index[cut],
                "test_last_period": series.index[-1],
                "common_first_target": common.index[0],
                "common_last_target": common.index[-1],
                "common_targets": len(common),
            }
        )
    return pd.DataFrame(rows)


def backtest(series: pd.Series, freq: str) -> pd.DataFrame:
    """Фиксирует все траектории из расширяющегося окна внутри тестового хвоста."""
    spec = EXPERIMENTS[freq]
    cut = initial_size(series, spec)
    max_horizon = max(spec.horizons)
    common = series.iloc[cut + max_horizon - 1 :].dropna().index
    parts = []
    for stop in range(cut, len(series)):
        history = series.iloc[:stop]
        actual = series.iloc[stop : stop + max_horizon]
        origin = history.index[-1]
        as_of = origin + pd.offsets.MonthEnd(0) if freq == "M" else origin
        if as_of >= actual.index[0]:
            raise ValueError("Прогноз должен относиться к периодам после момента выпуска")
        step = np.arange(1, len(actual) + 1)
        scored = actual.index.isin(common)
        scored &= np.isin(step, spec.horizons)
        for model_name, forecast in MODELS.items():
            predicted = np.maximum(0.0, forecast(history, actual.index, spec))
            if predicted.shape != (len(actual),) or not np.isfinite(predicted).all():
                raise ValueError(f"{freq}, {model_name}, {as_of:%Y-%m-%d}: неверный прогноз")
            parts.append(
                pd.DataFrame(
                    {
                        "frequency": freq,
                        "model": model_name,
                        "as_of": as_of,
                        "target_period": actual.index,
                        "step": step,
                        "actual": actual.to_numpy(),
                        "forecast": predicted,
                        "scored": scored,
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def mase_scale(train: pd.Series, lag: int) -> float:
    """Средняя наивная ошибка на исходных, наблюдаемых парах первого окна."""
    scale = float(train.diff(lag).abs().mean())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("MASE не определена для данного обучающего ряда")
    return scale


def evaluate(paths: pd.DataFrame, series_by_freq: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Оценивает ровно h-й шаг на общей для всех h группе дат."""
    points = paths.loc[paths["scored"]].drop(columns="scored").rename(columns={"step": "horizon"})
    points = points.assign(
        error=points["forecast"] - points["actual"],
        abs_error=lambda table: table["error"].abs(),
        squared_error=lambda table: table["error"].pow(2),
    ).reset_index(drop=True)
    metrics = (
        points.groupby(["frequency", "horizon", "model"], sort=False)
        .agg(n=("error", "size"), MAE=("abs_error", "mean"), MSE=("squared_error", "mean"))
        .reset_index()
    )
    scales = {
        freq: mase_scale(series.iloc[:initial_size(series, EXPERIMENTS[freq])], EXPERIMENTS[freq].naive_lag)
        for freq, series in series_by_freq.items()
    }
    metrics["RMSE"] = np.sqrt(metrics.pop("MSE"))
    metrics["MASE"] = metrics["MAE"] / metrics["frequency"].map(scales)
    metrics["MASE_scale"] = metrics["frequency"].map(scales)

    # Каждая строка points задаёт конкретную траекторию 1..h.
    keys = ["frequency", "as_of", "model"]
    scored_paths = points[keys + ["horizon"]].merge(paths, on=keys)
    valid = (scored_paths["step"] <= scored_paths["horizon"]) & scored_paths["actual"].notna()
    scored_paths = scored_paths.loc[valid].assign(
        abs_error=lambda table: (table["forecast"] - table["actual"]).abs()
    )
    groups = ["frequency", "horizon", "model"]
    path_metrics = scored_paths.groupby(groups, sort=False).agg(
        path_n=("abs_error", "size"), PathMAE=("abs_error", "mean")
    )
    metrics = metrics.merge(path_metrics, on=groups, validate="one_to_one")

    for freq, spec in EXPERIMENTS.items():
        series = series_by_freq[freq]
        common = series.iloc[initial_size(series, spec) + max(spec.horizons) - 1 :].dropna().index
        for horizon in spec.horizons:
            for model in MODELS:
                selected = points.query("frequency == @freq and horizon == @horizon and model == @model")
                if not pd.DatetimeIndex(selected["target_period"]).equals(common):
                    raise ValueError(f"{freq}/{horizon}/{model}: метрики рассчитаны на разных датах")
    return metrics, points


def best_by_mae(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.loc[metrics.groupby(["frequency", "horizon"], sort=False)["MAE"].idxmin()].reset_index(drop=True)
