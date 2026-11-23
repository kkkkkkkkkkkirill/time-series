"""Таблицы и графики сравнения прогнозов."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from forecasting import EXPERIMENTS, MODELS, best_by_mae

COLORS = {"Naive": "#747681", "ETS": "#1a6f9b", "Gradient Boosting": "#b7502c"}
ACTUAL = "#182831"


def format_dates(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_forecasts(points: pd.DataFrame, freq: str) -> plt.Figure:
    """Сравнивает оценки ровно на h шагов вперёд на одинаковых датах."""
    spec = EXPERIMENTS[freq]
    n = len(spec.horizons)
    columns = 2 if n == 4 else n
    fig, axes = plt.subplots((n + columns - 1) // columns, columns,
                             figsize=(11, 3.05 * ((n + columns - 1) // columns) + 0.65),
                             sharey=True, layout="constrained")
    fig.suptitle(f"{spec.label}. Факт и прогноз на общих тестовых датах", fontsize=15)
    selected = points.loc[points["frequency"] == freq]
    axes.flat[0].set_ylim(0, 1.08 * selected[["forecast", "actual"]].max().max())
    for ax, horizon in zip(axes.flat, spec.horizons, strict=True):
        table = selected.loc[selected["horizon"] == horizon]
        actual = table.drop_duplicates("target_period").sort_values("target_period")
        ax.plot(actual["target_period"], actual["actual"], color=ACTUAL, linewidth=2.2,
                marker="o", markersize=3, label="Факт", zorder=4)
        for name in MODELS:
            model = table.loc[table["model"] == name].sort_values("target_period")
            ax.plot(model["target_period"], model["forecast"], color=COLORS[name],
                    linewidth=1.45, marker=".", linestyle="--", label=name)
        ax.set(title=f"h = {horizon}", xlabel="Целевой период", ylabel=spec.unit)
        ax.grid(axis="y", alpha=0.23)
        format_dates(ax)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncols=4)
    return fig


def plot_first_paths(paths: pd.DataFrame, series_by_freq: dict[str, pd.Series]) -> plt.Figure:
    """Показывает траектории максимальной длины в первом тестовом окне."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), layout="constrained")
    fig.suptitle("Траектории из первого среза. Прогноз без обновления внутри горизонта", fontsize=15)
    for ax, (freq, spec) in zip(axes, EXPERIMENTS.items(), strict=True):
        selected = paths.loc[paths["frequency"] == freq]
        first = selected.loc[selected["as_of"] == selected["as_of"].min()]
        actual = first.drop_duplicates("target_period")
        origin = first["as_of"].iloc[0]
        prior = series_by_freq[freq].loc[:origin].tail(10)
        ax.axvline(origin, color="#adb7bf", linewidth=1, linestyle=":")
        ax.plot(prior.index, prior, color=ACTUAL, linewidth=1.4, marker=".", label="История")
        ax.plot(actual["target_period"], actual["actual"], color=ACTUAL, linewidth=2.2,
                marker="o", markersize=3, label="Факт", zorder=4)
        for name in MODELS:
            forecast = first.loc[first["model"] == name]
            ax.plot(forecast["target_period"], forecast["forecast"], color=COLORS[name],
                    linewidth=1.45, linestyle="--", marker=".", label=name)
        ax.set(title=f"{spec.label}, h = {max(spec.horizons)}\nСрез {origin:%d.%m.%Y}",
               xlabel="Период", ylabel=spec.unit, ylim=(0, None))
        ax.grid(axis="y", alpha=0.23)
        format_dates(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=5)
    return fig


def plot_metrics(metrics: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, layout="constrained")
    fig.suptitle("Качество по горизонту. Меньше MASE — лучше", fontsize=15)
    axes[0].set_ylim(0, 1.08 * max(1.0, metrics["MASE"].max()))
    for ax, (freq, spec) in zip(axes, EXPERIMENTS.items(), strict=True):
        subset = metrics.loc[metrics["frequency"] == freq]
        for name in MODELS:
            model = subset.loc[subset["model"] == name].set_index("horizon").loc[list(spec.horizons)]
            ax.plot(range(len(model)), model["MASE"], color=COLORS[name], marker="o",
                    linewidth=1.8, label=name)
        ax.axhline(1, color="#aeb7be", linewidth=1, linestyle=":", label="Масштаб MASE")
        ax.set(title=spec.label, xlabel="Горизонт, шагов", ylabel="MASE")
        ax.set_xticks(range(len(spec.horizons)), labels=spec.horizons)
        ax.grid(axis="y", alpha=0.23)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=4)
    return fig


def save_results(output: Path, paths: pd.DataFrame, points: pd.DataFrame,
                 metrics: pd.DataFrame, splits: pd.DataFrame) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tables = (
        ("forecast_paths", paths),
        ("forecast_points", points),
        ("metrics", metrics),
        ("best_by_horizon", best_by_mae(metrics)),
        ("splits", splits),
    )
    for name, table in tables:
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.12g")


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=155, facecolor="white")
