"""Графики. Каждая функция задаёт заголовок, подписи осей и легенду."""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from .descriptive import WEEKDAY_ORDER

COLORS = {"main": "#1f4e79", "accent": "#d1495b", "trend": "#edae49", "muted": "#8d99ae",
          "green": "#2a9d8f", "closed": "#cfd8dc"}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (13, 4.5), "figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.titlesize": 12, "legend.frameon": False,
    })


def _date_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def plot_series(s: pd.Series, title: str, ylabel: str, ax=None, label: str | None = None,
                marker: str | None = None, color: str = COLORS["main"], xlabel: str = "Дата"):
    ax = ax or plt.subplots()[1]
    ax.plot(s.index, s.values, color=color, lw=1.2, marker=marker, ms=4, label=label or s.name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if isinstance(s.index, pd.DatetimeIndex):
        _date_axis(ax)
    ax.legend(loc="upper left")
    return ax


def plot_three_frequencies(daily: pd.Series, weekly: pd.DataFrame, monthly: pd.DataFrame, ylabel: str):
    """Ряд в трёх частотах; неполные периоды выделены отдельным маркером."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    plot_series(daily, "Дневные продажи (сетка Вс–Пт; разрывы линии — праздники, магазин закрыт)", ylabel,
                ax=axes[0], label="продажи за день")

    for ax, df, name, unit in [(axes[1], weekly, "Недельные продажи (неделя Вс–Сб)", "неделю"),
                               (axes[2], monthly, "Месячные продажи", "месяц")]:
        full = df[~df["is_partial"]]
        part = df[df["is_partial"]]
        ax.plot(full.index, full["value"], marker="o", ms=4, color=COLORS["main"], label=f"продажи за {unit}")
        ax.scatter(part.index, part["value"], color=COLORS["accent"], zorder=3, s=45,
                   label="неполный период (не используется)")
        ax.set_title(name)
        ax.set_xlabel("Дата")
        ax.set_ylabel(ylabel)
        _date_axis(ax)
        ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_daily_status(daily: pd.DataFrame, title: str):
    """Календарь статусов дней: продажи / нулевые продажи / магазин закрыт."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sold = daily[daily["status"] == "sold"]
    zero = daily[daily["status"] == "no_sales"]
    closed = daily[daily["status"] == "closed"]
    hol = closed[closed["weekday"] != "Saturday"]
    sat = closed[closed["weekday"] == "Saturday"]
    ax.bar(sold.index, sold["value"], width=0.9, color=COLORS["main"], label="продажи (магазин открыт)")
    ymax = daily["value"].max()
    ax.bar(sat.index, [ymax * 0.04] * len(sat), width=0.9, color=COLORS["closed"],
           label="суббота — магазин закрыт (нет наблюдения)")
    ax.bar(hol.index, [ymax] * len(hol), width=0.9, color=COLORS["trend"], alpha=0.5,
           label="праздник/каникулы — магазин закрыт")
    ax.scatter(zero.index, [0] * len(zero), color=COLORS["accent"], marker="x", s=80, zorder=4,
               label="магазин открыт, продаж нет (истинный 0)")
    ax.set_title(title)
    ax.set_xlabel("Дата")
    ax.set_ylabel("Продано, шт.")
    _date_axis(ax)
    ax.legend(loc="upper left", ncol=2)
    return fig


def plot_outliers(s: pd.Series, flags: pd.DataFrame, hampel: pd.DataFrame, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(s.index, s.values, color=COLORS["main"], lw=1, label="продажи за день")
    ax.plot(hampel.index, hampel["median"], color=COLORS["trend"], lw=1.5, label="скользящая медиана (Хампель)")
    ax.fill_between(hampel.index, hampel["lower"].clip(lower=0), hampel["upper"], color=COLORS["trend"],
                    alpha=0.15, label="границы Хампеля ±3σ (MAD)")
    single = flags[(flags["votes"] == 1)]
    cons = flags[flags["consensus"]]
    ax.scatter(single.index, single["value"], color=COLORS["muted"], s=25, zorder=3,
               label="выброс по 1 методу")
    ax.scatter(cons.index, cons["value"], color=COLORS["accent"], s=55, zorder=4,
               label="выброс по ≥2 методам (консенсус)")
    ax.set_title(title)
    ax.set_xlabel("Дата")
    ax.set_ylabel(ylabel)
    _date_axis(ax)
    ax.legend(loc="upper left", ncol=2)
    return fig


def plot_distribution(s: pd.Series, title: str, xlabel: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(s, bins=40, color=COLORS["main"], alpha=0.85, label="частота")
    axes[0].axvline(s.mean(), color=COLORS["accent"], ls="--", label=f"среднее = {s.mean():.0f}")
    axes[0].axvline(s.median(), color=COLORS["trend"], ls="--", label=f"медиана = {s.median():.0f}")
    axes[0].set(title=f"{title}: гистограмма", xlabel=xlabel, ylabel="Число дней")
    axes[0].legend()
    axes[1].hist(np.log1p(s), bins=40, color=COLORS["green"], alpha=0.85, label="частота")
    axes[1].set(title="Гистограмма log(1 + y)", xlabel=f"log(1 + {xlabel})", ylabel="Число дней")
    axes[1].legend()
    bp = axes[2].boxplot(s, vert=True, patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(COLORS["main"])
    bp["boxes"][0].set_alpha(0.5)
    axes[2].set(title="Ящик с усами (IQR-правило)", ylabel=xlabel, xticks=[1], xticklabels=["продажи за день"])
    axes[2].plot([], [], "o", color="k", mfc="none", label="точки за 1.5·IQR")
    axes[2].legend()
    fig.tight_layout()
    return fig


def plot_rolling(roll: pd.DataFrame, title: str, ylabel: str, window_label: str):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(roll.index, roll["value"], color=COLORS["muted"], lw=0.8, label="продажи")
    axes[0].plot(roll.index, roll["rolling_mean"], color=COLORS["main"], lw=2, label=f"скользящее среднее ({window_label})")
    axes[0].set(title=f"{title}: изменение уровня", ylabel=ylabel)
    axes[0].legend(loc="upper left")
    axes[1].plot(roll.index, roll["rolling_std"], color=COLORS["accent"], lw=2, label=f"скользящее СКО ({window_label})")
    ax2 = axes[1].twinx()
    ax2.plot(roll.index, roll["rolling_cv"], color=COLORS["green"], lw=1.5, ls="--", label="коэф. вариации (правая ось)")
    ax2.set_ylabel("CV")
    ax2.grid(False)
    axes[1].set(title=f"{title}: изменение разброса", xlabel="Дата", ylabel=f"СКО, {ylabel.split(',')[-1].strip()}")
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, loc="upper left")
    _date_axis(axes[1])
    fig.tight_layout()
    return fig


def plot_level_vs_spread(table: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(table["mean"], table["std"], color=COLORS["main"], s=50, label="месяц")
    for idx, row in table.iterrows():
        ax.annotate(idx.strftime("%b %y"), (row["mean"], row["std"]), fontsize=8, xytext=(3, 3),
                    textcoords="offset points")
    k = np.polyfit(table["mean"], table["std"], 1)
    xs = np.linspace(table["mean"].min(), table["mean"].max(), 10)
    ax.plot(xs, np.polyval(k, xs), color=COLORS["accent"], ls="--", label="линейная аппроксимация")
    ax.set(title=title, xlabel="Среднее дневных продаж в месяце, шт.", ylabel="СКО дневных продаж в месяце, шт.")
    ax.legend()
    return fig


def plot_weekday(s: pd.Series, title: str, ylabel: str):
    df = pd.DataFrame({"value": s.values, "weekday": s.index.day_name()})
    order = [d for d in WEEKDAY_ORDER if d in set(df["weekday"])]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    data = [df.loc[df["weekday"] == d, "value"].values for d in order]
    bp = ax.boxplot(data, patch_artist=True, showmeans=True, meanprops=dict(marker="D", mfc="white", mec="k"))
    for b in bp["boxes"]:
        b.set_facecolor(COLORS["main"])
        b.set_alpha(0.45)
    ax.set_xticks(range(1, len(order) + 1), order)
    ax.plot([], [], "D", mfc="white", mec="k", label="среднее")
    ax.plot([], [], color="orange", label="медиана")
    ax.set(title=title, xlabel="День недели", ylabel=ylabel)
    ax.legend(loc="upper left")
    return fig


def plot_monthly_profile(table: pd.DataFrame, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    x = np.arange(len(table))
    ax.bar(x, table["mean"], color=COLORS["main"], alpha=0.8, label="среднее за рабочий день")
    ax.errorbar(x, table["mean"], yerr=table["std"], fmt="none", ecolor=COLORS["accent"], capsize=4,
                label="± СКО внутри месяца")
    ax.plot(x, table["median"], "o-", color=COLORS["trend"], label="медиана")
    ax.set_xticks(x, [i.strftime("%b\n%Y") for i in table.index])
    ax.set(title=title, xlabel="Месяц", ylabel=ylabel)
    ax.legend(loc="upper left")
    return fig


def plot_acf_pacf(s: pd.Series, lags: int, title: str, unit: str = "лаг, дней"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(s.dropna(), lags=lags, ax=axes[0], color=COLORS["main"], vlines_kwargs={"colors": COLORS["main"]})
    plot_pacf(s.dropna(), lags=lags, ax=axes[1], method="ywm", color=COLORS["main"],
              vlines_kwargs={"colors": COLORS["main"]})
    for ax, name in zip(axes, ["ACF", "PACF"]):
        ax.set_title(f"{name}: {title}")
        ax.set_xlabel(unit)
        ax.set_ylabel("корреляция")
        ax.plot([], [], color=COLORS["main"], label=f"{name} по лагам")
        ax.fill_between([], [], color="C0", alpha=0.25, label="95% доверительный интервал")
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_decomposition(comp: pd.DataFrame, title: str, ylabel: str, imputed: pd.Index | None = None):
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    specs = [("observed", "Исходный ряд", COLORS["main"]), ("trend", "Тренд", COLORS["trend"]),
             ("seasonal", "Сезонная компонента", COLORS["green"]), ("resid", "Остаток", COLORS["muted"])]
    for ax, (col, name, color) in zip(axes, specs):
        if col == "resid":
            ax.scatter(comp.index, comp[col], s=8, color=color, label=name)
            ax.axhline(0, color="k", lw=0.8)
        else:
            ax.plot(comp.index, comp[col], color=color, lw=1.3, label=name)
        if col == "observed" and imputed is not None and len(imputed):
            ax.scatter(imputed, comp.loc[imputed, col], color=COLORS["accent"], s=25, zorder=4,
                       label="интерполированные праздничные дни")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: {name.lower()}")
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("Дата")
    _date_axis(axes[-1])
    fig.tight_layout()
    return fig


def plot_seasonal_profile(profile: pd.Series, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    order = [d for d in WEEKDAY_ORDER if d in profile.index]
    p = profile.reindex(order)
    ax.bar(order, p.values, color=COLORS["green"], alpha=0.85, label="сезонный индекс")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="средний день (индекс = 1)")
    for i, v in enumerate(p.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    ax.set(title=title, xlabel="День недели", ylabel=ylabel)
    ax.legend(loc="upper right")
    return fig


def plot_permutation(null: np.ndarray, observed: float, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(null, bins=30, color=COLORS["muted"], label="F_S при перемешанных данных (H0)")
    ax.axvline(observed, color=COLORS["accent"], lw=2, label=f"наблюдаемая F_S = {observed:.3f}")
    ax.set(title=title, xlabel="Сила сезонности F_S", ylabel="Число перестановок")
    ax.legend()
    return fig


def plot_periodogram(s: pd.Series, title: str, max_period: float = 40):
    from scipy import signal
    x = s.dropna().values.astype(float)
    f, p = signal.periodogram(x, detrend="linear")
    m = f > 1 / max_period
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(1 / f[m], p[m], color=COLORS["main"], label="спектральная плотность")
    for per in (6, 3, 2):
        ax.axvline(per, color=COLORS["accent"], ls=":", lw=1)
    ax.plot([], [], color=COLORS["accent"], ls=":", label="период 6 и гармоники (3, 2)")
    ax.set(title=title, xlabel="Период, рабочих дней", ylabel="Мощность")
    ax.legend()
    return fig


def plot_transformations(series: dict[str, pd.Series], title: str):
    fig, axes = plt.subplots(len(series), 1, figsize=(13, 2.8 * len(series)), sharex=True)
    for ax, (name, s) in zip(np.atleast_1d(axes), series.items()):
        ax.plot(s.index, s.values, color=COLORS["main"], lw=0.9, label=name)
        roll = s.rolling(24, min_periods=12)
        ax.plot(s.index, roll.mean(), color=COLORS["trend"], lw=2, label="скользящее среднее (24 дня)")
        ax.plot(s.index, roll.std(), color=COLORS["accent"], lw=1.5, ls="--", label="скользящее СКО (24 дня)")
        ax.set_title(f"{title}: {name}")
        ax.set_ylabel("значение")
        ax.legend(loc="upper left", ncol=3)
    np.atleast_1d(axes)[-1].set_xlabel("Дата")
    _date_axis(np.atleast_1d(axes)[-1])
    fig.tight_layout()
    return fig


def plot_customer_mix(tx: pd.DataFrame, title: str):
    m = tx.groupby([pd.Grouper(key="Date", freq="MS"), "CustomerType"])["Quantity"].sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    bottom = np.zeros(len(m))
    for col, color in zip(m.columns, [COLORS["trend"], COLORS["main"]]):
        ax.bar(m.index, m[col], width=20, bottom=bottom, color=color, label=col)
        bottom += m[col].values
    ax.set(title=title, xlabel="Месяц", ylabel="Продано, шт.")
    _date_axis(ax)
    ax.legend(loc="upper left")
    return fig


def plot_price(tx: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for ctype, color in [("с ID", COLORS["main"]), ("без ID (розница)", COLORS["trend"])]:
        d = tx[tx["CustomerType"] == ctype]
        ax.scatter(d["InvoiceDate"], d["UnitPrice"], s=8, alpha=0.5, color=color, label=f"цена: покупатель {ctype}")
    ax.set(title=title, xlabel="Дата", ylabel="Цена за единицу, £")
    _date_axis(ax)
    ax.legend(loc="upper left")
    return fig
