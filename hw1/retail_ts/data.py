"""Загрузка датасета Online Retail, очистка транзакций и построение временных рядов продаж.

Основные сущности:
    * :func:`load_online_retail` — загрузка (с автоматическим скачиванием) и кэширование датасета;
    * :func:`describe_raw` — структура сырых данных (типы, пропуски, уникальные значения);
    * :class:`TransactionCleaner` — очистка транзакций выбранного товара с журналом решений;
    * :class:`StoreCalendar` — календарь работы магазина (рабочие / нерабочие дни);
    * :class:`SalesSeriesBuilder` — агрегация продаж по дням, неделям и месяцам.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
XLSX_NAME = "Online Retail.xlsx"

# Государственные праздники Великобритании в периоде данных (магазин — британский).
UK_BANK_HOLIDAYS = {
    "2010-12-24": "Сочельник (предпраздничное закрытие)",
    "2010-12-26": "Рождественские каникулы",
    "2010-12-27": "Christmas Day (substitute day)",
    "2010-12-28": "Boxing Day (substitute day)",
    "2010-12-29": "Рождественские каникулы",
    "2010-12-30": "Рождественские каникулы",
    "2010-12-31": "Рождественские каникулы",
    "2011-01-02": "Рождественские каникулы",
    "2011-01-03": "New Year's Day (substitute day)",
    "2011-04-22": "Good Friday",
    "2011-04-24": "Easter Sunday",
    "2011-04-25": "Easter Monday",
    "2011-04-29": "Royal Wedding (разовый выходной)",
    "2011-05-02": "Early May Bank Holiday",
    "2011-05-30": "Spring Bank Holiday",
    "2011-08-29": "Summer Bank Holiday",
}

RAW_DTYPES = {"InvoiceNo": str, "StockCode": str, "Description": str, "Country": str}


# --------------------------------------------------------------------------------------
# Загрузка
# --------------------------------------------------------------------------------------
def load_online_retail(data_dir: str | Path = "data", url: str = DATA_URL) -> pd.DataFrame:
    """Загружает датасет Online Retail.

    Если в ``data_dir`` нет ни кэша ``online_retail.parquet``, ни ``Online Retail.xlsx``,
    архив скачивается с UCI. После первого чтения Excel-файла создаётся parquet-кэш
    (чтение Excel занимает ~1 минуту, parquet — доли секунды).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "online_retail.parquet"
    xlsx = data_dir / XLSX_NAME

    if cache.exists():
        return pd.read_parquet(cache)

    if not xlsx.exists():
        with urllib.request.urlopen(url) as resp:  # noqa: S310 — официальный источник UCI
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extract(XLSX_NAME, data_dir)

    df = pd.read_excel(xlsx, dtype=RAW_DTYPES)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    try:
        df.to_parquet(cache)
    except ImportError:  # pyarrow не установлен — просто работаем без кэша
        pass
    return df


def describe_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Сводка по столбцам сырых данных: тип, число пропусков, доля пропусков, число уникальных."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "missing_share_%": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(),
            "example": df.iloc[0],
        }
    )


def top_products(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Товары с наибольшим числом дней продаж (кандидаты для анализа временного ряда)."""
    sales = df[(df["Quantity"] > 0) & ~df["InvoiceNo"].str.startswith("C")]
    agg = sales.groupby("StockCode").agg(
        description=("Description", lambda s: s.mode().iat[0] if s.notna().any() else None),
        units=("Quantity", "sum"),
        invoices=("InvoiceNo", "nunique"),
        selling_days=("InvoiceDate", lambda s: s.dt.normalize().nunique()),
    )
    return agg.sort_values(["selling_days", "units"], ascending=False).head(n)


# --------------------------------------------------------------------------------------
# Очистка транзакций
# --------------------------------------------------------------------------------------
@dataclass
class TransactionCleaner:
    """Отбор и очистка транзакций одного товара.

    Правила (каждое фиксируется в :attr:`log`):
        1. отбор по ``StockCode`` (а не по описанию: описание может меняться/отсутствовать);
        2. отмены (``InvoiceNo`` начинается с ``C``) исключаются из показателя «продано»
           и сохраняются отдельно — они относятся к ранее оформленным заказам, и их вычитание
           в день отмены искажало бы динамику спроса;
        3. строки с ``UnitPrice <= 0`` исключаются — это служебные корректировки / образцы,
           а не продажи;
        4. полные дубликаты строк исключаются (повтор одной и той же позиции
           в одном счёте с той же минутой — техническое задвоение).
    """

    stock_code: str
    drop_duplicates: bool = True
    log: list[dict] = field(default_factory=list, init=False)
    cancellations: pd.DataFrame | None = field(default=None, init=False)

    def _record(self, step: str, before: int, after: int, comment: str) -> None:
        self.log.append(
            {"шаг": step, "строк до": before, "строк после": after, "удалено": before - after,
             "комментарий": comment}
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log = []
        prod = df[df["StockCode"] == self.stock_code].copy()
        self._record("отбор товара", len(df), len(prod), f"StockCode == {self.stock_code}")

        is_cancel = prod["InvoiceNo"].str.startswith("C")
        self.cancellations = prod[is_cancel].copy()
        before = len(prod)
        prod = prod[~is_cancel]
        self._record("отмены", before, len(prod), "InvoiceNo начинается с 'C' — сохранены отдельно")

        before = len(prod)
        prod = prod[prod["Quantity"] > 0]
        self._record("неположительное количество", before, len(prod), "Quantity <= 0 без признака отмены")

        before = len(prod)
        prod = prod[prod["UnitPrice"] > 0]
        self._record("нулевая цена", before, len(prod), "UnitPrice <= 0: корректировки/образцы")

        if self.drop_duplicates:
            before = len(prod)
            prod = prod.drop_duplicates()
            self._record("полные дубликаты", before, len(prod), "идентичные строки")

        prod["Revenue"] = prod["Quantity"] * prod["UnitPrice"]
        prod["Date"] = prod["InvoiceDate"].dt.normalize()
        prod["CustomerType"] = np.where(prod["CustomerID"].isna(), "без ID (розница)", "с ID")
        return prod.reset_index(drop=True)

    @property
    def report(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)


# --------------------------------------------------------------------------------------
# Календарь работы магазина
# --------------------------------------------------------------------------------------
class StoreCalendar:
    """Календарь работы магазина, восстановленный по ВСЕМ транзакциям датасета.

    День считается рабочим, если в нём есть хотя бы одна транзакция по любому товару.
    Это позволяет отличить:
        * «магазин работал, но товар не продавался» → истинный ноль продаж;
        * «магазин не работал» (суббота, праздник) → наблюдения нет, это НЕ ноль и НЕ пропуск данных.
    """

    def __init__(self, df: pd.DataFrame):
        days = df["InvoiceDate"].dt.normalize()
        self.trading_days = pd.DatetimeIndex(np.sort(days.unique()), name="Date")
        self.start, self.end = self.trading_days.min(), self.trading_days.max()
        self.calendar_days = pd.date_range(self.start, self.end, freq="D", name="Date")
        self.last_timestamp = df["InvoiceDate"].max()

    @property
    def closed_days(self) -> pd.DataFrame:
        closed = self.calendar_days.difference(self.trading_days)
        out = pd.DataFrame(index=closed)
        out["weekday"] = closed.day_name()
        out["reason"] = [
            "суббота (магазин не работает)" if d.dayofweek == 5
            else UK_BANK_HOLIDAYS.get(str(d.date()), "неизвестно")
            for d in closed
        ]
        out["type"] = np.where(out["weekday"] == "Saturday", "weekly_closure", "holiday")
        return out

    @property
    def holidays(self) -> pd.DatetimeIndex:
        cd = self.closed_days
        return cd.index[cd["type"] == "holiday"]

    def summary(self) -> pd.Series:
        cd = self.closed_days
        return pd.Series(
            {
                "первая дата": self.start.date(),
                "последняя дата": self.end.date(),
                "календарных дней": len(self.calendar_days),
                "рабочих дней": len(self.trading_days),
                "нерабочих суббот": int((cd["type"] == "weekly_closure").sum()),
                "праздничных/каникулярных дней (кроме суббот)": int((cd["type"] == "holiday").sum()),
                "последняя транзакция": self.last_timestamp,
            }
        )

    def trading_week_grid(self) -> pd.DatetimeIndex:
        """Сетка «торговой недели» Вс–Пт: все календарные дни, кроме суббот.

        Субботы исключаются систематически (магазин не работает по субботам никогда),
        поэтому сетка остаётся регулярной с периодом 6 дней.
        """
        return self.calendar_days[self.calendar_days.dayofweek != 5]


# --------------------------------------------------------------------------------------
# Построение рядов
# --------------------------------------------------------------------------------------
METRICS = {"units": "Quantity", "revenue": "Revenue", "orders": "InvoiceNo"}


class SalesSeriesBuilder:
    """Агрегирует очищенные транзакции товара в ряды разной частоты.

    Parameters
    ----------
    transactions : очищенные транзакции товара (результат :class:`TransactionCleaner`).
    calendar     : :class:`StoreCalendar`, построенный по всему датасету.
    """

    def __init__(self, transactions: pd.DataFrame, calendar: StoreCalendar):
        self.tx = transactions
        self.calendar = calendar

    def _daily_values(self, metric: str) -> pd.Series:
        col = METRICS[metric]
        grp = self.tx.groupby("Date")[col]
        return grp.nunique() if metric == "orders" else grp.sum()

    def daily(self, metric: str = "units") -> pd.DataFrame:
        """Дневной ряд на полном календаре со статусом каждого дня.

        status:
            * ``sold``    — магазин работал, товар продавался;
            * ``no_sales``— магазин работал, товар не продавался → value = 0 (истинный ноль);
            * ``closed``  — магазин закрыт (суббота/праздник) → value = NaN (наблюдения нет).
        """
        cal = self.calendar
        values = self._daily_values(metric).reindex(cal.calendar_days)
        is_open = cal.calendar_days.isin(cal.trading_days)
        status = np.where(~is_open, "closed", np.where(values.notna(), "sold", "no_sales"))
        out = pd.DataFrame({"value": values, "status": status}, index=cal.calendar_days)
        out.loc[out["status"] == "no_sales", "value"] = 0.0
        out["weekday"] = out.index.day_name()
        return out

    def daily_trading(self, metric: str = "units") -> pd.Series:
        """Ряд только по рабочим дням магазина (нерабочие дни исключены из индекса)."""
        d = self.daily(metric)
        s = d.loc[d["status"] != "closed", "value"].astype(float)
        s.name = metric
        return s

    def daily_grid(self, metric: str = "units") -> pd.Series:
        """Ряд на регулярной сетке Вс–Пт; праздники остаются NaN (для последующей обработки)."""
        d = self.daily(metric)
        s = d["value"].reindex(self.calendar.trading_week_grid()).astype(float)
        s.name = metric
        return s

    def _aggregate(self, metric: str, freq: str) -> pd.DataFrame:
        d = self.daily(metric)
        open_mask = d["status"] != "closed"
        grp = d.resample(freq)
        out = pd.DataFrame(
            {
                "value": grp["value"].sum(min_count=1),
                "trading_days": open_mask.resample(freq).sum(),
                "calendar_days_covered": d["value"].resample(freq).size(),
            }
        )
        # Период без единого рабочего дня (например, рождественская неделя) — это НЕ ноль продаж,
        # а отсутствие наблюдения: оставляем NaN. Рабочие дни без продаж уже учтены как 0.
        out.loc[out["trading_days"] == 0, "value"] = np.nan
        out["per_trading_day"] = out["value"] / out["trading_days"].replace(0, np.nan)
        return out

    def weekly(self, metric: str = "units") -> pd.DataFrame:
        """Недельный ряд. Неделя Вс–Сб (``W-SAT``) совпадает с торговой неделей магазина Вс–Пт.

        Неделя помечается как неполная (``is_partial``), если календарь данных покрывает
        её не целиком (первая неделя начинается в среду 01.12.2010).
        """
        out = self._aggregate(metric, "W-SAT")
        out["is_partial"] = out["calendar_days_covered"] < 7
        # последняя неделя: данные заканчиваются в пятницу 09.12.2011 в 12:50 — день неполный
        last_week = out.index[-1]
        if self.calendar.last_timestamp.hour < 17:
            out.loc[last_week, "is_partial"] = True
        out.index.name = "week_end"
        return out

    def monthly(self, metric: str = "units") -> pd.DataFrame:
        """Месячный ряд. Декабрь 2011 покрыт только до 09.12 и помечается как неполный."""
        out = self._aggregate(metric, "MS")
        days_in_month = out.index.days_in_month
        out["is_partial"] = out["calendar_days_covered"] < days_in_month
        out.index.name = "month"
        return out

    @staticmethod
    def calendar_adjusted(agg: pd.DataFrame, standard_days: int) -> pd.Series:
        """Корректировка на число рабочих дней: продажи в пересчёте на «стандартный» период.

        adjusted = (продажи / рабочих дней) × standard_days. Убирает искажение, вызванное
        праздниками (неделя с 3 рабочими днями не должна выглядеть как падение спроса вдвое).
        """
        return (agg["per_trading_day"] * standard_days).rename("calendar_adjusted")

    def all_frequencies(self, metric: str = "units") -> dict[str, pd.Series]:
        """Удобный доступ к рядам всех частот (полные периоды, без неполных краёв)."""
        w, m = self.weekly(metric), self.monthly(metric)
        return {
            "D": self.daily_trading(metric),
            "W": w.loc[~w["is_partial"], "value"],
            "M": m.loc[~m["is_partial"], "value"],
        }
