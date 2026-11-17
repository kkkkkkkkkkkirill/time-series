"""retail_ts — переиспользуемые инструменты анализа временных рядов продаж (Online Retail)."""
from .data import (DATA_URL, SalesSeriesBuilder, StoreCalendar, TransactionCleaner, describe_raw,
                   load_online_retail, top_products)
from .decomposition import (SeasonalityTester, STLDecomposer, classical_decomposition, mann_kendall, periodogram_peaks,
                            seasonality_identifiable)
from .descriptive import (concentration, descriptive_stats, intermittency, kruskal_by_group,
                          level_spread_by_period, mean_variance_relation, order_size_distribution,
                          price_profile, rolling_stats, variance_homogeneity, weekday_profile)
from .quality import (MissingDiagnostics, OutlierDetector, cap_outliers, explain_outliers, hampel_outliers,
                      impute_closed_days, iqr_outliers, stl_residual_outliers)
from .stationarity import (SeriesTransformer, StationarityTester, adf_test, boxcox1p, combine_verdict,
                           guerrero_lambda, inv_boxcox1p, kpss_test, variance_stabilization_table,
                           ljung_box, overdifferencing_check, rolling_moments_test, suggest_differencing, zivot_andrews_test)

__all__ = [name for name in dir() if not name.startswith("_")]
