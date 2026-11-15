# EDA Report — Paid Media Dataset

Generated from `unified_performance.csv` (36,490 rows, 2024-07-01 to 2026-06-30)

## Channel summary statistics

| channel   |   n_rows |   total_spend |   total_revenue |   mean_daily_spend |   median_daily_spend |   spend_std |   mean_roas |   median_roas |   pct_imputed_revenue |
|:----------|---------:|--------------:|----------------:|-------------------:|---------------------:|------------:|------------:|--------------:|----------------------:|
| Amazon    |    13140 |   3.63067e+06 |     9.37082e+06 |             276.31 |               213.45 |      216.66 |        2.7  |          2.7  |                  0.02 |
| Google    |    10220 |   2.3073e+06  |     6.90525e+06 |             225.76 |               182.94 |      163.5  |        3.17 |          3.07 |                  0.02 |
| Meta      |    13130 |   3.6516e+06  |     1.10825e+07 |             278.11 |               216.56 |      215.36 |        3.21 |          2.66 |                  0.03 |

## Key observations

- **Blended ROAS across the dataset is 2.85x**, below the project's 4.0x target -- see `03_roas_boxplot.png`.
- **1,831 rows (5.0%)** are IQR-based spend outliers (mostly BFCM/Christmas spikes) -- handled via winsorization in feature engineering, not deletion.
- **Imputed-revenue share by channel** ranges 2.0%-3.0% -- see `06_missingness.png`.
- **Weekly seasonality**: weekend revenue is visibly higher across all channels (`04_weekly_seasonality.png`), consistent with the generator's design and typical DTC apparel behavior.
- **Spend/revenue correlation** is strong but channel-dependent (`05_correlation_heatmap.png`) -- this directly informs which features matter for forecasting.
- **Hero SKUs outperform tail SKUs on ROAS** in most channels (`07_sku_tier_performance.png`), consistent with the budget-tiering logic in the generator.