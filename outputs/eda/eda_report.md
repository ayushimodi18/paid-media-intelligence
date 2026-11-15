# EDA Report — Paid Media Dataset

Generated from `unified_performance.csv` (39,410 rows, 2024-07-01 to 2026-06-30)

## Channel summary statistics

| channel   |   n_rows |   total_spend |   total_revenue |   mean_spend_per_sku_day |   median_spend_per_sku_day |   spend_std |   mean_roas |   median_roas |   pct_imputed_revenue |
|:----------|---------:|--------------:|----------------:|-------------------------:|---------------------------:|------------:|------------:|--------------:|----------------------:|
| Amazon    |   13,140 |     3,630,673 |       9,370,820 |                   276.31 |                     213.45 |      216.66 |        2.70 |          2.70 |                 0.020 |
| Google    |   13,140 |     3,655,484 |      10,877,723 |                   278.20 |                     215.12 |      221.20 |        3.16 |          3.08 |                 0.020 |
| Meta      |   13,130 |     3,651,598 |      11,082,459 |                   278.11 |                     216.56 |      215.36 |        3.21 |          2.66 |                 0.030 |

## Key observations

- **Blended ROAS across the dataset is 2.86x**, below the project's 4.0x target -- see `03_roas_boxplot.png`.
- **1,937 rows (4.9%)** are IQR-based spend outliers -- 32% fall in the Nov 20-Dec 31 holiday window (2.8x that window's share of rows), and 68% come from just three SKUs (SKU-TEE-BLK-M, SKU-TEE-BLK-W, SKU-TEE-WHT-M). Handled via winsorization in feature engineering, not deletion.
- **Imputed-revenue share by channel** ranges 2.0%-3.0% -- see `06_missingness.png`.
- **Weekend revenue lift** of 17-20% over weekdays across channels (`04_weekly_seasonality.png`): Amazon +17.5%, Google +20.0%, Meta +18.9%.
- **Daily spend/revenue correlation** is strong but channel-dependent (`05_correlation_heatmap.png`): Amazon 0.89, Google 0.88, Meta 0.69 -- Meta's weaker link is consistent with its noisier attribution and harder forecasting.
- **ROAS is nearly flat across SKU tiers** (max spread 0.03x within a channel) -- tiering drives *how much* is spent per SKU, not how efficiently it converts (`07_sku_tier_performance.png`): Amazon hero 2.70x / core 2.72x / tail 2.69x; Google hero 3.14x / core 3.17x / tail 3.17x; Meta hero 3.20x / core 3.22x / tail 3.21x.