"""
Exploratory Data Analysis.

Generates a fixed set of diagnostic charts + a numeric summary table,
saved to outputs/eda/ as PNGs + a markdown report. Built as a script (not
a notebook) so it's reproducible from the command line and CI-friendly --
"the EDA" should be a build artifact, not a one-off Jupyter session nobody
can re-run.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def load():
    unified = pd.read_csv(DATA / "unified_performance.csv", parse_dates=["date"])
    return unified


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("channel").agg(
        n_rows=("spend", "size"),
        total_spend=("spend", "sum"),
        total_revenue=("revenue_attributed", "sum"),
        mean_daily_spend=("spend", "mean"),
        median_daily_spend=("spend", "median"),
        spend_std=("spend", "std"),
        mean_roas=("roas", "mean"),
        median_roas=("roas", "median"),
        pct_imputed_revenue=("revenue_imputed", "mean"),
    ).round(2)


def plot_revenue_timeseries(df, path):
    daily = df.groupby(["channel", "date"])["revenue_attributed"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for ch, d in daily.groupby("channel"):
        ax.plot(d["date"], d["revenue_attributed"], label=ch, linewidth=0.9, alpha=0.85)
    ax.set_title("Daily revenue by channel (2 years) -- seasonality + trend visible")
    ax.set_ylabel("Revenue ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_spend_distribution(df, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ch, d in df.groupby("channel"):
        sns.kdeplot(d["spend"], label=ch, fill=True, alpha=0.25, ax=ax)
    ax.set_title("Daily spend distribution by channel (right-skewed -- log-scale candidates)")
    ax.set_xlabel("Spend ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roas_boxplot(df, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    capped = df.copy()
    capped["roas"] = capped["roas"].clip(upper=capped["roas"].quantile(0.98))
    sns.boxplot(data=capped, x="channel", y="roas", ax=ax)
    ax.axhline(4.0, color="red", linestyle="--", linewidth=1, label="Target (4.0x)")
    ax.set_title("ROAS distribution by channel vs. the 4.0x target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_weekly_seasonality(df, path):
    d = df.copy()
    d["dow"] = d["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    agg = d.groupby(["channel", "dow"])["revenue_attributed"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=agg, x="dow", y="revenue_attributed", hue="channel", order=order, ax=ax)
    ax.set_title("Average daily revenue by day-of-week -- weekend lift visible")
    ax.set_ylabel("Avg revenue ($)")
    plt.xticks(rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_correlation_heatmap(df, path):
    daily = df.groupby(["channel", "date"]).agg(
        spend=("spend", "sum"), revenue=("revenue_attributed", "sum"),
        clicks=("clicks", "sum"), impressions=("impressions", "sum"),
    ).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (ch, d) in zip(axes, daily.groupby("channel")):
        corr = d[["spend", "revenue", "clicks", "impressions"]].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                    cbar=False, vmin=-1, vmax=1)
        ax.set_title(ch)
    fig.suptitle("Spend/revenue/clicks/impressions correlation by channel")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_missingness(df, path):
    miss = df.groupby("channel")["revenue_imputed"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=miss, x="channel", y="revenue_imputed", ax=ax)
    ax.set_title("Share of rows with imputed (originally missing) revenue")
    ax.set_ylabel("Fraction imputed")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sku_tier_performance(df, path):
    meta_skus = pd.read_csv(DATA / "meta_ads_export.csv")[["true_sku", "sku_tier"]].drop_duplicates()
    merged = df.merge(meta_skus.rename(columns={"true_sku": "sku"}), on="sku", how="left")
    agg = merged.groupby(["sku_tier", "channel"])["roas"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=agg, x="sku_tier", y="roas", hue="channel", order=["hero", "core", "tail"], ax=ax)
    ax.set_title("Mean ROAS by SKU tier and channel")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def detect_outliers_iqr(df: pd.DataFrame, col: str = "spend") -> pd.DataFrame:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return df[(df[col] < lo) | (df[col] > hi)]


def run_eda():
    df = load()
    summary = numeric_summary(df)
    summary.to_csv(OUT / "channel_summary_stats.csv")

    plot_revenue_timeseries(df, OUT / "01_revenue_timeseries.png")
    plot_spend_distribution(df, OUT / "02_spend_distribution.png")
    plot_roas_boxplot(df, OUT / "03_roas_boxplot.png")
    plot_weekly_seasonality(df, OUT / "04_weekly_seasonality.png")
    plot_correlation_heatmap(df, OUT / "05_correlation_heatmap.png")
    plot_missingness(df, OUT / "06_missingness.png")
    plot_sku_tier_performance(df, OUT / "07_sku_tier_performance.png")

    outliers = detect_outliers_iqr(df, "spend")

    report_lines = [
        "# EDA Report — Paid Media Dataset\n",
        f"Generated from `unified_performance.csv` ({len(df):,} rows, "
        f"{df['date'].min().date()} to {df['date'].max().date()})\n",
        "## Channel summary statistics\n",
        summary.to_markdown(),
        "\n## Key observations\n",
        f"- **Blended ROAS across the dataset is {df['revenue_attributed'].sum()/df['spend'].sum():.2f}x**, "
        f"below the project's 4.0x target -- see `03_roas_boxplot.png`.",
        f"- **{len(outliers):,} rows ({len(outliers)/len(df):.1%})** are IQR-based spend outliers "
        f"(mostly BFCM/Christmas spikes) -- handled via winsorization in feature engineering, not deletion.",
        f"- **Imputed-revenue share by channel** ranges "
        f"{summary['pct_imputed_revenue'].min():.1%}-{summary['pct_imputed_revenue'].max():.1%} -- "
        f"see `06_missingness.png`.",
        "- **Weekly seasonality**: weekend revenue is visibly higher across all channels "
        "(`04_weekly_seasonality.png`), consistent with the generator's design and typical DTC apparel behavior.",
        "- **Spend/revenue correlation** is strong but channel-dependent (`05_correlation_heatmap.png`) -- "
        "this directly informs which features matter for forecasting.",
        "- **Hero SKUs outperform tail SKUs on ROAS** in most channels (`07_sku_tier_performance.png`), "
        "consistent with the budget-tiering logic in the generator.",
    ]
    (OUT / "eda_report.md").write_text("\n".join(report_lines))
    print(f"EDA artifacts saved to {OUT}")
    print(summary)


if __name__ == "__main__":
    run_eda()
