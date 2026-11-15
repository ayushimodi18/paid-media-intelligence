"""
Data validation layer.

Two tiers of checks, both of which FAIL LOUDLY (raise) rather than silently
passing bad data downstream -- a production pipeline should stop, not guess:

  1. Schema validation (pandera): column presence, dtypes, non-null
     constraints, value ranges (spend >= 0, percentages in [0,1], etc.)
  2. Business-rule validation: cross-field logic that a schema can't express
     alone -- e.g. revenue/spend implying an implausible ROAS, duplicate
     (date, channel, sku) keys, date-range continuity, referential
     integrity of sku against the canonical product master.

Returns a structured ValidationReport so the pipeline can log warnings vs.
hard-fail on errors, and so the Streamlit app can surface a "data health"
panel instead of a raw stack trace.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from ingest_reconcile import CANON

# ---------------------------------------------------------------- schemas
RAW_EXPORT_SCHEMA = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "platform_sku_id": Column(str, nullable=False),
        "true_sku": Column(str, nullable=False),
        "channel": Column(str, Check.isin(["Meta", "Google", "Amazon"])),
        "spend": Column(float, Check.ge(0), nullable=False),
        "impressions": Column(float, Check.ge(0)),
        "clicks": Column(float, Check.ge(0)),
        "revenue_attributed": Column(float, Check.ge(0), nullable=True),  # nullable: missingness is expected & tracked
        "new_customer_pct": Column(float, Check.in_range(0, 1), nullable=True),
        "sku_tier": Column(str, Check.isin(["hero", "core", "tail"])),
    },
    strict=False,  # allow channel-specific extra columns (campaign_objective, ad_type, etc.)
    coerce=True,
)

UNIFIED_SCHEMA = DataFrameSchema(
    {
        "date": Column(pa.DateTime, nullable=False),
        "channel": Column(str, Check.isin(["Meta", "Google", "Amazon"])),
        "sku": Column(str, Check.isin(list(CANON.keys())), nullable=False),  # referential integrity
        "spend": Column(float, Check.ge(0), nullable=False),
        "revenue_attributed": Column(float, Check.ge(0), nullable=False),  # post-imputation: no nulls allowed
        "roas": Column(float, Check.ge(0)),
    },
    strict=False,
    coerce=True,
)


@dataclass
class ValidationReport:
    source: str
    passed: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def add_error(self, msg: str):
        self.passed = False
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = [f"[{self.source}] {'PASS' if self.passed else 'FAIL'}"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)


def validate_raw_export(df: pd.DataFrame, source: str) -> ValidationReport:
    report = ValidationReport(source=source)
    try:
        RAW_EXPORT_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        report.add_error(f"Schema validation failed: {len(exc.failure_cases)} failing rows. "
                          f"Top failure: {exc.failure_cases.iloc[0].to_dict() if len(exc.failure_cases) else 'n/a'}")

    # business-rule checks beyond schema
    dupes = df.duplicated(subset=["date", "platform_sku_id"]).sum()
    if dupes > 0:
        report.add_error(f"{dupes} duplicate (date, platform_sku_id) rows -- would double-count spend.")

    null_frac = df["revenue_attributed"].isna().mean()
    if null_frac > 0.10:
        report.add_warning(f"{null_frac:.1%} of rows have missing revenue -- above the 10% comfort threshold, "
                            f"investigate before imputing blindly.")
    elif null_frac > 0:
        report.add_warning(f"{null_frac:.1%} of rows have missing revenue (expected -- will be imputed downstream).")

    date_min, date_max = df["date"].min(), df["date"].max()
    expected_days = (date_max - date_min).days + 1
    actual_days = df["date"].nunique()
    if actual_days < expected_days:
        report.add_warning(f"{expected_days - actual_days} calendar day(s) missing entirely from the date range "
                            f"{date_min.date()} -> {date_max.date()} (known outage windows are expected here).")

    implied_roas = (df["revenue_attributed"] / df["spend"].replace(0, np.nan))
    extreme = (implied_roas > 50).sum()
    if extreme > 0:
        report.add_warning(f"{extreme} rows have implied ROAS > 50x -- plausible for near-zero spend rows, "
                            f"but worth a sanity check.")

    return report


def validate_unified(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport(source="unified_performance")
    try:
        UNIFIED_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        report.add_error(f"Schema validation failed: {len(exc.failure_cases)} failing rows.")

    if df["revenue_attributed"].isna().any():
        report.add_error("Unified table still contains nulls in revenue_attributed -- "
                          "imputation step should have eliminated these before this point.")

    unknown_skus = set(df["sku"]) - set(CANON.keys())
    if unknown_skus:
        report.add_error(f"Unified table contains SKUs not in the canonical master: {unknown_skus}")

    return report


def run_all_validations(meta, google, amazon, unified) -> list[ValidationReport]:
    reports = [
        validate_raw_export(meta, "meta_raw"),
        validate_raw_export(google, "google_raw"),
        validate_raw_export(amazon, "amazon_raw"),
        validate_unified(unified),
    ]
    return reports


if __name__ == "__main__":
    DATA = Path(__file__).resolve().parents[1] / "data"
    meta = pd.read_csv(DATA / "meta_ads_export.csv", parse_dates=["date"])
    google = pd.read_csv(DATA / "google_ads_export.csv", parse_dates=["date"])
    amazon = pd.read_csv(DATA / "amazon_ads_export.csv", parse_dates=["date"])
    unified = pd.read_csv(DATA / "unified_performance.csv", parse_dates=["date"])

    reports = run_all_validations(meta, google, amazon, unified)
    for r in reports:
        print(r.summary())
        print()

    n_failed = sum(1 for r in reports if not r.passed)
    print(f"{'ALL CHECKS PASSED' if n_failed == 0 else f'{n_failed} VALIDATION STAGE(S) FAILED'}")
