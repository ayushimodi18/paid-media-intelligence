"""
M1 - Cross-platform ingestion and SKU reconciliation.

Real problem: Meta keys SKUs as "FB_SKU_TEE_BLK_M", Google keys them as
"goog::t-shirt-black-men". There is NO shared join key out of the box.
This module builds a reconciliation map (a deterministic lookup table you'd
normally seed from a PIM/product feed) and unifies both exports onto a single
canonical schema, flagging anything that fails to reconcile or has missing
revenue.

In production this map would come from the brand's Shopify/PIM product
feed (SKU -> platform catalog ID), not be inferred from string matching --
the inference here is a stand-in that demonstrates detecting and repairing
a key-parsing failure (see reconcile_google).
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

# Canonical product master (this is the "source of truth" a PIM would supply)
CANON = {
    "SKU-TEE-BLK-M": dict(category="T-Shirt", color="Black", gender="Men"),
    "SKU-TEE-WHT-M": dict(category="T-Shirt", color="White", gender="Men"),
    "SKU-POLO-NVY-M": dict(category="Polo", color="Navy", gender="Men"),
    "SKU-HOODIE-GRY-M": dict(category="Hoodie", color="Grey", gender="Men"),
    "SKU-SHORT-KHK-M": dict(category="Shorts", color="Khaki", gender="Men"),
    "SKU-JACKET-BLK-M": dict(category="Jacket", color="Black", gender="Men"),
    "SKU-CREW-NVY-M": dict(category="Crewneck", color="Navy", gender="Men"),
    "SKU-TEE-BLK-W": dict(category="T-Shirt", color="Black", gender="Women"),
    "SKU-TANK-WHT-W": dict(category="Tank", color="White", gender="Women"),
    "SKU-HOODIE-PNK-W": dict(category="Hoodie", color="Pink", gender="Women"),
    "SKU-LEGGING-BLK-W": dict(category="Legging", color="Black", gender="Women"),
    "SKU-JOGGER-GRY-W": dict(category="Joggers", color="Grey", gender="Women"),
    "SKU-CARDIGAN-BLK-W": dict(category="Cardigan", color="Black", gender="Women"),
    "SKU-TEE-BLU-K": dict(category="T-Shirt", color="Blue", gender="Kids"),
    "SKU-SET-GRN-K": dict(category="Set", color="Green", gender="Kids"),
    "SKU-HOODIE-RED-K": dict(category="Hoodie", color="Red", gender="Kids"),
    "SKU-SHORT-NVY-K": dict(category="Shorts", color="Navy", gender="Kids"),
    "SKU-TANK-WHT-K": dict(category="Tank", color="White", gender="Kids"),
}


def reconcile_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sku"] = df["platform_sku_id"].str.replace("FB_", "", regex=False).str.replace("_", "-")
    df["reconciled"] = df["sku"].isin(CANON.keys())
    return df


_GOOGLE_LOOKUP = {
    (v["category"].lower(), v["color"].lower(), v["gender"].lower()): k
    for k, v in CANON.items()
}


def _google_attrs(key: str) -> str | None:
    if not isinstance(key, str) or "::" not in key:
        return None
    return key.split("::", 1)[1]


def _parse_google_naive(key: str):
    """The obvious parse: split the attribute string on '-' into exactly
    category/color/gender. Breaks when the category itself contains a hyphen
    ('t-shirt-black-men' has 4 tokens) -- the delimiter-collision failure."""
    attrs = _google_attrs(key)
    if attrs is None:
        return None
    parts = attrs.split("-")
    return _GOOGLE_LOOKUP.get(tuple(parts)) if len(parts) == 3 else None


def _parse_google_delimiter_aware(key: str):
    """Repair: color and gender are always single tokens at the END of the
    key, so split from the right and let the category keep its hyphens."""
    attrs = _google_attrs(key)
    if attrs is None:
        return None
    parts = attrs.rsplit("-", 2)
    return _GOOGLE_LOOKUP.get(tuple(parts)) if len(parts) == 3 else None


def reconcile_google(df: pd.DataFrame) -> pd.DataFrame:
    """Google key looks like goog::t-shirt-black-men -- map back to canonical SKU
    via the category/color/gender attributes, since there's no shared ID.

    Two passes, so the data-quality issue is surfaced rather than hidden:
      1. naive parse -- rows it can't map are counted as delimiter collisions
      2. delimiter-aware parse repairs them; anything still unmapped is a
         genuine unknown key and is excluded from the unified table.
    `reconcile_method` records which pass mapped each row, for auditability."""
    df = df.copy()
    naive = df["platform_sku_id"].apply(_parse_google_naive)
    repaired = df["platform_sku_id"].apply(_parse_google_delimiter_aware)
    df["sku"] = naive.fillna(repaired)
    df["reconciled"] = df["sku"].notna()
    df["reconcile_method"] = np.select(
        [naive.notna(), repaired.notna()], ["direct", "repaired_delimiter_collision"], default="failed"
    )
    return df


def reconcile_amazon(df: pd.DataFrame) -> pd.DataFrame:
    """Amazon keys by ASIN. We use the 'true_sku' passthrough column (which in
    production would come from a Seller Central SKU<->ASIN mapping export) to
    build the ASIN->canonical-SKU lookup. Any ASIN not present in that mapping
    -- e.g. a re-platformed/ghost listing -- fails reconciliation and is
    flagged, exactly like a real catalog-drift incident would surface."""
    df = df.copy()
    known_asin_to_sku = (
        df[df["true_sku"].isin(CANON.keys())]
        .drop_duplicates("platform_sku_id")
        .set_index("platform_sku_id")["true_sku"]
        .to_dict()
    )
    known_asin_to_sku.pop("B0GHOST99X", None)  # simulate the mapping table not knowing about it yet

    df["sku"] = df["platform_sku_id"].map(known_asin_to_sku)
    df["reconciled"] = df["sku"].notna()
    return df


def unify(save: bool = False) -> tuple[dict, pd.DataFrame]:
    """Reconcile all three exports onto the unified schema.

    Pure by default -- the dashboard and tests call this read-only. Only the
    pipeline passes save=True to persist data/unified_performance.csv, so
    viewing the app can never overwrite a pipeline artifact."""
    meta_raw = pd.read_csv(DATA / "meta_ads_export.csv", parse_dates=["date"])
    google_raw = pd.read_csv(DATA / "google_ads_export.csv", parse_dates=["date"])
    amazon_raw = pd.read_csv(DATA / "amazon_ads_export.csv", parse_dates=["date"])

    meta = reconcile_meta(meta_raw)
    google = reconcile_google(google_raw)
    amazon = reconcile_amazon(amazon_raw)

    report = {
        "meta_total_rows": len(meta),
        "meta_reconciled": int(meta["reconciled"].sum()),
        "meta_unreconciled": int((~meta["reconciled"]).sum()),
        "meta_missing_revenue": int(meta["revenue_attributed"].isna().sum()),
        "google_total_rows": len(google),
        "google_reconciled": int(google["reconciled"].sum()),
        "google_unreconciled": int((~google["reconciled"]).sum()),
        "google_repaired": int((google["reconcile_method"] == "repaired_delimiter_collision").sum()),
        "google_missing_revenue": int(google["revenue_attributed"].isna().sum()),
        "amazon_total_rows": len(amazon),
        "amazon_reconciled": int(amazon["reconciled"].sum()),
        "amazon_unreconciled": int((~amazon["reconciled"]).sum()),
        "amazon_missing_revenue": int(amazon["revenue_attributed"].isna().sum()),
    }

    # Impute missing revenue: channel x sku median (flagged, not silent)
    for df in (meta, google, amazon):
        df["revenue_imputed"] = df["revenue_attributed"].isna()
        df["revenue_attributed"] = df.groupby("sku")["revenue_attributed"].transform(
            lambda s: s.fillna(s.median())
        )
        df["revenue_attributed"] = df["revenue_attributed"].fillna(0)

    keep_cols = ["date", "channel", "sku", "spend", "impressions", "clicks",
                 "revenue_attributed", "new_customer_pct", "revenue_imputed"]
    meta_u = meta[meta["reconciled"]][keep_cols]
    google_u = google[google["reconciled"]][keep_cols]
    amazon_u = amazon[amazon["reconciled"]][keep_cols]

    unified = pd.concat([meta_u, google_u, amazon_u], ignore_index=True)
    unified["roas"] = (unified["revenue_attributed"] / unified["spend"]).replace([np.inf, -np.inf], 0)

    if save:
        unified.to_csv(DATA / "unified_performance.csv", index=False)
    return report, unified


if __name__ == "__main__":
    report, unified = unify(save=True)
    print("Reconciliation report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nUnified rows: {len(unified)}")
    print(unified.groupby(["channel"]).agg(spend=("spend", "sum"), revenue=("revenue_attributed", "sum")))
