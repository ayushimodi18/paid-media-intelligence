"""
M1 - Synthetic data generator (v3 - benchmark-anchored).

Key change from v2: revenue is no longer generated from an arbitrary Hill
curve. Instead, each row is built bottom-up from a CPC and a target ROAS,
both sampled within RANGES taken from cited 2026 industry benchmark reports
(Meta/Google/Amazon, apparel/ecommerce vertical) -- see SOURCES below.
Diminishing returns emerge from auction saturation: as daily spend on a
SKU rises relative to that SKU's available efficient inventory, realized
ROAS decays from the benchmark range toward zero. This mirrors the real
mechanism (bidding into thinner, lower-intent inventory raises effective
CPC) rather than fitting an unmotivated curve.

SOURCES (compiled July 2026, all flagged as ranges because reporting
agencies disagree with each other -- normal for this kind of aggregated
industry data, not a data quality issue):
  - Meta apparel CPC ~$0.45, CTR ~1.2-2.0%, CPA ~$31; retargeting ROAS
    5-10x vs prospecting ROAS ~2-3x (get-ryze.ai 2026 Meta benchmarks;
    hawky.ai 2026 ROAS benchmarks)
  - Google ecommerce/retail CPC ~$1-2, conversion rate 3-5%, target ROAS
    3-5x (ppcchief.com WordStream-sourced 2026 PPC benchmarks; hawky.ai)
  - Amazon apparel CPC ~$2.15, ROAS ~3.35x (implied ACOS ~30%), CTR ~0.47%
    (get-ryze.ai 2026 Amazon benchmarks); category ACOS range 28-42%
    across sources (Trellis, Autron, Accrue 2026 Amazon Ads benchmark posts)

Still 100% synthetic -- no real company numbers -- but every range is
now a cited external anchor rather than an internal assumption.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)

# ---- Product catalog: 18 SKUs across Men/Women/Kids ----
# (category, color, gender, tier) -- AOV comes from CATEGORY_AOV below
SKUS = [
    ("SKU-TEE-BLK-M", "Men", "T-Shirt", "Black", "hero"),
    ("SKU-TEE-WHT-M", "Men", "T-Shirt", "White", "hero"),
    ("SKU-POLO-NVY-M", "Men", "Polo", "Navy", "core"),
    ("SKU-HOODIE-GRY-M", "Men", "Hoodie", "Grey", "core"),
    ("SKU-SHORT-KHK-M", "Men", "Shorts", "Khaki", "tail"),
    ("SKU-JACKET-BLK-M", "Men", "Jacket", "Black", "tail"),
    ("SKU-CREW-NVY-M", "Men", "Crewneck", "Navy", "core"),
    ("SKU-TEE-BLK-W", "Women", "T-Shirt", "Black", "hero"),
    ("SKU-TANK-WHT-W", "Women", "Tank", "White", "core"),
    ("SKU-HOODIE-PNK-W", "Women", "Hoodie", "Pink", "core"),
    ("SKU-LEGGING-BLK-W", "Women", "Legging", "Black", "tail"),
    ("SKU-JOGGER-GRY-W", "Women", "Joggers", "Grey", "tail"),
    ("SKU-CARDIGAN-BLK-W", "Women", "Cardigan", "Black", "tail"),
    ("SKU-TEE-BLU-K", "Kids", "T-Shirt", "Blue", "core"),
    ("SKU-SET-GRN-K", "Kids", "Set", "Green", "core"),
    ("SKU-HOODIE-RED-K", "Kids", "Hoodie", "Red", "tail"),
    ("SKU-SHORT-NVY-K", "Kids", "Shorts", "Navy", "tail"),
    ("SKU-TANK-WHT-K", "Kids", "Tank", "White", "tail"),
]

CATEGORY_AOV = {
    "T-Shirt": 34, "Polo": 46, "Hoodie": 64, "Tank": 30, "Legging": 52,
    "Set": 38, "Shorts": 36, "Jacket": 98, "Crewneck": 56, "Joggers": 54,
    "Cardigan": 62,
}

TIER_NOISE = {"hero": 0.08, "core": 0.13, "tail": 0.22}
TIER_SCALE = {"hero": 1.6, "core": 1.0, "tail": 0.5}  # budget AND saturation headroom

DAYS = 730  # 2 full years -> two seasonal cycles
START = pd.Timestamp("2024-07-01")

SAT_POWER = 0.6  # how sharply ROAS decays as spend approaches saturation

# ---------------- benchmark-anchored channel specs ----------------
META_SPEC = {
    "cpc_range": (0.35, 0.55),
    "ctr_range": (0.011, 0.020),
    "objective_p": {"Prospecting": 0.5, "Retargeting": 0.3, "DPA": 0.2},
    "objective_roas": {"Prospecting": (2.0, 3.0), "Retargeting": (5.0, 8.0), "DPA": (3.5, 5.0)},
    "s_sat_base": 950,
}
GOOGLE_SPEC = {
    "type_p": {"Search": 0.35, "PMax": 0.30, "Shopping": 0.25, "Demand Gen": 0.10},
    "type_cpc": {"Search": (1.00, 1.80), "PMax": (0.50, 1.00), "Shopping": (0.40, 0.80), "Demand Gen": (0.50, 1.00)},
    "type_roas": {"Search": (3.0, 4.0), "PMax": (3.5, 5.5), "Shopping": (3.5, 5.5), "Demand Gen": (2.0, 3.0)},
    "ctr_range": (0.030, 0.065),
    "s_sat_base": 750,
}
AMAZON_SPEC = {
    "adtype_p": {"Sponsored Products": 0.6, "Sponsored Brands": 0.25, "Sponsored Display": 0.15},
    "adtype_cpc": {"Sponsored Products": (1.80, 2.50), "Sponsored Brands": (2.00, 2.80), "Sponsored Display": (1.20, 1.80)},
    "adtype_roas": {"Sponsored Products": (3.0, 4.0), "Sponsored Brands": (2.5, 3.5), "Sponsored Display": (1.8, 2.8)},
    "ctr_range": (0.0035, 0.0065),
    "s_sat_base": 1150,
}


def weekly_multiplier(dow):
    return {0: 0.90, 1: 0.92, 2: 0.95, 3: 1.00, 4: 1.10, 5: 1.25, 6: 1.15}[dow]

def holiday_multiplier(date):
    m, d = date.month, date.day
    if (m == 11 and d >= 24) or (m == 12 and d <= 2):
        return 2.4
    if m == 12 and 3 <= d <= 23:
        return 1.5
    if m == 1 and d <= 7:
        return 0.6
    if m == 7 and d <= 7:
        return 1.4
    if m == 8 and 15 <= d <= 31:
        return 1.3
    return 1.0

def trend_multiplier(day_index, total_days, annual_growth=0.18):
    return 1.0 + annual_growth * (day_index / 365.0)

def attribution_shock(date, channel):
    """Meta-only iOS14.5/ATT-style reporting distortion: reported revenue
    drops ~22% for an 8-week window each year (recurring, like a real
    platform policy change followed by partial recovery), true demand
    unaffected."""
    if channel != "Meta":
        return 1.0
    for offset in (150, 515):  # once per year of the 2-year window
        shock_start = START + pd.Timedelta(days=offset)
        shock_end = shock_start + pd.Timedelta(weeks=8)
        if shock_start <= date <= shock_end:
            return 0.78
    return 1.0

def saturation_factor(spend, s_sat, power=SAT_POWER):
    return (s_sat / (s_sat + spend)) ** power


def gen_channel(channel, spec, key_fn, type_field, type_dist, type_cpc_or_roas):
    rows = []
    for d in range(DAYS):
        date = START + pd.Timedelta(days=d)
        wk = weekly_multiplier(date.dayofweek)
        hol = holiday_multiplier(date)
        tr = trend_multiplier(d, DAYS)
        for sku, gender, cat, color, tier in SKUS:
            s_sat = spec["s_sat_base"] * TIER_SCALE[tier]
            base_spend = np.random.gamma(4, 60) * TIER_SCALE[tier] * wk * hol * tr
            spend = round(max(base_spend, 5), 2)

            type_val = np.random.choice(list(type_dist.keys()), p=list(type_dist.values()))
            cpc_lo, cpc_hi = (spec["cpc_range"] if "cpc_range" in spec else spec["type_cpc" if "type_cpc" in spec else "adtype_cpc"][type_val])
            roas_lo, roas_hi = (spec["objective_roas"][type_val] if "objective_roas" in spec
                                 else spec["type_roas"][type_val] if "type_roas" in spec
                                 else spec["adtype_roas"][type_val])
            ctr_lo, ctr_hi = spec["ctr_range"]

            cpc = np.random.uniform(cpc_lo, cpc_hi)
            roas_target = np.random.uniform(roas_lo, roas_hi)
            ctr = np.random.uniform(ctr_lo, ctr_hi)

            noise = TIER_NOISE[tier]
            sat = saturation_factor(spend, s_sat)
            roas_effective = roas_target * sat * np.random.normal(1, noise)
            roas_effective = max(roas_effective, 0.1)

            revenue = spend * roas_effective * attribution_shock(date, channel)
            revenue = max(revenue, 0)
            clicks = max(int(round(spend / cpc)), 0)
            impressions = max(int(round(clicks / max(ctr, 1e-4))), clicks)
            aov = CATEGORY_AOV[cat]
            conversions = max(int(round(revenue / aov)), 0)
            new_cust_share = np.random.uniform(0.25, 0.6)

            row = {
                "date": date, "platform_sku_id": key_fn(sku, gender, cat, color),
                "true_sku": sku, "channel": channel,
                "spend": spend, "impressions": impressions, "clicks": clicks,
                "revenue_attributed": round(revenue, 2),
                "new_customer_pct": round(new_cust_share, 2),
                "sku_tier": tier, type_field: type_val,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def inject_realism(df, channel, missing_frac=0.025, late_report_days=3):
    df = df.copy()
    drop_idx = df.sample(frac=missing_frac, random_state=hash(channel) % 1000).index
    df.loc[drop_idx, "revenue_attributed"] = np.nan
    df["is_preliminary"] = df["date"].dt.day <= late_report_days
    prelim_mask = df["is_preliminary"]
    df.loc[prelim_mask, "revenue_attributed"] = (
        df.loc[prelim_mask, "revenue_attributed"] * np.random.uniform(0.55, 0.8, prelim_mask.sum())
    )
    return df


def gen_meta():
    spec = META_SPEC
    df = gen_channel("Meta", spec,
                      key_fn=lambda sku, g, c, col: f"FB_{sku.replace('-', '_')}",
                      type_field="campaign_objective", type_dist=spec["objective_p"],
                      type_cpc_or_roas=None)
    df["purchases_7d_click_1d_view"] = (df["revenue_attributed"] / df["true_sku"].map(
        {s: CATEGORY_AOV[c] for s, g, c, col, t in SKUS})).round().clip(lower=0).astype(int)
    df = inject_realism(df, "Meta", missing_frac=0.03)
    mask = (df["date"].between(START + pd.Timedelta(days=200), START + pd.Timedelta(days=201))) & (df["true_sku"].str.endswith("-K"))
    df = df[~mask]
    return df


def gen_google():
    spec = GOOGLE_SPEC
    df = gen_channel("Google", spec,
                      key_fn=lambda sku, g, c, col: f"goog::{c.lower()}-{col.lower()}-{g.lower()}",
                      type_field="campaign_type", type_dist=spec["type_p"],
                      type_cpc_or_roas=None)
    df["conversions_ddattr"] = (df["revenue_attributed"] / df["true_sku"].map(
        {s: CATEGORY_AOV[c] for s, g, c, col, t in SKUS})).round().clip(lower=0).astype(int)
    df = inject_realism(df, "Google", missing_frac=0.02)
    return df


def gen_amazon():
    spec = AMAZON_SPEC
    rng = np.random.default_rng(7)
    asin_map = {sku: f"B0{rng.integers(100000,999999)}{rng.integers(10,99)}" for sku, *_ in SKUS}
    df = gen_channel("Amazon", spec,
                      key_fn=lambda sku, g, c, col: asin_map[sku],
                      type_field="ad_type", type_dist=spec["adtype_p"],
                      type_cpc_or_roas=None)
    aov_map = {s: CATEGORY_AOV[c] for s, g, c, col, t in SKUS}
    df["orders"] = (df["revenue_attributed"] / df["true_sku"].map(aov_map)).round().clip(lower=0).astype(int)
    df["organic_sales_est"] = (df["revenue_attributed"] * np.random.uniform(2.0, 6.0, len(df))).round(2)  # TACOS context
    df = inject_realism(df, "Amazon", missing_frac=0.025)
    ghost_rows = []
    for d in range(5):
        date = START + pd.Timedelta(days=400 + d)
        spend = round(np.random.uniform(40, 90), 2)
        ghost_rows.append({
            "date": date, "platform_sku_id": "B0GHOST99X", "true_sku": "SKU-SHORT-KHK-M",
            "channel": "Amazon", "spend": spend, "impressions": spend * 200, "clicks": int(spend / 2.0),
            "revenue_attributed": round(spend * 3.1, 2), "new_customer_pct": 0.4, "sku_tier": "tail",
            "ad_type": "Sponsored Products", "orders": 4, "organic_sales_est": round(spend * 3.1 * 3, 2),
            "is_preliminary": False,
        })
    df = pd.concat([df, pd.DataFrame(ghost_rows)], ignore_index=True)
    return df


if __name__ == "__main__":
    meta = gen_meta()
    google = gen_google()
    amazon = gen_amazon()

    meta.to_csv(OUT / "meta_ads_export.csv", index=False)
    google.to_csv(OUT / "google_ads_export.csv", index=False)
    amazon.to_csv(OUT / "amazon_ads_export.csv", index=False)

    print(f"Date range: {START.date()} -> {(START + pd.Timedelta(days=DAYS-1)).date()}  ({DAYS} days)")
    print(f"SKUs: {len(SKUS)}  | Channels: Meta, Google, Amazon")
    print(f"Meta rows: {len(meta):,}   nulls: {meta['revenue_attributed'].isna().sum()}")
    print(f"Google rows: {len(google):,}   nulls: {google['revenue_attributed'].isna().sum()}")
    print(f"Amazon rows: {len(amazon):,}   nulls: {amazon['revenue_attributed'].isna().sum()}")
    print(f"Total raw rows: {len(meta) + len(google) + len(amazon):,}")

    for name, df in [("Meta", meta), ("Google", google), ("Amazon", amazon)]:
        realized_roas = df["revenue_attributed"].sum() / df["spend"].sum()
        print(f"{name} realized blended ROAS: {realized_roas:.2f}x")
    print("Saved to", OUT)
