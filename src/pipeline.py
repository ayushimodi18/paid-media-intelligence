"""
Production pipeline orchestrator.

Runs the full path: generate -> validate (raw) -> reconcile/unify ->
validate (unified) -> feature engineering -> EDA -> model CV + selection ->
budget utilization -> allocation optimization.

Production-pipeline conventions applied here:
  - Structured logging (not print) with timestamps and stage names, so a
    failure is traceable to exactly which stage and why.
  - A validation GATE: if raw or unified data fails hard validation
    (pandera schema errors, referential integrity), the pipeline stops
    before wasting compute on modeling against bad data. Warnings don't
    block; errors do.
  - Each stage is wrapped so a failure in one stage produces a clear error
    message naming the stage, rather than an unhandled traceback from deep
    inside pandas/sklearn.
  - A machine-readable run manifest (JSON) is written at the end, recording
    what ran, how long each stage took, and pass/fail per validation gate
    -- the kind of artifact a real orchestrator (Airflow/Prefect/cron) would
    want to log.

Run with: python src/pipeline.py [--skip-generate] [--cv-folds N]
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("paid_media_pipeline")

DATA = Path(__file__).resolve().parents[1] / "data"


class PipelineStageError(Exception):
    """Raised when a stage fails in a way that should stop the pipeline."""


class PipelineRun:
    def __init__(self):
        self.manifest = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stages": [],
        }

    def stage(self, name):
        return _StageTimer(self, name)

    def finalize(self, status="success"):
        self.manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.manifest["status"] = status
        with open(DATA / "pipeline_run_manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)
        log.info(f"Run manifest written to {DATA / 'pipeline_run_manifest.json'}")


class _StageTimer:
    def __init__(self, run: PipelineRun, name: str):
        self.run = run
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        log.info(f"STAGE START: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = round(time.time() - self.t0, 2)
        if exc_type is not None:
            log.error(f"STAGE FAILED: {self.name} after {elapsed}s -- {exc_val}")
            self.run.manifest["stages"].append({"name": self.name, "status": "failed",
                                                 "seconds": elapsed, "error": str(exc_val)})
            return False  # re-raise
        log.info(f"STAGE OK: {self.name} ({elapsed}s)")
        self.run.manifest["stages"].append({"name": self.name, "status": "ok", "seconds": elapsed})
        return False


def main():
    parser = argparse.ArgumentParser(description="Paid media intelligence pipeline")
    parser.add_argument("--skip-generate", action="store_true",
                         help="Skip synthetic data generation, reuse existing CSVs in data/")
    parser.add_argument("--cv-folds", type=int, default=4, help="Number of rolling-window CV folds")
    parser.add_argument("--fail-on-warning", action="store_true",
                         help="Treat validation warnings as hard failures (strict mode)")
    args = parser.parse_args()

    run = PipelineRun()
    log.info("=" * 70)
    log.info("Paid Media Intelligence Pipeline -- starting run")
    log.info("=" * 70)

    try:
        if not args.skip_generate:
            with run.stage("01_generate_synthetic_data"):
                import generate_data
                generate_data.gen_meta().to_csv(DATA / "meta_ads_export.csv", index=False)
                generate_data.gen_google().to_csv(DATA / "google_ads_export.csv", index=False)
                generate_data.gen_amazon().to_csv(DATA / "amazon_ads_export.csv", index=False)
        else:
            log.info("Skipping generation, using existing files in data/")

        with run.stage("02_validate_raw_exports"):
            import pandas as pd
            import validation
            meta = pd.read_csv(DATA / "meta_ads_export.csv", parse_dates=["date"])
            google = pd.read_csv(DATA / "google_ads_export.csv", parse_dates=["date"])
            amazon = pd.read_csv(DATA / "amazon_ads_export.csv", parse_dates=["date"])
            reports = [validation.validate_raw_export(meta, "meta_raw"),
                       validation.validate_raw_export(google, "google_raw"),
                       validation.validate_raw_export(amazon, "amazon_raw")]
            for r in reports:
                for w in r.warnings:
                    log.warning(f"[{r.source}] {w}")
                if not r.passed:
                    for e in r.errors:
                        log.error(f"[{r.source}] {e}")
            hard_fail = any(not r.passed for r in reports)
            soft_fail = args.fail_on_warning and any(r.warnings for r in reports)
            if hard_fail or soft_fail:
                raise PipelineStageError("Raw export validation gate failed -- see errors above")

        with run.stage("03_ingest_and_reconcile"):
            import ingest_reconcile
            report, unified = ingest_reconcile.unify(save=True)
            log.info(f"Reconciliation: Meta {report['meta_reconciled']}/{report['meta_total_rows']}, "
                      f"Google {report['google_reconciled']}/{report['google_total_rows']}, "
                      f"Amazon {report['amazon_reconciled']}/{report['amazon_total_rows']}")

        with run.stage("04_validate_unified"):
            import pandas as pd
            unified_df = pd.read_csv(DATA / "unified_performance.csv", parse_dates=["date"])
            unified_report = validation.validate_unified(unified_df)
            for w in unified_report.warnings:
                log.warning(w)
            if not unified_report.passed:
                for e in unified_report.errors:
                    log.error(e)
                raise PipelineStageError("Unified data validation gate failed")

        with run.stage("05_feature_engineering"):
            import feature_engineering
            feature_table = feature_engineering.build_feature_table()
            log.info(f"Feature table: {feature_table.shape[0]} rows x {feature_table.shape[1]} cols")

        with run.stage("06_eda"):
            import eda
            eda.run_eda()

        with run.stage("07_model_cv_and_selection"):
            import forecast_engine
            cv_summary, best_models = forecast_engine.compare_models_cv(n_folds=args.cv_folds)
            for ch, m in best_models.items():
                log.info(f"  Best model for {ch}: {m}")

        with run.stage("08_budget_utilization"):
            import budget_utilization as bu
            flagged = bu.flag_waste_and_underspend(unified_df)
            summary = bu.summarize(flagged)
            summary.to_csv(DATA / "budget_utilization_summary.csv", index=False)
            waste_rate = flagged["flag_budget_waste"].mean()
            underspend_rate = flagged["flag_underspend_opportunity"].mean()
            log.info(f"  Waste flagged on {waste_rate:.1%} of SKU-channel-days; "
                      f"underspend opportunity on {underspend_rate:.1%}")

        with run.stage("09_budget_allocation_optimization"):
            import optimize_allocation
            curves, _ = optimize_allocation.fit_efficiency_curves()
            rec_df, summary = optimize_allocation.recommend_allocation(curves)
            rec_df.to_csv(DATA / "allocation_recommendation.csv", index=False)
            log.info(f"  Current blended ROAS {summary['current_blended_roas']}x -> "
                      f"projected {summary['projected_blended_roas']}x")

        with run.stage("10_nc_cpa_secondary_metric"):
            import nc_cpa
            nc_result = nc_cpa.run_all_channels()
            nc_result.to_csv(DATA / "nc_cpa_summary.csv", index=False)
            blended = nc_result[nc_result["cut"] == "channel_blended"]
            fails = blended[~blended["within_target"]]
            if len(fails):
                log.warning(f"  NC-CPA exceeds ${nc_cpa.NC_CPA_TARGET} target on: {fails['channel'].tolist()}")
            else:
                log.info(f"  All channels within ${nc_cpa.NC_CPA_TARGET} NC-CPA target")

        with run.stage("11_business_impact_dio_fulfillment"):
            import business_impact
            flags_for_impact = business_impact.load_flags()
            dio = business_impact.estimate_dio_impact(flags_for_impact)
            fulfillment = business_impact.estimate_fulfillment_impact(flags_for_impact)
            log.info(f"  Hero-SKU unused efficient ad capacity: "
                      f"${dio['hero_unused_efficient_capacity_dollars']:,.0f} "
                      f"({dio['headroom_as_pct_of_hero_spend']}% of hero spend)")
            log.info(f"  Flagged budget waste: ${fulfillment['total_flagged_waste_dollars']:,.0f} "
                      f"({fulfillment['waste_as_pct_of_total_spend']}% of total spend)")

        run.finalize(status="success")
        log.info("=" * 70)
        log.info("PIPELINE COMPLETE -- all stages passed")
        log.info("=" * 70)

    except PipelineStageError as e:
        run.finalize(status="failed")
        log.error(f"PIPELINE HALTED: {e}")
        sys.exit(1)
    except Exception as e:
        run.finalize(status="error")
        log.exception(f"PIPELINE CRASHED with unhandled exception: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
