"""
ML Sentinel — Background Scheduler
APScheduler jobs for periodic drift detection, scheduled retraining,
and model version checking.
"""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import get_settings

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()


def check_drift_job():
    """Periodic job: run drift detection and trigger retraining if drift found."""
    from drift.detector import drift_detector
    from retraining.pipeline import retraining_pipeline

    logger.info("[Scheduler] Running drift detection check...")
    try:
        drift_detected, details = drift_detector.detect_drift()
        if drift_detected:
            logger.warning(f"[Scheduler] DRIFT DETECTED! Score: {details.get('drift_score', 'N/A')}")
            logger.info("[Scheduler] Triggering drift-based retraining...")
            success = retraining_pipeline.run(trigger_reason="drift_detected")
            if success:
                logger.info("[Scheduler] Drift-triggered retraining completed successfully")
            else:
                logger.warning("[Scheduler] Drift-triggered retraining did not produce a better model")
        else:
            logger.info(f"[Scheduler] No drift. Score: {details.get('drift_score', 'N/A')}")
    except Exception as e:
        logger.error(f"[Scheduler] Drift check failed: {e}")


def scheduled_retraining_job():
    """Periodic job: retrain on a fixed schedule regardless of drift."""
    from retraining.pipeline import retraining_pipeline

    logger.info("[Scheduler] Running scheduled retraining...")
    try:
        success = retraining_pipeline.run(trigger_reason="scheduled")
        if success:
            logger.info("[Scheduler] Scheduled retraining completed successfully")
        else:
            logger.info("[Scheduler] Scheduled retraining — no improvement over current model")
    except Exception as e:
        logger.error(f"[Scheduler] Scheduled retraining failed: {e}")


def check_model_update_job():
    """Periodic job: check for new model versions in MLflow."""
    from models.predictor import predictor

    try:
        updated = predictor.check_for_update()
        if updated:
            logger.info(f"[Scheduler] Model hot-reloaded to v{predictor.model_version}")
    except Exception as e:
        logger.error(f"[Scheduler] Model update check failed: {e}")


def start_scheduler():
    """Start the background scheduler with all jobs."""
    settings = get_settings()

    # Drift detection — every N minutes (default 5)
    scheduler.add_job(
        check_drift_job,
        trigger=IntervalTrigger(minutes=settings.drift_check_interval_minutes),
        id="drift_check",
        name="Drift Detection Check",
        replace_existing=True,
        max_instances=1
    )

    # Scheduled retraining — every 6 hours regardless of drift
    scheduler.add_job(
        scheduled_retraining_job,
        trigger=IntervalTrigger(hours=6),
        id="scheduled_retrain",
        name="Scheduled Retraining",
        replace_existing=True,
        max_instances=1
    )

    # Model version check — every 2 minutes
    scheduler.add_job(
        check_model_update_job,
        trigger=IntervalTrigger(minutes=2),
        id="model_update_check",
        name="Model Version Check",
        replace_existing=True,
        max_instances=1
    )

    scheduler.start()
    logger.info(
        f"[Scheduler] Started — "
        f"drift check every {settings.drift_check_interval_minutes}min, "
        f"scheduled retrain every 6h, "
        f"model check every 2min"
    )


def stop_scheduler():
    """Gracefully stop the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("[Scheduler] Stopped")
