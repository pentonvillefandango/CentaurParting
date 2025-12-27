from __future__ import annotations

import argparse
import textwrap
from queue import Empty
from typing import List, Tuple

from centaur.config import default_config
from centaur.init_db import init_db
from centaur.logging import Logger
from centaur.watcher import Watcher
from centaur.pipeline import build_worker_registry, run_pipeline_for_event
from centaur.database import Database, DbConfig

from centaur.schema_sky_basic import ensure_sky_basic_schema
from centaur.schema_sky_background2d import ensure_sky_background2d_schema
from centaur.schema_camera_constants import ensure_camera_constants_schema
from centaur.schema_exposure_advice import ensure_exposure_advice_schema
from centaur.seed_camera_constants import seed_camera_constants
from centaur.schema_flat_profiles import ensure_flat_profiles_schema
from centaur.schema_flat_capture_sets import ensure_flat_capture_sets_schema
from centaur.schema_flat_metrics import ensure_flat_metrics_schema
from centaur.schema_flat_links import ensure_flat_links_schema
from centaur.schema_psf_detect import ensure_psf_detect_schema
from centaur.schema_psf_basic import ensure_psf_basic_schema
from centaur.schema_psf_model import ensure_psf_model_schema
from centaur.schema_psf_grid import ensure_psf_grid_schema
from centaur.schema_saturation import ensure_saturation_schema
from centaur.schema_roi_signal import ensure_roi_signal_schema
from centaur.ensure_migrations import ensure_migrations
from centaur.schema_signal_structure import ensure_signal_structure_schema
from centaur.schema_nebula_mask import ensure_nebula_mask_schema
from centaur.schema_masked_signal import ensure_masked_signal_schema
from centaur.schema_star_headroom import ensure_star_headroom_schema
from centaur.schema_frame_quality import ensure_frame_quality_schema

# Training / advice data layer schemas
from centaur.schema_training_sessions import ensure_training_sessions_schema
from centaur.schema_training_session_frames import ensure_training_session_frames_schema
from centaur.schema_training_session_results import (
    ensure_training_session_results_schema,
)
from centaur.schema_dark_library_profiles import ensure_dark_library_profiles_schema
from centaur.schema_dark_library_exposures import ensure_dark_library_exposures_schema
from centaur.schema_observing_conditions import ensure_observing_conditions_schema
from centaur.schema_training_derived_metrics import (
    ensure_training_derived_metrics_schema,
)

# Preflight to avoid surprise downloads mid-run
from centaur.iers_preflight import ensure_iers_ready

# In-process training session monitor
from centaur.training_session_monitor import start_monitor_thread


HELP_TEXT = """
CentaurParting (cp_start)
========================
Starts the realtime FITS watcher + modular metrics pipeline.

Normal dev loop
---------------
  cp_db_reset
  cp_start
  (drop FITS files into your watcher root)

Training session “easy mode”
----------------------------
Enable the monitor inside cp_start (recommended):
  cp_start --training-monitor

Then start a session (interactive):
  python3 -m centaur.tools.training_session_start --db data/centaurparting.db

The monitor will:
  - auto-tag matching LIGHT frames into training_session_frames
  - log progress 1/9, 2/9, ...
  - when finish_after is reached, auto-solve + write training_session_results
  - auto-close the session

If there are NO open sessions, the monitor is essentially idle (cheap query + sleep).

Quick: list setup_id
--------------------
  cp_sql
  SELECT setup_id, telescop, instrume, detector, site_name FROM optical_setups;
""".strip()


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument(
        "-h2",
        "--help2",
        action="store_true",
        help="Show extended help (training sessions, workflows, commands) and exit.",
    )

    ap.add_argument(
        "--training-monitor",
        action="store_true",
        help="Run the training session monitor inside cp_start (auto-tag + auto-close).",
    )
    ap.add_argument(
        "--training-monitor-poll",
        type=float,
        default=5.0,
        help="Training monitor poll interval in seconds (default: 5.0).",
    )

    args = ap.parse_args()

    if args.help2:
        print(textwrap.dedent(HELP_TEXT))
        return

    cfg = default_config()
    logger = Logger(cfg.logging)

    # Base schema (safe to run repeatedly)
    init_db(cfg.db_path)
    ensure_sky_basic_schema(cfg.db_path)
    ensure_sky_background2d_schema(cfg.db_path)
    ensure_camera_constants_schema(cfg.db_path)
    ensure_exposure_advice_schema(cfg.db_path)
    seed_camera_constants()
    ensure_flat_profiles_schema(cfg.db_path)
    ensure_flat_capture_sets_schema(cfg.db_path)
    ensure_flat_metrics_schema(cfg.db_path)
    ensure_flat_links_schema(cfg.db_path)
    ensure_psf_detect_schema(cfg.db_path)
    ensure_psf_basic_schema(cfg.db_path)
    ensure_psf_model_schema(cfg.db_path)
    ensure_psf_grid_schema(cfg.db_path)
    ensure_nebula_mask_schema(cfg.db_path)
    ensure_saturation_schema(cfg.db_path)
    ensure_roi_signal_schema(cfg.db_path)
    ensure_signal_structure_schema(cfg.db_path)
    ensure_masked_signal_schema(cfg.db_path)
    ensure_star_headroom_schema(cfg.db_path)
    ensure_migrations(cfg.db_path)
    ensure_frame_quality_schema(cfg.db_path)

    # --------------------------------------------------
    # Training / advice data layer schemas
    # --------------------------------------------------
    ensure_training_sessions_schema(cfg.db_path)
    ensure_training_session_frames_schema(cfg.db_path)
    ensure_training_session_results_schema(cfg.db_path)

    ensure_dark_library_profiles_schema(cfg.db_path)
    ensure_dark_library_exposures_schema(cfg.db_path)

    ensure_observing_conditions_schema(cfg.db_path)
    ensure_training_derived_metrics_schema(cfg.db_path)

    # Preflight IERS data once at startup
    ensure_iers_ready(logger)

    # Optional: in-process training monitor thread
    monitor = None
    if args.training_monitor:
        _, monitor = start_monitor_thread(
            db_path=str(cfg.db_path),
            logger=logger,
            poll_seconds=float(args.training_monitor_poll),
            only_usable=True,
        )

    watcher = Watcher(cfg, logger)
    watcher.start()

    registry = build_worker_registry()

    # IMPORTANT: pipeline-owned module_runs writer must point at cfg.db_path
    db = Database(DbConfig(db_path=cfg.db_path))
    db.connect()

    ready_total = 0
    modules_enabled_total = 0
    modules_skipped_total = 0
    modules_ok_total = 0
    modules_failed_total = 0

    failed_items_all: List[Tuple[str, str]] = []  # (module_name, file_path)

    print("Centaur Parting running. Ctrl+C to stop.\n")

    try:
        while True:
            try:
                event = watcher.out_queue.get(timeout=0.5)
            except Empty:
                continue

            ready_total += 1

            per_event = run_pipeline_for_event(cfg, logger, event, registry, db=db)
            modules_enabled_total += per_event.enabled
            modules_skipped_total += per_event.skipped
            modules_ok_total += per_event.ok
            modules_failed_total += per_event.failed

            if per_event.failed_items:
                failed_items_all.extend(per_event.failed_items)

            if ready_total % 10 == 0:
                print(
                    f"\nTOTALS: "
                    f"ready={ready_total} "
                    f"modules_enabled={modules_enabled_total} "
                    f"ok={modules_ok_total} "
                    f"failed={modules_failed_total} "
                    f"skipped={modules_skipped_total}\n"
                )

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        if monitor is not None:
            monitor.stop()

        watcher.stop()
        db.close()

        print(
            f"\nFINAL TOTALS: "
            f"ready={ready_total} "
            f"modules_enabled={modules_enabled_total} "
            f"ok={modules_ok_total} "
            f"failed={modules_failed_total} "
            f"skipped={modules_skipped_total}\n"
        )

        if failed_items_all:
            print(f"FAILED MODULES ({len(failed_items_all)}):")
            for module_name, file_path in failed_items_all[-20:]:
                print(f"  {module_name} :: {file_path}")
            if len(failed_items_all) > 20:
                print(f"  ... (showing last 20 of {len(failed_items_all)})")
            print("")


if __name__ == "__main__":
    main()
