from __future__ import annotations

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


def main() -> None:
    cfg = default_config()
    logger = Logger(cfg.logging)

    # Base schema (safe to run repeatedly)
    init_db(cfg.db_path)

    # Schema add-ons (safe to run repeatedly)
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

    # NEW
    ensure_saturation_schema(cfg.db_path)
    ensure_roi_signal_schema(cfg.db_path)
    ensure_signal_structure_schema(cfg.db_path)
    ensure_masked_signal_schema(cfg.db_path)
    ensure_star_headroom_schema(cfg.db_path)

    ensure_migrations(cfg.db_path)

    watcher = Watcher(cfg, logger)
    watcher.start()

    registry = build_worker_registry()

    # IMPORTANT: pipeline-owned module_runs writer must point at cfg.db_path (not DEFAULT_DB_PATH)
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
