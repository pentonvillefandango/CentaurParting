from __future__ import annotations

from queue import Empty

from centaur.config import default_config
from centaur.init_db import init_db
from centaur.logging import Logger
from centaur.watcher import Watcher
from centaur.pipeline import build_worker_registry, run_pipeline_for_event
from centaur.schema_sky_basic import ensure_sky_basic_schema


def main() -> None:
    cfg = default_config()
    logger = Logger(cfg.logging)

    # Base schema (safe to run repeatedly)
    init_db(cfg.db_path)

    # Sky Basic schema add-on (safe to run repeatedly)
    ensure_sky_basic_schema(cfg.db_path)

    watcher = Watcher(cfg, logger)
    watcher.start()

    registry = build_worker_registry()

    ready_total = 0
    modules_enabled_total = 0
    modules_skipped_total = 0
    modules_ok_total = 0
    modules_failed_total = 0

    print("Centaur Parting running. Ctrl+C to stop.\n")

    try:
        while True:
            try:
                event = watcher.out_queue.get(timeout=0.5)
            except Empty:
                continue

            ready_total += 1

            per_event = run_pipeline_for_event(cfg, logger, event, registry)
            modules_enabled_total += per_event.enabled
            modules_skipped_total += per_event.skipped
            modules_ok_total += per_event.ok
            modules_failed_total += per_event.failed

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
        print(
            f"\nFINAL TOTALS: "
            f"ready={ready_total} "
            f"modules_enabled={modules_enabled_total} "
            f"ok={modules_ok_total} "
            f"failed={modules_failed_total} "
            f"skipped={modules_skipped_total}\n"
        )


if __name__ == "__main__":
    main()
