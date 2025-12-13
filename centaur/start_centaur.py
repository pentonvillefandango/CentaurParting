from __future__ import annotations

from queue import Empty

from centaur.config import default_config
from centaur.fits_header_worker import process_file_event
from centaur.init_db import init_db
from centaur.logging import Logger
from centaur.watcher import Watcher


def main() -> None:
    cfg = default_config()
    logger = Logger(cfg.logging)

    init_db(cfg.db_path)

    watcher = Watcher(cfg, logger)
    watcher.start()

    seen_ready = 0
    processed_ok = 0
    processed_failed = 0
    skipped = 0

    print("Centaur Parting running. Ctrl+C to stop.\n")

    try:
        while True:
            try:
                event = watcher.out_queue.get(timeout=0.5)
            except Empty:
                continue

            seen_ready += 1

            if cfg.is_module_enabled("fits_header_worker"):
                result = process_file_event(cfg, logger, event)

                # IMPORTANT:
                # - Older workers return None (meaning: no explicit success/fail signal)
                # - We treat None as success because the worker already logs OK/FAILED
                if result is False:
                    processed_failed += 1
                else:
                    processed_ok += 1
            else:
                skipped += 1

            if seen_ready % 10 == 0:
                print(
                    f"\nTOTALS: "
                    f"ready={seen_ready} "
                    f"ok={processed_ok} "
                    f"failed={processed_failed} "
                    f"skipped={skipped}\n"
                )

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        watcher.stop()
        print(
            f"\nFINAL TOTALS: "
            f"ready={seen_ready} "
            f"ok={processed_ok} "
            f"failed={processed_failed} "
            f"skipped={skipped}\n"
        )


if __name__ == "__main__":
    main()
