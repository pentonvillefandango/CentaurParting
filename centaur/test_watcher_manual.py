from __future__ import annotations

import time
from pathlib import Path

from centaur.config import AppConfig, WatchRoot
from centaur.logging import Logger, LoggingConfig
from centaur.watcher import Watcher


def main() -> None:
    # CHANGE THIS to a folder you can safely drop dummy *.fits files into
    watch_path = Path("data/watch_test")
    watch_path.mkdir(parents=True, exist_ok=True)

    cfg = AppConfig(
        watch_roots=[WatchRoot(watch_path.resolve(), "LocalTest")],
        ignore_existing_on_start=True,
        allow_backfill=False,
        stability_window_seconds=2,
        stability_poll_interval_seconds=0.5,
        logging=LoggingConfig(enabled=True, module_verbosity={}),
    )

    logger = Logger(cfg.logging)
    watcher = Watcher(cfg, logger)

    watcher.start()
    print("Drop a file into:", watch_path.resolve())
    print("Example:  echo 'hello' > data/watch_test/test1.fits")
    print("Ctrl+C to stop.\n")

    try:
        while True:
            event = watcher.out_queue.get()
            print("\nEVENT RECEIVED:")
            print("  file_path:", event.file_path)
            print("  root_label:", event.watch_root_label)
            print("  relative :", event.relative_path)
    except KeyboardInterrupt:
        print("\nStopping watcher...")
    finally:
        watcher.stop()


if __name__ == "__main__":
    main()

