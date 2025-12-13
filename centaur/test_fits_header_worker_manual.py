from __future__ import annotations

from pathlib import Path

from centaur.config import AppConfig, WatchRoot
from centaur.logging import Logger, LoggingConfig
from centaur.watcher import FileReadyEvent
from centaur.fits_header_worker import process_file_event, MODULE_NAME


def main() -> None:
    fits_path = Path("/Users/admin/Documents/Windowsshared/Astro_Data/Rig24/test.fits").resolve()

    cfg = AppConfig(
        watch_roots=[WatchRoot(fits_path.parent, "LocalTest")],
        logging=LoggingConfig(
            enabled=True,
            module_verbosity={MODULE_NAME: True},  # set False to suppress verbose fields
        ),
    )

    logger = Logger(cfg.logging)

    event = FileReadyEvent(
        file_path=fits_path,
        watch_root_label="LocalTest",
        watch_root_path=fits_path.parent,
        relative_path=fits_path.name,
    )

    process_file_event(cfg, logger, event)


if __name__ == "__main__":
    main()
