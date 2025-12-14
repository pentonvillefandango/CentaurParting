from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

from centaur.init_db import DEFAULT_DB_PATH
from centaur.logging import LoggingConfig


MetricFailureAction = Literal["continue", "ignore_file"]


@dataclass
class WatchRoot:
    """
    A folder Centaur Parting watches for new FITS files.
    root_label is a friendly name for GUI/logging later.
    """
    root_path: Path
    root_label: str


@dataclass
class AppConfig:
    """
    Central configuration for Centaur Parting (v1).
    In v1, this will be code-defined. Later it can be GUI-driven.
    """

    # Database
    db_path: Path = DEFAULT_DB_PATH

    # Watching behavior
    watch_roots: List[WatchRoot] = field(default_factory=list)
    ignore_existing_on_start: bool = True
    allow_backfill: bool = False

    # File stability check (process only when file stops changing)
    stability_window_seconds: int = 3
    stability_poll_interval_seconds: float = 0.5

    # What to do if a metric module fails
    on_metric_failure: MetricFailureAction = "continue"

    # Logging
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Module enable/disable switches (future-proofing)
    enabled_modules: Dict[str, bool] = field(
        default_factory=lambda: {
            "fits_header_worker": True,
            "sky_basic_worker": True,
            "exposure_advice_worker": True,
            "sky_background2d_worker": True,       
            "flat_basic_worker": True,
            "flat_group_worker": True,
        }


    )

    def is_module_enabled(self, module_name: str) -> bool:
        return self.enabled_modules.get(module_name, False)


def default_config() -> AppConfig:
    """
    Create a sensible default config.
    You will edit watch_roots to match your folders.
    """
    return AppConfig(
        watch_roots=[
            WatchRoot(Path("/Users/admin/Documents/Windowsshared/Astro_Data/Rig24"),"Rig24")
            # Example (edit this)
            # WatchRoot(Path("/Volumes/NAS/rig1/captures"), "Rig1 NAS"),
        ],
        logging=LoggingConfig(
            enabled=True,
            module_verbosity={
                "fits_header_worker": False, 
                "sky_basic_worker": False,# Example: "fits_header_worker": True
                "exposure_advice_worker": False,
                "sky_background2d_worker": False,
            },
        ),
    )

