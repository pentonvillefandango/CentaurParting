from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal

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
    Central configuration for CentaurParting (v1).
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
            # Sky
            "sky_basic_worker": True,
            "sky_background2d_worker": True,
            # Structure
            "signal_structure_worker": True,
            # New: saturation + ROI signal
            "saturation_worker": True,
            "roi_signal_worker": True,
            # Advice
            "exposure_advice_worker": True,
            # Flats
            "flat_basic_worker": True,
            "flat_group_worker": True,
            # PSF
            "psf_detect_worker": True,
            "psf_basic_worker": True,
            "psf_grid_worker": True,
            "psf_model_worker": True,
            # Nebula / masking chain
            "nebula_mask_worker": True,
            "masked_signal_worker": True,
            "star_headroom_worker": True,
        }
    )

    # ROI signal worker params
    roi_signal_obj_fraction: float = 0.20
    roi_signal_bg_outer_fraction: float = 0.45

    # PSF params
    psf_use_roi: bool = False
    psf_roi_fraction: float = 0.5
    psf_threshold_sigma: float = 8.0
    psf_min_separation_px: int = 8
    psf_max_stars: int = 0
    psf_edge_margin_px: int = 16
    psf_debug_dump_candidates_csv: bool = False
    psf_good_extra_sigma: float = 8.0
    psf_cutout_radius_px: int = 8
    psf1_max_stars_measured: int = 0  # 0 = unlimited
    psf1_debug_dump_measurements_csv: bool = False
    psf2_enabled: bool = True
    psf2_max_stars: int = 0
    psf2_fit_radius_px: int = 8
    psf2_models: list[str] = field(default_factory=lambda: ["gaussian", "moffat"])
    psf2_min_good_fits = 50

    # Nebula mask params
    nebula_mask_threshold_sigma: float = 3.0
    nebula_mask_smooth_sigma_px: float = 2.0
    nebula_mask_bg_clip_sigma: float = 3.0
    nebula_mask_bg_clip_maxiters: int = 5

    # NEW: nebula mask component-labeling requirements/options
    # We are making SciPy a requirement, so the worker should fail if SciPy is missing.
    nebula_mask_require_scipy: bool = True

    # Optional performance knob: downsample before connected-components labeling.
    # 1 = off (default), 2 = half-res, 4 = quarter-res, etc.
    # (leave this at 1 unless you explicitly want it.)
    nebula_mask_components_downsample: int = 1

    # Star headroom params
    star_headroom_max_stars: int = 0
    star_headroom_sample_radius_px: int = 3
    star_headroom_peak_percentile: float = 99

    def is_module_enabled(self, module_name: str) -> bool:
        return self.enabled_modules.get(module_name, False)


def default_config() -> AppConfig:
    """
    Create a sensible default config.
    You will edit watch_roots to match your folders.
    """
    return AppConfig(
        watch_roots=[
            WatchRoot(
                Path("/Users/admin/Documents/Windowsshared/Astro_Data/Rig24"), "Rig24"
            )
            # Example (edit this)
            # WatchRoot(Path("/Volumes/NAS/rig1/captures"), "Rig1 NAS"),
        ],
        logging=LoggingConfig(
            enabled=True,
            module_verbosity={
                "fits_header_worker": False,
                "sky_basic_worker": False,
                "sky_background2d_worker": False,
                "saturation_worker": False,
                "roi_signal_worker": False,
                "exposure_advice_worker": False,
                "psf_detect_worker": False,
                "psf_basic_worker": False,
                "psf_model_worker": False,
                "psf_grid_worker": False,
                "signal_structure_worker": False,
                "nebula_mask_worker": False,
                "masked_signal_worker": False,
                "star_headroom_worker": True,
            },
        ),
    )
