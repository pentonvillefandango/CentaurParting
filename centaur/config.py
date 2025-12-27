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
            "frame_quality_worker": True,
            # Training worker
            "observing_conditions_worker": True,
            "training_derived_worker": True,
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

    # Frame quality (v1 fixed-threshold scoring)
    quality_reject_saturated_pixel_fraction: float = 0.0005
    quality_warn_saturated_pixel_fraction: float = 0.0001

    quality_reject_min_stars_measured: int = 30
    quality_warn_min_stars_measured: int = 80

    quality_reject_ecc_median: float = 0.70
    quality_warn_ecc_median: float = 0.55

    quality_reject_fwhm_px_median: float = 12.0
    quality_warn_fwhm_px_median: float = 8.0

    quality_warn_plane_slope_mag_adu_per_tile: float = 50.0
    quality_warn_grad_p95_adu_per_tile: float = 80.0
    quality_warn_corner_delta_adu: float = 500.0
    quality_warn_bkg2d_rms_of_map_adu: float = 200.0

    quality_warn_if_score_below: int = 70
    quality_cap_score_on_warn: int = 70
    quality_cap_score_on_reject: int = 30
    # ---------------------------------------------------------------------
    # Frame quality v1.1 (per-image only; no rolling/session context yet)
    # Expands scoring to include: signal, headroom, psf robustness.
    # ---------------------------------------------------------------------

    # --- Signal (per-image) ---
    # Uses roi_signal_metrics.obj_minus_bg_adu_s when available.
    # If missing/unusable, signal scoring should degrade confidence rather than hard fail.
    quality_warn_min_obj_minus_bg_adu_s: float = 0.0

    # Uses masked_signal_metrics.nebula_minus_bg_adu_s when available (and usable).
    # (Optional enhancement: only consider if nebula mask usable, otherwise ignore.)
    quality_warn_min_nebula_minus_bg_adu_s: float = 0.0

    # --- PSF robustness (per-image) ---
    # Use psf_basic_metrics.ecc_p90 to catch “some stars are ugly” frames.
    quality_reject_ecc_p90: float = 0.85
    quality_warn_ecc_p90: float = 0.70

    # Use psf_basic_metrics.fwhm_px_p90 as a “worst-case seeing/focus” proxy.
    quality_warn_fwhm_px_p90: float = 10.0

    # Spread = p90 - p10: catches instability / variable HFR across the frame.
    quality_warn_fwhm_spread_px: float = 4.0

    # --- Star headroom (per-image) ---
    # Uses star_headroom_metrics.headroom_p99.
    # headroom is typically in [0,1] where lower = closer to saturation.
    quality_warn_headroom_p99_below: float = 0.10
    # --------------------------------------------------
    # Training session recommendation defaults
    # These act as defaults shown in training_session_start prompts.
    # Session-level overrides are stored in training_sessions.params_json.
    # --------------------------------------------------

    # Default required filters for evaluation (order matters)
    training_required_filters_default: list[str] = field(
        default_factory=lambda: ["SII", "HA", "OIII"]
    )

    # Minimum linear headroom (p99) to consider an exposure "safe".
    # 0.10 = keep at least 10% headroom.
    training_min_linear_headroom_p99: float = 0.10

    # Scoring weights: favor overall quality (mean) vs weakest filter (min)
    training_score_w_mean: float = 0.55
    training_score_w_min: float = 0.45

    # Exposure time penalties: discourage extremes within the candidate set
    training_short_penalty: float = 0.10
    training_long_penalty: float = 0.08

    # Whether to include excluded candidates + reasons in stats_json
    training_include_excluded: bool = True

    # Human-friendly explanation threshold:
    # if best_score - candidate_score < this, we’ll report “close call”
    training_close_call_delta: float = 0.05
    # -----------------------------
    # Training session recommendation defaults
    # -----------------------------
    training_required_filters_default: List[str] = field(
        default_factory=lambda: ["SII", "HA", "OIII"]
    )
    training_min_linear_headroom_p99: float = 0.10

    training_score_w_mean: float = 0.55
    training_score_w_min: float = 0.45
    training_short_penalty: float = 0.10
    training_long_penalty: float = 0.08
    training_include_excluded: bool = True
    training_close_call_delta: float = 0.05

    # -----------------------------
    # Ratio recommendation strategy
    # -----------------------------
    # Exposure selection is always based on best score.
    # This controls how the *ratio* is computed.
    # Options: "best", "second", "aggregate_weighted", "aggregate_equal"
    training_ratio_strategy: str = "aggregate_weighted"

    # Weighting used by aggregate modes:
    # "score_weighted" = shifted score weights (stable even if scores negative)
    # "equal"         = every eligible exposure contributes equally
    training_ratio_weight_mode: str = "score_weighted"
    # -----------------------------
    # Training session scoring + ratio behavior (global defaults)
    # -----------------------------

    # (1) Sky-limited soft penalty:
    # If avg sky_limited_ratio < target, apply a small score penalty.
    training_sky_limited_target_ratio: float = 10.0
    training_sky_limited_penalty_weight: float = 0.08

    # (3) Transparency variation warning / ratio stabilisation:
    # If transparency spread is high, ratios can jump around when you only have 1 frame/filter/exposure.
    training_transparency_variation_warn_frac: float = 0.35
    training_prefer_aggregate_ratio_when_unstable: bool = True

    # (2) Aggregate ratio weighting:
    # "nebula_over_sky" stabilises ratios using a brightness proxy; "uniform" averages exposures equally.
    training_aggregate_ratio_weight_mode: str = "nebula_over_sky"

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
                "star_headroom_worker": False,
                "frame_quality_worker": False,
                "observing_conditions_worker": False,
                "training_derived_worker": False,
                "training_session_monitor": True,
            },
        ),
    )
