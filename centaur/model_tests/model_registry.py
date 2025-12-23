from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from centaur.model_tests.common import TestResult


@dataclass(frozen=True)
class ModelTest:
    name: str
    test_func: Callable[..., TestResult]  # (db_path, out_path) -> TestResult


def get_registry() -> List[ModelTest]:
    from centaur.model_tests.models.generic_table import make_generic_test

    # All “model tables” we should validate
    tables = [
        "sky_basic_metrics",
        "sky_background2d_metrics",
        "exposure_advice",
        "psf_detect_metrics",
        "psf_basic_metrics",
        "psf_grid_metrics",
        "psf_model_metrics",
        "saturation_metrics",
        "signal_structure_metrics",
        "roi_signal_metrics",
        "nebula_mask_metrics",
        "masked_signal_metrics",
        "star_headroom_metrics",
        "training_sessions",
        "training_session_frames",
        "dark_library_profiles",
        "dark_library_exposures",
        "observing_conditions",
        "training_derived_metrics",
        "training_session_results",
    ]

    return [ModelTest(t, make_generic_test(t)) for t in tables]
