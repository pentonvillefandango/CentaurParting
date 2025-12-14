from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, List

from centaur.config import AppConfig
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent

WorkerFn = Callable[[AppConfig, Logger, FileReadyEvent], Optional[bool]]


@dataclass
class PipelineResult:
    enabled: int = 0
    skipped: int = 0
    ok: int = 0
    failed: int = 0
    failed_items: List[Tuple[str, str]] = field(default_factory=list)  # (module_name, file_path)


def build_worker_registry() -> Dict[str, WorkerFn]:
    """
    Central registry of pipeline workers.
    Order matters.
    """
    from centaur.fits_header_worker import process_file_event as fits_header_process
    from centaur.sky_basic_worker import process_file_event as sky_basic_process
    from centaur.sky_background2d_worker import process_file_event as sky_bkg2d_process
    from centaur.exposure_advice_worker import process_file_event as exposure_advice_process
    from centaur.flat_basic_worker import process_file_event as flat_basic_process
    from centaur.flat_group_worker import process_file_event as flat_group_process
    from centaur.psf_detect_worker import process_file_event as psf_detect_process
    from centaur.psf_basic_worker import process_file_event as psf_basic_process




    return {
        # Must run first so image_id exists and header tables are populated.
        "fits_header_worker": fits_header_process,

        # Flats
        "flat_group_worker": flat_group_process,
        "flat_basic_worker": flat_basic_process,
        

        # Sky metrics (LIGHT frames)
        "sky_basic_worker": sky_basic_process,
        "sky_background2d_worker": sky_bkg2d_process,

        # Decision layer (Module0) (LIGHT frames)
        "exposure_advice_worker": exposure_advice_process,

        # PSF layer
        "psf_detect_worker": psf_detect_process,
        "psf_basic_worker": psf_basic_process,
    }


def _run_one(
    cfg: AppConfig,
    logger: Logger,
    event: FileReadyEvent,
    registry: Dict[str, WorkerFn],
    result: PipelineResult,
    module_name: str,
) -> None:
    """
    Run a single worker and update totals.
    """
    worker_fn = registry[module_name]

    if not cfg.is_module_enabled(module_name):
        result.skipped += 1
        return

    result.enabled += 1
    worker_result = worker_fn(cfg, logger, event)

    if worker_result is False:
        result.failed += 1
        result.failed_items.append((module_name, str(event.file_path)))
    else:
        result.ok += 1


def run_pipeline_for_event(
    cfg: AppConfig,
    logger: Logger,
    event: FileReadyEvent,
    registry: Dict[str, WorkerFn],
) -> PipelineResult:
    """
    Route events through the correct pipeline.

    Rule (v1):
    - Always run fits_header_worker first.
    - If IMAGETYP == FLAT: run flat_group_worker + flat_basic_worker.
    - If IMAGETYP == LIGHT (or anything else): run sky_basic_worker, sky_background2d_worker, exposure_advice_worker.
    """
    result = PipelineResult()

    # 1) Always run FITS header first
    _run_one(cfg, logger, event, registry, result, "fits_header_worker")

    # If header ingest failed, stop here.
    if result.failed > 0:
        return result

    # 2) Determine frame type by reading FITS header directly (no DB dependency)
    imagetyp = ""
    try:
        from astropy.io import fits
        # Use memmap=False to avoid BZERO/BSCALE issues across platforms
        with fits.open(str(event.file_path), memmap=False) as hdul:
            imagetyp = str(hdul[0].header.get("IMAGETYP", "")).strip().upper()
    except Exception:
        imagetyp = ""

    # 3) Route
    if imagetyp == "FLAT":
        _run_one(cfg, logger, event, registry, result, "flat_group_worker")
        _run_one(cfg, logger, event, registry, result, "flat_basic_worker")

        # Light-only modules not applicable for flats
        for name in ("sky_basic_worker", "sky_background2d_worker", "exposure_advice_worker", "psf_detect_worker", "psf_basic_worker"):
            if cfg.is_module_enabled(name):
                result.skipped += 1

        return result

    # Non-FLAT frames (LIGHT, DARK, BIAS, unknown): run sky + advice
    # Flat modules not applicable here
    for name in ("flat_group_worker", "flat_basic_worker"):
        if cfg.is_module_enabled(name):
            result.skipped += 1

    for name in ("sky_basic_worker", "sky_background2d_worker", "exposure_advice_worker", "psf_detect_worker", "psf_basic_worker"):
        _run_one(cfg, logger, event, registry, result, name)

    return result
