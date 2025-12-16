from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, List, Union

from centaur.config import AppConfig
from centaur.logging import Logger, utc_now
from centaur.watcher import FileReadyEvent
from centaur.database import Database

# Workers may accept (cfg, logger, event) OR (cfg, logger, event, ctx)
WorkerFn = Callable[..., Any]


@dataclass
class PipelineResult:
    enabled: int = 0
    skipped: int = 0
    ok: int = 0
    failed: int = 0
    failed_items: List[Tuple[str, str]] = field(default_factory=list)  # (module_name, file_path)


@dataclass
class PipelineContext:
    """
    Per-event in-memory context carrier.
    Used to pass runtime-computed per-star data from PSF-1 to downstream workers
    without storing per-star rows in the DB.
    """
    psf1: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ModuleRunRecord:
    """
    What a worker returns so the pipeline can write module_runs centrally.

    IMPORTANT:
    - If a worker returns this, it MUST NOT write into module_runs itself.
    - Worker still writes its own metric tables as usual.
    """
    image_id: int
    expected_read: int
    read: int
    expected_written: int
    written: int
    status: str  # "OK" / "FAILED" / "SKIPPED"
    message: Optional[str] = None


WorkerReturn = Union[None, bool, ModuleRunRecord]


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
    from centaur.psf_grid_worker import process_file_event as psf_grid_process
    from centaur.psf_model_worker import process_file_event as psf_model_process

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
        "psf_grid_worker": psf_grid_process,
        "psf_model_worker": psf_model_process,
    }


def _insert_module_run(
    db: Database,
    *,
    image_id: int,
    module_name: str,
    expected_fields: int,
    read_fields: int,
    written_fields: int,
    status: str,
    message: Optional[str],
    started_utc: str,
    ended_utc: str,
    duration_us: int,
) -> None:
    """
    Centralized insert into module_runs.

    duration_us is the source of truth.
    duration_ms is derived for compatibility while both columns exist.
    """
    duration_us_i = int(max(0, int(duration_us)))
    duration_ms = int(duration_us_i // 1000)

    db.execute(
        """
        INSERT INTO module_runs (
            image_id,
            module_name,
            expected_fields,
            read_fields,
            written_fields,
            status,
            message,
            started_utc,
            ended_utc,
            duration_ms,
            duration_us,
            db_written_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(image_id),
            str(module_name),
            int(expected_fields),
            int(read_fields),
            int(written_fields),
            str(status),
            message,
            started_utc,
            ended_utc,
            int(duration_ms),
            int(duration_us_i),
            utc_now(),
        ),
    )


def _run_one(
    cfg: AppConfig,
    logger: Logger,
    event: FileReadyEvent,
    registry: Dict[str, WorkerFn],
    result: PipelineResult,
    ctx: PipelineContext,
    module_name: str,
    db: Database,
) -> None:
    worker_fn = registry[module_name]

    if not cfg.is_module_enabled(module_name):
        result.skipped += 1
        return

    result.enabled += 1

    started_utc = utc_now()
    start_ns = time.perf_counter_ns()

    worker_ret: WorkerReturn
    try:
        # Prefer calling with ctx; fall back for older workers that only take 3 args.
        try:
            worker_ret = worker_fn(cfg, logger, event, ctx)
        except TypeError:
            worker_ret = worker_fn(cfg, logger, event)
    except Exception as e:
        worker_ret = False
        logger.log_failure(module_name, str(event.file_path), action="continue", reason=repr(e))

    end_ns = time.perf_counter_ns()
    ended_utc = utc_now()
    duration_us = max(0, (end_ns - start_ns) // 1000)

    # Required path: workers MUST return ModuleRunRecord
    if isinstance(worker_ret, ModuleRunRecord):
        # One expected_fields value for DB: choose the larger of the two expectations
        expected_fields_db = int(max(int(worker_ret.expected_read), int(worker_ret.expected_written)))

        with db.transaction() as tx:
            _insert_module_run(
                tx,
                image_id=int(worker_ret.image_id),
                module_name=module_name,
                expected_fields=expected_fields_db,
                read_fields=int(worker_ret.read),
                written_fields=int(worker_ret.written),
                status=str(worker_ret.status),
                message=worker_ret.message,
                started_utc=started_utc,
                ended_utc=ended_utc,
                duration_us=int(duration_us),
            )

        s = str(worker_ret.status).upper()
        if s == "FAILED":
            result.failed += 1
            result.failed_items.append((module_name, str(event.file_path)))
        elif s == "SKIPPED":
            result.skipped += 1
        else:
            result.ok += 1
        return

    # If any worker still returns old types, treat as FAILED (migration incomplete)
    result.failed += 1
    result.failed_items.append((module_name, str(event.file_path)))
    logger.log_failure(
        module=module_name,
        file=str(event.file_path),
        action="continue",
        reason="worker_did_not_return_ModuleRunRecord (migration_incomplete)",
    )


def run_pipeline_for_event(
    cfg: AppConfig,
    logger: Logger,
    event: FileReadyEvent,
    registry: Dict[str, WorkerFn],
    db: Database,
) -> PipelineResult:
    """
    Pipeline owns module_runs (duration_us).

    Requirement to be “finished”:
    - All workers return ModuleRunRecord.
    - No worker writes to module_runs directly.
    """
    result = PipelineResult()
    ctx = PipelineContext()

    # 1) Always run FITS header first
    _run_one(cfg, logger, event, registry, result, ctx, "fits_header_worker", db)
    if result.failed > 0:
        return result

    # 2) Determine frame type by reading FITS header directly (no DB dependency)
    imagetyp = ""
    try:
        from astropy.io import fits
        with fits.open(str(event.file_path), memmap=False) as hdul:
            imagetyp = str(hdul[0].header.get("IMAGETYP", "")).strip().upper()
    except Exception:
        imagetyp = ""

    # 3) Route
    if imagetyp == "FLAT":
        _run_one(cfg, logger, event, registry, result, ctx, "flat_group_worker", db)
        _run_one(cfg, logger, event, registry, result, ctx, "flat_basic_worker", db)

        for name in (
            "sky_basic_worker",
            "sky_background2d_worker",
            "exposure_advice_worker",
            "psf_detect_worker",
            "psf_basic_worker",
            "psf_grid_worker",
            "psf_model_worker",
        ):
            if cfg.is_module_enabled(name):
                result.skipped += 1
        return result

    for name in ("flat_group_worker", "flat_basic_worker"):
        if cfg.is_module_enabled(name):
            result.skipped += 1

    for name in (
        "sky_basic_worker",
        "sky_background2d_worker",
        "exposure_advice_worker",
        "psf_detect_worker",
        "psf_basic_worker",
        "psf_grid_worker",
        "psf_model_worker",
    ):
        _run_one(cfg, logger, event, registry, result, ctx, name, db)

    return result
