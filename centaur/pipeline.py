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
    failed_items: List[Tuple[str, str]] = field(default_factory=list)


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
    from centaur.saturation_worker import process_file_event as saturation_process
    from centaur.roi_signal_worker import process_file_event as roi_signal_process
    from centaur.exposure_advice_worker import (
        process_file_event as exposure_advice_process,
    )
    from centaur.flat_basic_worker import process_file_event as flat_basic_process
    from centaur.flat_group_worker import process_file_event as flat_group_process
    from centaur.psf_detect_worker import process_file_event as psf_detect_process
    from centaur.psf_basic_worker import process_file_event as psf_basic_process
    from centaur.psf_grid_worker import process_file_event as psf_grid_process
    from centaur.psf_model_worker import process_file_event as psf_model_process
    from centaur.signal_structure_worker import (
        process_file_event as signal_structure_process,
    )
    from centaur.nebula_mask_worker import process_file_event as nebula_mask_process
    from centaur.masked_signal_worker import process_file_event as masked_signal_process
    from centaur.star_headroom_worker import process_file_event as star_headroom_process
    from centaur.frame_quality_worker import process_file_event as frame_quality_process
    from centaur.observing_conditions_worker import (
        process_file_event as observing_conditions_process,
    )
    from centaur.training_derived_worker import (
        process_file_event as training_derived_process,
    )

    return {
        # Must run first
        "fits_header_worker": fits_header_process,
        # Observing conditions - part of training
        "observing_conditions_worker": observing_conditions_process,
        # Flats
        "flat_group_worker": flat_group_process,
        "flat_basic_worker": flat_basic_process,
        # Sky / signal
        "sky_basic_worker": sky_basic_process,
        "sky_background2d_worker": sky_bkg2d_process,
        "saturation_worker": saturation_process,
        "roi_signal_worker": roi_signal_process,
        # Decision / advice
        "exposure_advice_worker": exposure_advice_process,
        # PSF
        "psf_detect_worker": psf_detect_process,
        "psf_basic_worker": psf_basic_process,
        "psf_grid_worker": psf_grid_process,
        "psf_model_worker": psf_model_process,
        # Structure / masking
        "signal_structure_worker": signal_structure_process,
        "nebula_mask_worker": nebula_mask_process,
        "masked_signal_worker": masked_signal_process,
        "star_headroom_worker": star_headroom_process,
        # Final synthesis
        "frame_quality_worker": frame_quality_process,
        "training_derived_worker": training_derived_process,
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
    duration_us_i = int(max(0, duration_us))
    duration_ms = duration_us_i // 1000

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
            duration_us_i,
            utc_now(),
        ),
    )


def _lookup_image_id(db: Database, event: FileReadyEvent) -> Optional[int]:
    row = db.execute(
        "SELECT image_id FROM images WHERE file_path = ?",
        (str(event.file_path),),
    ).fetchone()
    return int(row["image_id"]) if row else None


def _read_imagetyp_from_db(db: Database, image_id: int) -> str:
    row = db.execute(
        "SELECT imagetyp FROM fits_header_core WHERE image_id = ?",
        (int(image_id),),
    ).fetchone()
    if not row:
        return ""
    v = row["imagetyp"]
    return str(v or "").strip().upper()


def _coerce_to_run_record(
    db: Database,
    event: FileReadyEvent,
    module_name: str,
    worker_ret: WorkerReturn,
    *,
    started_utc: str,
    ended_utc: str,
    duration_us: int,
    failure_reason: Optional[str],
) -> Optional[ModuleRunRecord]:
    """
    Convert legacy returns (True/False/None) or exceptions into a ModuleRunRecord.
    - True  -> SKIPPED (not-applicable / legacy success)
    - False/None -> FAILED
    Returns None if we cannot resolve image_id.
    """
    image_id = _lookup_image_id(db, event)
    if image_id is None:
        return None

    if isinstance(worker_ret, ModuleRunRecord):
        return worker_ret

    if worker_ret is True:
        return ModuleRunRecord(
            image_id=int(image_id),
            expected_read=0,
            read=0,
            expected_written=0,
            written=0,
            status="SKIPPED",
            message="legacy_true_return",
        )

    # False or None
    msg = failure_reason or "legacy_false_or_none_return"
    return ModuleRunRecord(
        image_id=int(image_id),
        expected_read=0,
        read=0,
        expected_written=0,
        written=0,
        status="FAILED",
        message=str(msg),
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

    worker_ret: WorkerReturn = None
    failure_reason: Optional[str] = None

    try:
        try:
            worker_ret = worker_fn(cfg, logger, event, ctx)
        except TypeError:
            worker_ret = worker_fn(cfg, logger, event)
    except Exception as e:
        # Log, but ALSO create a failed module_runs row (pipeline owns module_runs).
        failure_reason = f"{type(e).__name__}:{e}"
        logger.log_failure(
            module=module_name,
            file=str(event.file_path),
            action=cfg.on_metric_failure,
            reason=failure_reason,
        )
        worker_ret = None

    end_ns = time.perf_counter_ns()
    ended_utc = utc_now()
    duration_us = max(0, (end_ns - start_ns) // 1000)

    # If worker returned a proper record, use it.
    if isinstance(worker_ret, ModuleRunRecord):
        expected_fields_db = max(worker_ret.expected_read, worker_ret.expected_written)
        with db.transaction() as tx:
            _insert_module_run(
                tx,
                image_id=worker_ret.image_id,
                module_name=module_name,
                expected_fields=expected_fields_db,
                read_fields=worker_ret.read,
                written_fields=worker_ret.written,
                status=worker_ret.status,
                message=worker_ret.message,
                started_utc=started_utc,
                ended_utc=ended_utc,
                duration_us=duration_us,
            )

        if worker_ret.status == "FAILED":
            result.failed += 1
            result.failed_items.append((module_name, str(event.file_path)))
        elif worker_ret.status == "SKIPPED":
            result.skipped += 1
        else:
            result.ok += 1
        return

    # Legacy/bad return type: coerce to ModuleRunRecord if possible.
    coerced = _coerce_to_run_record(
        db,
        event,
        module_name,
        worker_ret,
        started_utc=started_utc,
        ended_utc=ended_utc,
        duration_us=duration_us,
        failure_reason=failure_reason or "worker_did_not_return_ModuleRunRecord",
    )

    if coerced is not None:
        expected_fields_db = max(coerced.expected_read, coerced.expected_written)
        with db.transaction() as tx:
            _insert_module_run(
                tx,
                image_id=coerced.image_id,
                module_name=module_name,
                expected_fields=expected_fields_db,
                read_fields=coerced.read,
                written_fields=coerced.written,
                status=coerced.status,
                message=coerced.message,
                started_utc=started_utc,
                ended_utc=ended_utc,
                duration_us=duration_us,
            )

        if coerced.status == "FAILED":
            result.failed += 1
            result.failed_items.append((module_name, str(event.file_path)))
        elif coerced.status == "SKIPPED":
            result.skipped += 1
        else:
            result.ok += 1
        return

    # If we couldn't even resolve image_id, treat as failed (but we can't write module_runs).
    result.failed += 1
    result.failed_items.append((module_name, str(event.file_path)))
    logger.log_failure(
        module=module_name,
        file=str(event.file_path),
        action=cfg.on_metric_failure,
        reason="worker_failed_and_image_id_unknown",
    )


def run_pipeline_for_event(
    cfg: AppConfig,
    logger: Logger,
    event: FileReadyEvent,
    registry: Dict[str, WorkerFn],
    db: Database,
) -> PipelineResult:
    result = PipelineResult()
    ctx = PipelineContext()

    # Always run header
    _run_one(cfg, logger, event, registry, result, ctx, "fits_header_worker", db)
    if result.failed:
        return result

    # Routing: read IMAGETYP from DB (fits_header_worker already normalized source)
    image_id = _lookup_image_id(db, event)
    imagetyp = _read_imagetyp_from_db(db, image_id) if image_id is not None else ""

    # FLATS
    if imagetyp == "FLAT":
        _run_one(cfg, logger, event, registry, result, ctx, "flat_group_worker", db)
        _run_one(cfg, logger, event, registry, result, ctx, "flat_basic_worker", db)

        # Explicitly count as skipped for LIGHT-only workers if enabled
        for name in (
            "sky_basic_worker",
            "sky_background2d_worker",
            "signal_structure_worker",
            "nebula_mask_worker",
            "saturation_worker",
            "roi_signal_worker",
            "exposure_advice_worker",
            "psf_detect_worker",
            "psf_basic_worker",
            "psf_grid_worker",
            "psf_model_worker",
            "masked_signal_worker",
            "star_headroom_worker",
            "frame_quality_worker",
            "observing_conditions_worker",
            "training_derived_worker",
        ):
            if cfg.is_module_enabled(name):
                result.skipped += 1

        return result

    # Skip flat workers on LIGHT
    for name in ("flat_group_worker", "flat_basic_worker"):
        if cfg.is_module_enabled(name):
            result.skipped += 1

    # LIGHT chain
    for name in (
        "sky_basic_worker",
        "sky_background2d_worker",
        "saturation_worker",
        "roi_signal_worker",
        "psf_detect_worker",
        "psf_basic_worker",
        "psf_grid_worker",
        "psf_model_worker",
        "signal_structure_worker",
        "nebula_mask_worker",
        "masked_signal_worker",
        "star_headroom_worker",
        "exposure_advice_worker",
        "frame_quality_worker",
        "observing_conditions_worker",
        "training_derived_worker",
    ):
        _run_one(cfg, logger, event, registry, result, ctx, name, db)

    return result
