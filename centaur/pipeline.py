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

    return {
        # Must run first so image_id exists and header tables are populated.
        "fits_header_worker": fits_header_process,

        # Sky metrics
        "sky_basic_worker": sky_basic_process,
        "sky_background2d_worker": sky_bkg2d_process,

        # Decision layer (Module0)
        "exposure_advice_worker": exposure_advice_process,
    }


def run_pipeline_for_event(
    cfg: AppConfig,
    logger: Logger,
    event: FileReadyEvent,
    registry: Dict[str, WorkerFn],
) -> PipelineResult:
    result = PipelineResult()

    for module_name, worker_fn in registry.items():
        if not cfg.is_module_enabled(module_name):
            result.skipped += 1
            continue

        result.enabled += 1
        worker_result = worker_fn(cfg, logger, event)

        if worker_result is False:
            result.failed += 1
            result.failed_items.append((module_name, str(event.file_path)))
        else:
            result.ok += 1

    return result
