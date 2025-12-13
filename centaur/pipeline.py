from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

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


def build_worker_registry() -> Dict[str, WorkerFn]:
    """
    Central registry of pipeline workers.

    To add a new module later:
    1) create centaur/<new_worker>.py
    2) import its process function here
    3) add ONE line to this dict
    4) flip enabled_modules in config
    """
    from centaur.fits_header_worker import process_file_event as fits_header_process
    from centaur.sky_basic_worker import process_file_event as sky_basic_process

    return {
        # Order matters: fits header first so image_id exists for later modules.
        "fits_header_worker": fits_header_process,
        "sky_basic_worker": sky_basic_process,
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
        else:
            result.ok += 1

    return result
