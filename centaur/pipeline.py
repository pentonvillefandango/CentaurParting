from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from centaur.config import AppConfig
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent

# A worker returns:
# - True  -> success
# - False -> failure
# - None  -> "no explicit result" (treat as success; worker already logs OK/FAILED)
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

    return {
        "fits_header_worker": fits_header_process,
        # Future:
        # "sky_metrics_worker": sky_metrics_process,
        # "psf_metrics_worker": psf_metrics_process,
    }


def run_pipeline_for_event(
    cfg: AppConfig,
    logger: Logger,
    event: FileReadyEvent,
    registry: Dict[str, WorkerFn],
) -> PipelineResult:
    """
    Run all enabled workers for a single FileReadyEvent.

    This function is intentionally simple and explicit.
    """
    result = PipelineResult()

    for module_name, worker_fn in registry.items():
        if not cfg.is_module_enabled(module_name):
            result.skipped += 1
            continue

        result.enabled += 1
        worker_result = worker_fn(cfg, logger, event)

        # Interpret results safely:
        # - If a worker returns False -> count as failed
        # - Otherwise (True or None) -> count as ok
        if worker_result is False:
            result.failed += 1
        else:
            result.ok += 1

    return result

