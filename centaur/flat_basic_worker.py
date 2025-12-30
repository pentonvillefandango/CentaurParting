from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional, Any

import numpy as np

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent
from centaur.pipeline import ModuleRunRecord

# V1b pixels loader (FITS + XISF)
from centaur.io.frame_loader import load_pixels

MODULE_NAME = "flat_basic_worker"


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _is_flat(imagetyp: Optional[str]) -> bool:
    return (imagetyp or "").strip().upper() == "FLAT"


def process_file_event(cfg: AppConfig, logger: Logger, event: FileReadyEvent) -> Any:
    """
    Optical flat metrics (v1b).

    Runs only when IMAGETYP=FLAT (from DB fits_header_core).
    Writes one row into flat_metrics keyed by image_id.
    """
    t0 = time.monotonic()
    file_path = str(event.file_path)

    image_id: Optional[int] = None

    try:
        # Decide applicability from DB (works for FITS + XISF since fits_header_worker populates fits_header_core)
        with Database().transaction() as db:
            row = db.execute(
                """
                SELECT i.image_id, f.imagetyp
                FROM images i
                JOIN fits_header_core f USING(image_id)
                WHERE i.file_path = ?
                """,
                (file_path,),
            ).fetchone()

            if row is None:
                logger.log_failure(
                    module=MODULE_NAME,
                    file=file_path,
                    action="continue",
                    reason="missing_fits_header_core (fits_header_worker must run first)",
                    duration_s=time.monotonic() - t0,
                )
                return False

            image_id = int(row["image_id"])

            if not _is_flat(row["imagetyp"]):
                # Not a flat; not applicable.
                return ModuleRunRecord(
                    image_id=image_id,
                    expected_read=1,
                    read=1,
                    expected_written=0,
                    written=0,
                    status="OK",
                    message="not_applicable_non_flat",
                )

        # Load pixels (FITS or XISF)
        arr2d = np.asarray(load_pixels(event.file_path), dtype=np.float32)
        if arr2d.ndim != 2 or arr2d.size == 0:
            raise ValueError("unsupported_image_data_shape")

        # Robust stats
        median = float(np.median(arr2d))
        mean = float(np.mean(arr2d))
        std = float(np.std(arr2d))
        madstd = float(1.4826 * np.median(np.abs(arr2d - median)))
        minv = float(np.min(arr2d))
        maxv = float(np.max(arr2d))

        # Clip fractions (robust percentile-based)
        p01 = float(np.percentile(arr2d, 1))
        p99 = float(np.percentile(arr2d, 99))
        clipped_low = float(np.mean(arr2d <= p01))
        clipped_high = float(np.mean(arr2d >= p99))

        # Corner median vs full-frame median as a simple vignetting proxy
        h, w = arr2d.shape
        ch = max(1, h // 10)
        cw = max(1, w // 10)
        corners = np.concatenate(
            [
                arr2d[:ch, :cw].ravel(),
                arr2d[:ch, -cw:].ravel(),
                arr2d[-ch:, :cw].ravel(),
                arr2d[-ch:, -cw:].ravel(),
            ]
        )
        corner_med = float(np.median(corners))

        corner_vignette_ratio_raw = float(corner_med / median) if median != 0 else 0.0
        corner_vignette_frac = max(0.0, min(1.0, corner_vignette_ratio_raw))

        # Gradient estimate via image gradients (simple)
        gy, gx = np.gradient(arr2d)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        grad_p95 = float(np.percentile(grad_mag, 95))

        # Basic usability heuristic (v1)
        usable = int((median > 0) and (clipped_high < 0.02) and (clipped_low < 0.02))

        with Database().transaction() as db:
            db.execute(
                """
                INSERT OR REPLACE INTO flat_metrics (
                    image_id,
                    mean_adu, median_adu, std_adu, madstd_adu,
                    min_adu, max_adu,
                    clipped_low_frac, clipped_high_frac,
                    corner_vignette_frac,
                    corner_vignette_ratio_raw,
                    gradient_p95,
                    usable,
                    db_written_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    int(image_id),
                    mean,
                    median,
                    std,
                    madstd,
                    minv,
                    maxv,
                    clipped_low,
                    clipped_high,
                    corner_vignette_frac,
                    corner_vignette_ratio_raw,
                    grad_p95,
                    usable,
                    utc_now(),
                ),
            )

        logger.log_module_summary(
            module=MODULE_NAME,
            file=file_path,
            expected_read=1,
            read=1,
            expected_written=1,
            written=1,
            status="OK",
            duration_s=time.monotonic() - t0,
        )

        return ModuleRunRecord(
            image_id=int(image_id),
            expected_read=1,
            read=1,
            expected_written=1,
            written=1,
            status="OK",
            message=None,
        )

    except Exception as e:
        logger.log_failure(
            module=MODULE_NAME,
            file=file_path,
            action="continue",
            reason=f"{type(e).__name__}:{e}",
            duration_s=time.monotonic() - t0,
        )

        if image_id is not None:
            return ModuleRunRecord(
                image_id=int(image_id),
                expected_read=1,
                read=0,
                expected_written=1,
                written=0,
                status="FAILED",
                message=f"{type(e).__name__}:{e}",
            )

        return False
