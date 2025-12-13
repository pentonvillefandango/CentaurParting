from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from astropy.io import fits

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent


MODULE_NAME = "flat_basic_worker"


def _is_flat(hdr) -> bool:
    return str(hdr.get("IMAGETYP", "")).strip().upper() == "FLAT"


def process_file_event(cfg: AppConfig, logger: Logger, event: FileReadyEvent) -> Optional[bool]:
    """
    Optical flat metrics (v1).

    Runs only when IMAGETYP=FLAT.
    Writes one row into flat_metrics keyed by image_id.

    Returns:
      True  -> success or not-applicable (non-flat)
      False -> failure
    """
    start = time.monotonic()
    file_path = str(event.file_path)

    try:
        with fits.open(file_path, memmap=False) as hdul:
            hdr = hdul[0].header
            if not _is_flat(hdr):
                # Not a flat; this worker is not applicable.
                return True

            data = hdul[0].data
            if data is None:
                logger.log_failure(
                    module=MODULE_NAME,
                    file=file_path,
                    action="continue",
                    reason="no_image_data",
                )
                return False

            arr = np.asarray(data, dtype=np.float32)

        # Robust stats
        median = float(np.median(arr))
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        madstd = float(1.4826 * np.median(np.abs(arr - median)))
        minv = float(np.min(arr))
        maxv = float(np.max(arr))

        # Clip fractions (basic; we can refine later once we standardize bit depth behavior)
        # We use robust percentiles so we don't need exact saturation ADU here.
        p01 = float(np.percentile(arr, 1))
        p99 = float(np.percentile(arr, 99))

        # Treat values at/below p01 as "low-clipped-ish" and at/above p99 as "high-clipped-ish"
        # (This gives a consistent signal even when we don't know the camera's true max ADU.)
        clipped_low = float(np.mean(arr <= p01))
        clipped_high = float(np.mean(arr >= p99))

        # Corner median vs full-frame median as a simple vignetting proxy
        h, w = arr.shape
        ch = max(1, h // 10)
        cw = max(1, w // 10)
        corners = np.concatenate(
            [
                arr[:ch, :cw].ravel(),
                arr[:ch, -cw:].ravel(),
                arr[-ch:, :cw].ravel(),
                arr[-ch:, -cw:].ravel(),
            ]
        )
        corner_med = float(np.median(corners))
        corner_vignette_frac = float(corner_med / median) if median != 0 else 0.0

        # Gradient estimate via image gradients (simple)
        gy, gx = np.gradient(arr)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        grad_p95 = float(np.percentile(grad_mag, 95))

        # Basic usability heuristic (v1; strict pass/fail comes later)
        usable = int(
            (median > 0) and
            (clipped_high < 0.02) and
            (clipped_low < 0.02)
        )

        # Write to DB (needs image_id from images table)
        with Database().transaction() as db:
            row = db.execute(
                "SELECT image_id FROM images WHERE file_path = ?",
                (file_path,),
            ).fetchone()

            if row is None:
                logger.log_failure(
                    module=MODULE_NAME,
                    file=file_path,
                    action="continue",
                    reason="image_id_not_found (fits_header_worker must run first)",
                )
                return False

            image_id = int(row["image_id"])

            db.execute(
                """
                INSERT OR REPLACE INTO flat_metrics (
                    image_id,
                    mean_adu, median_adu, std_adu, madstd_adu,
                    min_adu, max_adu,
                    clipped_low_frac, clipped_high_frac,
                    corner_vignette_frac, gradient_p95,
                    usable,
                    db_written_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    image_id,
                    mean, median, std, madstd,
                    minv, maxv,
                    clipped_low, clipped_high,
                    corner_vignette_frac, grad_p95,
                    usable,
                    datetime.now(timezone.utc).isoformat(),
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
            duration_s=time.monotonic() - start,
        )
        return True

    except Exception as e:
        logger.log_failure(
            module=MODULE_NAME,
            file=file_path,
            action="continue",
            reason=f"{type(e).__name__}:{e}",
        )
        return False
