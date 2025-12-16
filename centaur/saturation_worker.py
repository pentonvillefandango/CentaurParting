
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy.io import fits  # type: ignore

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent
from centaur.pipeline import ModuleRunRecord

MODULE_NAME = "saturation_worker"


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _nan_inf_fractions(arr: np.ndarray) -> Tuple[float, float]:
    total = arr.size if arr.size else 1
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    return nan_count / total, inf_count / total


def _flatten_image_data(data: np.ndarray) -> np.ndarray:
    if data is None:
        return np.array([], dtype=np.float64)

    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return np.array([], dtype=np.float64)

    return arr.astype(np.float64, copy=False)


def _get_image_id(db: Database, file_path: Path) -> Optional[int]:
    row = db.execute(
        "SELECT image_id FROM images WHERE file_path = ?",
        (str(file_path),),
    ).fetchone()
    return int(row["image_id"]) if row else None


def _get_header_fields(db: Database, image_id: int) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    row = db.execute(
        "SELECT exptime, datamax, naxis1, naxis2 FROM fits_header_core WHERE image_id = ?",
        (image_id,),
    ).fetchone()
    if not row:
        return None, None, None, None
    return _safe_float(row["exptime"]), _safe_float(row["datamax"]), row["naxis1"], row["naxis2"]


def process_file_event(cfg: AppConfig, logger: Logger, event: FileReadyEvent) -> Any:
    t0 = time.monotonic()

    expected_fields = 0
    read_fields = 0
    written_fields = 0
    image_id: Optional[int] = None
    fields: Dict[str, Any] = {}

    try:
        with fits.open(event.file_path, memmap=False) as hdul:
            data = hdul[0].data

        arr2d = _flatten_image_data(data)
        if arr2d.size == 0:
            raise ValueError("unsupported_fits_data_shape")

        nan_fraction, inf_fraction = _nan_inf_fractions(arr2d)

        finite = arr2d[np.isfinite(arr2d)]
        if finite.size == 0:
            raise ValueError("no_finite_pixels")

        max_pixel_adu = float(np.max(finite))

        with Database().transaction() as db:
            image_id = _get_image_id(db, event.file_path)
            if image_id is None:
                raise RuntimeError("image_id_not_found_for_file_path")

            exptime_s, datamax_adu, naxis1, naxis2 = _get_header_fields(db, int(image_id))

            saturation_adu = datamax_adu  # may be None

            if saturation_adu is not None:
                sat_mask = (np.isfinite(arr2d)) & (arr2d >= saturation_adu)
                saturated_pixel_count = int(sat_mask.sum())
            else:
                # Fallback: count pixels at the max value (weaker but deterministic)
                sat_mask = (np.isfinite(arr2d)) & (arr2d >= max_pixel_adu)
                saturated_pixel_count = int(sat_mask.sum())

            if naxis1 and naxis2:
                total_px = int(naxis1) * int(naxis2)
            else:
                total_px = int(arr2d.shape[0]) * int(arr2d.shape[1])
            total_px = max(1, total_px)

            saturated_pixel_fraction = float(saturated_pixel_count) / float(total_px)

            if saturation_adu is not None:
                below = finite[finite < saturation_adu]
                brightest_star_peak_adu = float(np.max(below)) if below.size else None
                headroom_fraction = _safe_float(1.0 - (brightest_star_peak_adu / saturation_adu)) if brightest_star_peak_adu is not None else None
            else:
                brightest_star_peak_adu = None
                headroom_fraction = None

            fields = {
                "exptime_s": exptime_s,
                "saturation_adu": saturation_adu,
                "max_pixel_adu": max_pixel_adu,
                "saturated_pixel_count": saturated_pixel_count,
                "saturated_pixel_fraction": saturated_pixel_fraction,
                "brightest_star_peak_adu": brightest_star_peak_adu,
                "headroom_fraction": headroom_fraction,
                "nan_fraction": nan_fraction,
                "inf_fraction": inf_fraction,
                "usable": 1,
                "reason": None,
            }

            expected_fields = len(fields)
            read_fields = expected_fields
            written_fields = sum(1 for v in fields.values() if v is not None)

            db.execute(
                """
                INSERT OR REPLACE INTO saturation_metrics (
                  image_id,
                  expected_fields, read_fields, written_fields,
                  exptime_s, parse_warnings, db_written_utc,
                  saturation_adu, max_pixel_adu, saturated_pixel_count, saturated_pixel_fraction,
                  brightest_star_peak_adu,
                  nan_fraction, inf_fraction,
                  usable, reason
                ) VALUES (
                  ?,
                  ?, ?, ?,
                  ?, ?, ?,
                  ?, ?, ?, ?,
                  ?,
                  ?, ?,
                  ?, ?
                )
                """,
                (
                    int(image_id),
                    int(expected_fields), int(read_fields), int(written_fields),
                    fields["exptime_s"], None, utc_now(),
                    fields["saturation_adu"], fields["max_pixel_adu"], fields["saturated_pixel_count"], fields["saturated_pixel_fraction"],
                    fields["brightest_star_peak_adu"],
                    fields["nan_fraction"], fields["inf_fraction"],
                    fields["usable"], fields["reason"],
                ),
            )

        duration_s = time.monotonic() - t0
        logger.log_module_result(
            module=MODULE_NAME,
            file=str(event.file_path),
            expected_read=expected_fields,
            read=expected_fields,
            expected_written=expected_fields,
            written=written_fields,
            status="OK",
            duration_s=duration_s,
            verbose_fields=fields,
        )

        return ModuleRunRecord(
            image_id=int(image_id),
            expected_read=int(expected_fields),
            read=int(read_fields),
            expected_written=0,
            written=int(written_fields),
            status="OK",
            message=None,
        )

    except Exception as e:
        duration_s = time.monotonic() - t0
        logger.log_failure(
            module=MODULE_NAME,
            file=str(event.file_path),
            action=cfg.on_metric_failure,
            reason=f"{type(e).__name__}:{e}",
            duration_s=duration_s,
        )

        if image_id is not None:
            return ModuleRunRecord(
                image_id=int(image_id),
                expected_read=int(expected_fields),
                read=int(read_fields),
                expected_written=0,
                written=int(written_fields),
                status="FAILED",
                message=f"{type(e).__name__}:{e}",
            )

        return False
