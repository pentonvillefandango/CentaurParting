from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent
from centaur.pipeline import ModuleRunRecord

# Shared pixel loader (FITS + XISF)
from centaur.io.frame_loader import load_pixels

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


def _get_image_id(db: Database, file_path: Path) -> Optional[int]:
    row = db.execute(
        "SELECT image_id FROM images WHERE file_path = ?",
        (str(file_path),),
    ).fetchone()
    return int(row["image_id"]) if row else None


def _derive_saturation_adu_from_fits_encoding(
    *,
    bitpix: Optional[int],
    bzero: Optional[float],
    bscale: Optional[float],
) -> Optional[float]:
    """
    Derive an ADU ceiling from FITS storage encoding.

    Uses FITS scaling: physical = raw * BSCALE + BZERO.

    For the common NINA/QHY case:
      BITPIX=16, BZERO=32768, BSCALE=1 => ceiling = 32767 + 32768 = 65535
    """
    if bitpix is None:
        return None

    bz = 0.0 if bzero is None else float(bzero)
    bs = 1.0 if bscale is None else float(bscale)

    if not np.isfinite(bz) or not np.isfinite(bs) or bs == 0.0:
        return None

    # Most common integer cases
    if bitpix == 16:
        raw_max = 32767.0  # signed int16
    elif bitpix == 8:
        raw_max = 255.0  # unsigned byte in FITS
    elif bitpix == 32:
        raw_max = 2147483647.0  # signed int32
    else:
        # BITPIX < 0 => float storage; ceiling not derivable generically
        return None

    ceiling = raw_max * bs + bz
    if not np.isfinite(ceiling) or ceiling <= 0:
        return None
    return float(ceiling)


def _get_header_fields(db: Database, image_id: int) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[float],
    Optional[float],
]:
    row = db.execute(
        """
        SELECT exptime, datamax, naxis1, naxis2, bitpix, bzero, bscale
        FROM fits_header_core
        WHERE image_id = ?
        """,
        (image_id,),
    ).fetchone()
    if not row:
        return None, None, None, None, None, None, None

    exptime_s = _safe_float(row["exptime"])
    datamax_adu = _safe_float(row["datamax"])
    naxis1 = row["naxis1"]
    naxis2 = row["naxis2"]
    bitpix = row["bitpix"]
    bzero = _safe_float(row["bzero"])
    bscale = _safe_float(row["bscale"])

    return exptime_s, datamax_adu, naxis1, naxis2, bitpix, bzero, bscale


def process_file_event(cfg: AppConfig, logger: Logger, event: FileReadyEvent) -> Any:
    t0 = time.monotonic()

    expected_fields = 0
    read_fields = 0
    written_fields = 0
    image_id: Optional[int] = None
    fields: Dict[str, Any] = {}

    try:
        # Load pixels (FITS + XISF) as float32 (or best-effort luminance proxy for multi-channel)
        px = load_pixels(event.file_path)  # expected 2D float32
        arr2d = np.asarray(px, dtype=np.float64)

        if arr2d.ndim == 3 and arr2d.shape[0] == 1:
            arr2d = arr2d[0]
        if arr2d.ndim != 2 or arr2d.size == 0:
            raise ValueError(
                f"unsupported_image_data_shape:{getattr(arr2d, 'shape', None)}"
            )

        nan_fraction, inf_fraction = _nan_inf_fractions(arr2d)

        finite = arr2d[np.isfinite(arr2d)]
        if finite.size == 0:
            raise ValueError("no_finite_pixels")

        max_pixel_adu = float(np.max(finite))

        with Database().transaction() as db:
            image_id = _get_image_id(db, event.file_path)
            if image_id is None:
                raise RuntimeError("image_id_not_found_for_file_path")

            exptime_s, datamax_adu, naxis1, naxis2, bitpix, bzero, bscale = (
                _get_header_fields(db, int(image_id))
            )

            parse_warnings: Optional[str] = None

            saturation_adu = datamax_adu
            if saturation_adu is None:
                saturation_adu = _derive_saturation_adu_from_fits_encoding(
                    bitpix=bitpix, bzero=bzero, bscale=bscale
                )
                if saturation_adu is not None:
                    parse_warnings = "derived_saturation_adu_from_fits_encoding"

            if saturation_adu is not None:
                sat_mask = (np.isfinite(arr2d)) & (arr2d >= float(saturation_adu))
                saturated_pixel_count = int(sat_mask.sum())
            else:
                # Fallback: count pixels at the max value (weaker but deterministic)
                sat_mask = (np.isfinite(arr2d)) & (arr2d >= max_pixel_adu)
                saturated_pixel_count = int(sat_mask.sum())
                parse_warnings = (
                    parse_warnings + ";" if parse_warnings else ""
                ) + "no_saturation_adu_ceiling"

            if naxis1 and naxis2:
                total_px = int(naxis1) * int(naxis2)
            else:
                total_px = int(arr2d.shape[0]) * int(arr2d.shape[1])
            total_px = max(1, total_px)

            saturated_pixel_fraction = float(saturated_pixel_count) / float(total_px)

            if saturation_adu is not None:
                below = finite[finite < float(saturation_adu)]
                brightest_star_peak_adu = float(np.max(below)) if below.size else None
                headroom_fraction = (
                    _safe_float(1.0 - (brightest_star_peak_adu / float(saturation_adu)))
                    if brightest_star_peak_adu is not None
                    else None
                )
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
                    int(expected_fields),
                    int(read_fields),
                    int(written_fields),
                    fields["exptime_s"],
                    parse_warnings,
                    utc_now(),
                    fields["saturation_adu"],
                    fields["max_pixel_adu"],
                    fields["saturated_pixel_count"],
                    fields["saturated_pixel_fraction"],
                    fields["brightest_star_peak_adu"],
                    fields["nan_fraction"],
                    fields["inf_fraction"],
                    fields["usable"],
                    fields["reason"],
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
