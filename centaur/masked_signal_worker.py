from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy.io import fits  # type: ignore
from astropy.stats import sigma_clip  # type: ignore

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent
from centaur.pipeline import ModuleRunRecord

MODULE_NAME = "masked_signal_worker"


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _flatten(data: Any) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return np.array([], dtype=np.float64)
    return arr.astype(np.float64, copy=False)


def _robust_stats(vals: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if vals.size == 0:
        return None, None

    clipped = sigma_clip(vals, sigma=3.0, maxiters=5, masked=True)
    mask = np.asarray(clipped.mask)
    kept = vals[~mask] if mask.size else vals
    if kept.size == 0:
        return None, None

    med = float(np.median(kept))
    mad = float(np.median(np.abs(kept - med)))
    return _safe_float(med), _safe_float(1.4826 * mad)


def _get_image_id(db: Database, path: Path) -> Optional[int]:
    r = db.execute(
        "SELECT image_id FROM images WHERE file_path=?",
        (str(path),),
    ).fetchone()
    return int(r["image_id"]) if r else None


def _get_exptime(db: Database, image_id: int) -> Optional[float]:
    r = db.execute(
        "SELECT exptime FROM fits_header_core WHERE image_id=?",
        (image_id,),
    ).fetchone()
    return _safe_float(r["exptime"]) if r else None


def process_file_event(cfg: AppConfig, logger: Logger, event: FileReadyEvent) -> Any:
    t0 = time.monotonic()

    image_id: Optional[int] = None
    expected = written = 0

    try:
        with fits.open(event.file_path, memmap=False) as hdul:
            arr = _flatten(hdul[0].data)
        if arr.size == 0:
            raise ValueError("unsupported_fits_data_shape")

        with Database().transaction() as db:
            image_id = _get_image_id(db, event.file_path)
            if image_id is None:
                raise RuntimeError("image_id_not_found")

            exptime_s = _get_exptime(db, image_id)

            nm = db.execute(
                "SELECT threshold_adu FROM nebula_mask_metrics WHERE image_id=?",
                (image_id,),
            ).fetchone()
            if not nm or nm["threshold_adu"] is None:
                raise RuntimeError("nebula_mask_missing")

            threshold = float(nm["threshold_adu"])
            finite = np.isfinite(arr)

            nebula = finite & (arr >= threshold)
            bg = finite & (arr < threshold)

            neb_vals = arr[nebula]
            bg_vals = arr[bg]

            neb_med, neb_mad = _robust_stats(neb_vals)
            bg_med, bg_mad = _robust_stats(bg_vals)

            neb_minus_bg = (
                neb_med - bg_med if neb_med is not None and bg_med is not None else None
            )
            neb_minus_bg_s = (
                neb_minus_bg / exptime_s
                if neb_minus_bg is not None and exptime_s and exptime_s > 0
                else None
            )

            snr = (
                neb_minus_bg / bg_mad
                if neb_minus_bg is not None and bg_mad and bg_mad > 0
                else None
            )
            snr_s = (
                snr / exptime_s
                if snr is not None and exptime_s and exptime_s > 0
                else None
            )

            fields: Dict[str, Any] = {
                "exptime_s": exptime_s,
                "nebula_pixel_count": int(nebula.sum()),
                "bg_pixel_count": int(bg.sum()),
                "nebula_frac": _safe_float(
                    float(nebula.sum()) / float(max(1, arr.size))
                ),
                "nebula_median_adu": neb_med,
                "nebula_madstd_adu": neb_mad,
                "bg_median_adu": bg_med,
                "bg_madstd_adu": bg_mad,
                "nebula_minus_bg_adu": neb_minus_bg,
                "nebula_minus_bg_adu_s": neb_minus_bg_s,
                "snr_proxy": snr,
                "snr_proxy_s": snr_s,
                "usable": 1,
                "reason": None,
            }

            expected = len(fields)
            written = sum(v is not None for v in fields.values())

            db.execute(
                """
                INSERT OR REPLACE INTO masked_signal_metrics (
                  image_id,
                  expected_fields, read_fields, written_fields,
                  exptime_s,
                  nebula_pixel_count, bg_pixel_count, nebula_frac,
                  nebula_median_adu, nebula_madstd_adu,
                  bg_median_adu, bg_madstd_adu,
                  nebula_minus_bg_adu, nebula_minus_bg_adu_s,
                  snr_proxy, snr_proxy_s,
                  parse_warnings, db_written_utc,
                  usable, reason
                ) VALUES (
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    image_id,
                    expected,
                    expected,
                    written,
                    fields["exptime_s"],
                    fields["nebula_pixel_count"],
                    fields["bg_pixel_count"],
                    fields["nebula_frac"],
                    fields["nebula_median_adu"],
                    fields["nebula_madstd_adu"],
                    fields["bg_median_adu"],
                    fields["bg_madstd_adu"],
                    fields["nebula_minus_bg_adu"],
                    fields["nebula_minus_bg_adu_s"],
                    fields["snr_proxy"],
                    fields["snr_proxy_s"],
                    None,
                    utc_now(),
                    fields["usable"],
                    fields["reason"],
                ),
            )

        logger.log_module_result(
            module=MODULE_NAME,
            file=str(event.file_path),
            expected_read=expected,
            read=expected,
            expected_written=expected,
            written=written,
            status="OK",
            duration_s=time.monotonic() - t0,
            verbose_fields=fields,
        )

        return ModuleRunRecord(
            image_id=image_id,
            expected_read=expected,
            read=expected,
            expected_written=expected,
            written=written,
            status="OK",
            message=None,
        )

    except Exception as e:
        logger.log_failure(
            module=MODULE_NAME,
            file=str(event.file_path),
            action=cfg.on_metric_failure,
            reason=f"{type(e).__name__}:{e}",
            duration_s=time.monotonic() - t0,
        )
        if image_id is not None:
            return ModuleRunRecord(
                image_id=image_id,
                expected_read=expected,
                read=expected,
                expected_written=expected,
                written=written,
                status="FAILED",
                message=str(e),
            )
        return False
