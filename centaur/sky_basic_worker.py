from __future__ import annotations

import time
from dataclasses import dataclass
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

# NEW: return a structured run record so pipeline can write module_runs centrally
from centaur.pipeline import ModuleRunRecord

MODULE_NAME = "sky_basic_worker"


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class SkyStats:
    # level
    mean: Optional[float]
    median: Optional[float]
    mode: Optional[float]
    sc_mean: Optional[float]
    sc_median: Optional[float]
    p10: Optional[float]
    p25: Optional[float]
    p50: Optional[float]
    p75: Optional[float]
    p90: Optional[float]
    p99: Optional[float]
    p999: Optional[float]
    vmin: Optional[float]
    vmax: Optional[float]
    # noise
    std: Optional[float]
    sc_std: Optional[float]
    mad: Optional[float]
    madstd: Optional[float]
    iqr: Optional[float]
    # clipping diagnostic
    clipped_fraction: Optional[float]


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


def _central_roi(arr: np.ndarray, roi_fraction: float) -> np.ndarray:
    """
    Extract central ROI (roi_fraction of width/height).
    roi_fraction=0.5 -> central 50% x 50%
    """
    roi_fraction = float(roi_fraction)
    roi_fraction = max(0.1, min(1.0, roi_fraction))

    h, w = arr.shape[:2]
    rh = int(h * roi_fraction)
    rw = int(w * roi_fraction)

    y0 = (h - rh) // 2
    x0 = (w - rw) // 2
    return arr[y0 : y0 + rh, x0 : x0 + rw]


def _flatten_image_data(data: np.ndarray) -> np.ndarray:
    """
    Convert FITS data to a 1D float array for statistics.

    Supports:
    - 2D images: (H, W)
    - 3D where first axis is 1: (1, H, W) -> squeeze
    """
    if data is None:
        return np.array([], dtype=np.float64)

    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        # Not an image we know how to handle
        return np.array([], dtype=np.float64)

    return arr.astype(np.float64, copy=False)


def _compute_stats(values_1d: np.ndarray) -> SkyStats:
    """
    Compute comprehensive basic sky stats on a 1D array.
    Applies sigma clipping to estimate robust background/noise.
    """
    if values_1d.size == 0:
        return SkyStats(
            mean=None, median=None, mode=None, sc_mean=None, sc_median=None,
            p10=None, p25=None, p50=None, p75=None, p90=None, p99=None, p999=None,
            vmin=None, vmax=None,
            std=None, sc_std=None, mad=None, madstd=None, iqr=None,
            clipped_fraction=None,
        )

    finite = values_1d[np.isfinite(values_1d)]
    if finite.size == 0:
        return SkyStats(
            mean=None, median=None, mode=None, sc_mean=None, sc_median=None,
            p10=None, p25=None, p50=None, p75=None, p90=None, p99=None, p999=None,
            vmin=None, vmax=None,
            std=None, sc_std=None, mad=None, madstd=None, iqr=None,
            clipped_fraction=None,
        )

    mean = _safe_float(np.mean(finite))
    median = _safe_float(np.median(finite))
    std = _safe_float(np.std(finite))
    vmin = _safe_float(np.min(finite))
    vmax = _safe_float(np.max(finite))

    p10, p25, p50, p75, p90, p99, p999 = [
        _safe_float(x) for x in np.percentile(finite, [10, 25, 50, 75, 90, 99, 99.9])
    ]

    iqr = _safe_float(p75 - p25) if (p75 is not None and p25 is not None) else None
    mode = _safe_float(3 * median - 2 * mean) if (median is not None and mean is not None) else None

    clipped = sigma_clip(finite, sigma=3.0, maxiters=5, masked=True)
    mask = np.asarray(clipped.mask)
    kept = finite[~mask] if mask.size else finite

    clipped_fraction = _safe_float(float(mask.sum()) / float(mask.size)) if mask.size else 0.0

    if kept.size == 0:
        sc_mean = None
        sc_median = None
        sc_std = None
        mad = None
        madstd = None
    else:
        sc_mean = _safe_float(np.mean(kept))
        sc_median = _safe_float(np.median(kept))
        sc_std = _safe_float(np.std(kept))

        med = np.median(kept)
        mad_val = np.median(np.abs(kept - med))
        mad = _safe_float(mad_val)
        madstd = _safe_float(1.4826 * mad_val)

    return SkyStats(
        mean=mean, median=median, mode=mode, sc_mean=sc_mean, sc_median=sc_median,
        p10=p10, p25=p25, p50=p50, p75=p75, p90=p90, p99=p99, p999=p999,
        vmin=vmin, vmax=vmax,
        std=std, sc_std=sc_std, mad=mad, madstd=madstd, iqr=iqr,
        clipped_fraction=clipped_fraction,
    )


def _get_image_id(db: Database, file_path: Path) -> Optional[int]:
    row = db.execute(
        "SELECT image_id FROM images WHERE file_path = ?",
        (str(file_path),),
    ).fetchone()
    return int(row["image_id"]) if row else None


def _get_exptime_s(db: Database, image_id: int) -> Optional[float]:
    row = db.execute(
        "SELECT exptime FROM fits_header_core WHERE image_id = ?",
        (image_id,),
    ).fetchone()
    if not row:
        return None
    return _safe_float(row["exptime"])


def process_file_event(cfg: AppConfig, logger: Logger, event: FileReadyEvent) -> Any:
    """
    Compute Sky Basic metrics from the image data and write to DB.

    NEW BEHAVIOR (for module_runs centralization):
    - This worker NO LONGER inserts into module_runs.
    - If we successfully obtain image_id, we return a ModuleRunRecord so the pipeline
      can insert into module_runs centrally (with consistent duration_us).
    """
    t0 = time.monotonic()

    roi_fraction = 0.5
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

        ff_vals = arr2d.reshape(-1)
        ff_stats = _compute_stats(ff_vals)

        roi2d = _central_roi(arr2d, roi_fraction=roi_fraction)
        roi_vals = roi2d.reshape(-1)
        roi_stats = _compute_stats(roi_vals)

        with Database().transaction() as db:
            image_id = _get_image_id(db, event.file_path)
            if image_id is None:
                raise RuntimeError("image_id_not_found_for_file_path")

            exptime_s = _get_exptime_s(db, image_id)

            if exptime_s is not None and exptime_s > 0:
                ff_median_adu_s = _safe_float(ff_stats.median / exptime_s) if ff_stats.median is not None else None
                ff_madstd_adu_s = _safe_float(ff_stats.madstd / exptime_s) if ff_stats.madstd is not None else None
                roi_median_adu_s = _safe_float(roi_stats.median / exptime_s) if roi_stats.median is not None else None
                roi_madstd_adu_s = _safe_float(roi_stats.madstd / exptime_s) if roi_stats.madstd is not None else None
            else:
                ff_median_adu_s = None
                ff_madstd_adu_s = None
                roi_median_adu_s = None
                roi_madstd_adu_s = None

            fields = {
                "exptime_s": exptime_s,
                "roi_fraction": roi_fraction,
                "nan_fraction": nan_fraction,
                "inf_fraction": inf_fraction,
                "clipped_fraction_ff": ff_stats.clipped_fraction,
                "clipped_fraction_roi": roi_stats.clipped_fraction,

                "ff_mean_adu": ff_stats.mean,
                "ff_median_adu": ff_stats.median,
                "ff_mode_adu": ff_stats.mode,
                "ff_sc_mean_adu": ff_stats.sc_mean,
                "ff_sc_median_adu": ff_stats.sc_median,
                "ff_p10_adu": ff_stats.p10,
                "ff_p25_adu": ff_stats.p25,
                "ff_p50_adu": ff_stats.p50,
                "ff_p75_adu": ff_stats.p75,
                "ff_p90_adu": ff_stats.p90,
                "ff_p99_adu": ff_stats.p99,
                "ff_p999_adu": ff_stats.p999,
                "ff_min_adu": ff_stats.vmin,
                "ff_max_adu": ff_stats.vmax,
                "ff_std_adu": ff_stats.std,
                "ff_sc_std_adu": ff_stats.sc_std,
                "ff_mad_adu": ff_stats.mad,
                "ff_madstd_adu": ff_stats.madstd,
                "ff_iqr_adu": ff_stats.iqr,
                "ff_median_adu_s": ff_median_adu_s,
                "ff_madstd_adu_s": ff_madstd_adu_s,

                "roi_mean_adu": roi_stats.mean,
                "roi_median_adu": roi_stats.median,
                "roi_mode_adu": roi_stats.mode,
                "roi_sc_mean_adu": roi_stats.sc_mean,
                "roi_sc_median_adu": roi_stats.sc_median,
                "roi_p10_adu": roi_stats.p10,
                "roi_p25_adu": roi_stats.p25,
                "roi_p50_adu": roi_stats.p50,
                "roi_p75_adu": roi_stats.p75,
                "roi_p90_adu": roi_stats.p90,
                "roi_p99_adu": roi_stats.p99,
                "roi_p999_adu": roi_stats.p999,
                "roi_min_adu": roi_stats.vmin,
                "roi_max_adu": roi_stats.vmax,
                "roi_std_adu": roi_stats.std,
                "roi_sc_std_adu": roi_stats.sc_std,
                "roi_mad_adu": roi_stats.mad,
                "roi_madstd_adu": roi_stats.madstd,
                "roi_iqr_adu": roi_stats.iqr,
                "roi_median_adu_s": roi_median_adu_s,
                "roi_madstd_adu_s": roi_madstd_adu_s,
            }

            expected_fields = len(fields)
            # “read” here means: we attempted/produced all these values; some can still be None legitimately
            read_fields = expected_fields
            written_fields = sum(1 for v in fields.values() if v is not None)

            db.execute(
                """
                INSERT OR REPLACE INTO sky_basic_metrics (
                  image_id,
                  expected_fields, read_fields, written_fields,
                  exptime_s, roi_fraction, parse_warnings, db_written_utc,
                  ff_mean_adu, ff_median_adu, ff_mode_adu, ff_sc_mean_adu, ff_sc_median_adu,
                  ff_p10_adu, ff_p25_adu, ff_p50_adu, ff_p75_adu, ff_p90_adu, ff_p99_adu, ff_p999_adu,
                  ff_min_adu, ff_max_adu,
                  ff_std_adu, ff_sc_std_adu, ff_mad_adu, ff_madstd_adu, ff_iqr_adu,
                  ff_median_adu_s, ff_madstd_adu_s,
                  roi_mean_adu, roi_median_adu, roi_mode_adu, roi_sc_mean_adu, roi_sc_median_adu,
                  roi_p10_adu, roi_p25_adu, roi_p50_adu, roi_p75_adu, roi_p90_adu, roi_p99_adu, roi_p999_adu,
                  roi_min_adu, roi_max_adu,
                  roi_std_adu, roi_sc_std_adu, roi_mad_adu, roi_madstd_adu, roi_iqr_adu,
                  roi_median_adu_s, roi_madstd_adu_s,
                  nan_fraction, inf_fraction,
                  clipped_fraction_ff, clipped_fraction_roi
                ) VALUES (
                  ?,
                  ?, ?, ?,
                  ?, ?, ?, ?,
                  ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?,
                  ?, ?,
                  ?, ?, ?, ?, ?,
                  ?, ?,
                  ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?,
                  ?, ?,
                  ?, ?, ?, ?, ?,
                  ?, ?,
                  ?, ?,
                  ?, ?
                )
                """,
                (
                    image_id,
                    expected_fields, read_fields, written_fields,
                    fields["exptime_s"], fields["roi_fraction"], None, utc_now(),
                    fields["ff_mean_adu"], fields["ff_median_adu"], fields["ff_mode_adu"], fields["ff_sc_mean_adu"], fields["ff_sc_median_adu"],
                    fields["ff_p10_adu"], fields["ff_p25_adu"], fields["ff_p50_adu"], fields["ff_p75_adu"], fields["ff_p90_adu"], fields["ff_p99_adu"], fields["ff_p999_adu"],
                    fields["ff_min_adu"], fields["ff_max_adu"],
                    fields["ff_std_adu"], fields["ff_sc_std_adu"], fields["ff_mad_adu"], fields["ff_madstd_adu"], fields["ff_iqr_adu"],
                    fields["ff_median_adu_s"], fields["ff_madstd_adu_s"],
                    fields["roi_mean_adu"], fields["roi_median_adu"], fields["roi_mode_adu"], fields["roi_sc_mean_adu"], fields["roi_sc_median_adu"],
                    fields["roi_p10_adu"], fields["roi_p25_adu"], fields["roi_p50_adu"], fields["roi_p75_adu"], fields["roi_p90_adu"], fields["roi_p99_adu"], fields["roi_p999_adu"],
                    fields["roi_min_adu"], fields["roi_max_adu"],
                    fields["roi_std_adu"], fields["roi_sc_std_adu"], fields["roi_mad_adu"], fields["roi_madstd_adu"], fields["roi_iqr_adu"],
                    fields["roi_median_adu_s"], fields["roi_madstd_adu_s"],
                    fields["nan_fraction"], fields["inf_fraction"],
                    fields["clipped_fraction_ff"], fields["clipped_fraction_roi"],
                ),
            )

        # Logging stays the same style
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

        # Tell pipeline to write module_runs centrally (duration_us measured in pipeline)
        return ModuleRunRecord(
            image_id=int(image_id),
            expected_read=int(expected_fields),
            read=int(read_fields),
            expected_written=0,  # IMPORTANT: pipeline currently sums expected_read + expected_written
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

        # If we managed to obtain image_id before failing, return a record so pipeline can audit it.
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
