from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from astropy.stats import sigma_clip  # type: ignore

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.pipeline import ModuleRunRecord
from centaur.watcher import FileReadyEvent

# Shared pixel loader (FITS + XISF)
from centaur.io.frame_loader import load_pixels

MODULE_NAME = "nebula_mask_worker"


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


def _robust_bg_stats(
    arr2d: np.ndarray, sigma: float = 3.0, maxiters: int = 5
) -> Tuple[Optional[float], Optional[float]]:
    """
    Robust estimate of background level and scatter for thresholding.
    Uses sigma clipping on all finite pixels.
    Returns (median, madstd).
    """
    finite = arr2d[np.isfinite(arr2d)]
    if finite.size == 0:
        return None, None

    clipped = sigma_clip(
        finite, sigma=float(sigma), maxiters=int(maxiters), masked=True
    )
    mask = np.asarray(clipped.mask)
    kept = finite[~mask] if mask.size else finite
    if kept.size == 0:
        return None, None

    med = float(np.median(kept))
    mad_val = float(np.median(np.abs(kept - med)))
    madstd = 1.4826 * mad_val
    return _safe_float(med), _safe_float(madstd)


def _maybe_gaussian_smooth(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    """
    Optional Gaussian smoothing to stabilize thresholding.
    Requires SciPy (we treat SciPy as a requirement for nebula mask now).
    """
    sigma_px = float(sigma_px)
    if sigma_px <= 0:
        return arr

    # SciPy is required; if missing, fail loudly.
    from scipy.ndimage import gaussian_filter  # type: ignore

    return gaussian_filter(arr, sigma=sigma_px)


def _downsample_mask_nn(mask: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample boolean mask by integer factor (nearest-neighbor via slicing).
    This is only used for connected-components labeling if enabled.
    """
    f = int(factor)
    if f <= 1:
        return mask
    return mask[::f, ::f]


def _mask_components_stats(
    mask: np.ndarray,
    *,
    require_scipy: bool,
    downsample_factor: int,
) -> Tuple[Optional[int], Optional[float], Optional[Dict[str, int]]]:
    """
    Component count + largest component fraction + bbox (of largest component).

    - Uses scipy.ndimage.label/find_objects.
    - If downsample_factor > 1, labeling is done on a downsampled mask, and bbox is scaled back.
    - largest_component_frac is defined as (largest component pixels) / (total mask pixels)
      in the SAME domain used for labeling (i.e., downsampled if enabled).
    """
    try:
        from scipy.ndimage import find_objects, label  # type: ignore
    except Exception as e:
        if require_scipy:
            raise RuntimeError(f"scipy_required_for_nebula_mask:{type(e).__name__}:{e}")
        return None, None, None

    f = max(1, int(downsample_factor))
    m = _downsample_mask_nn(mask, f) if f > 1 else mask

    lab, n = label(m.astype(np.uint8))
    n_i = int(n)
    if n_i <= 0:
        return 0, 0.0, None

    counts = np.bincount(lab.reshape(-1))
    if counts.size <= 1:
        return n_i, 0.0, None

    largest_label = int(np.argmax(counts[1:]) + 1)
    largest_px = int(counts[largest_label])
    total_mask_px = int(m.sum())
    total_mask_px = max(1, total_mask_px)
    largest_frac = float(largest_px) / float(total_mask_px)

    sl = find_objects(lab)
    bbox = None
    if sl and len(sl) >= largest_label and sl[largest_label - 1] is not None:
        s = sl[largest_label - 1]
        y0 = int(s[0].start)
        y1 = int(s[0].stop)
        x0 = int(s[1].start)
        x1 = int(s[1].stop)

        if f > 1:
            bbox = {"x0": x0 * f, "y0": y0 * f, "x1": x1 * f, "y1": y1 * f}
        else:
            bbox = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    return n_i, _safe_float(largest_frac), bbox


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
    Build a deterministic nebula/emission mask from the current frame and store aggregate mask metrics.
    One row per image. No per-pixel storage.

    SciPy is treated as a requirement (configurable via cfg.nebula_mask_require_scipy).
    Downsampling for connected-components is optional via cfg.nebula_mask_components_downsample
    (default 1 = off).
    """
    t0 = time.monotonic()

    expected_fields = 0
    read_fields = 0
    written_fields = 0
    image_id: Optional[int] = None
    fields: Dict[str, Any] = {}

    threshold_sigma = float(getattr(cfg, "nebula_mask_threshold_sigma", 3.0))
    smooth_sigma_px = float(getattr(cfg, "nebula_mask_smooth_sigma_px", 2.0))
    clip_sigma = float(getattr(cfg, "nebula_mask_bg_clip_sigma", 3.0))
    clip_maxiters = int(getattr(cfg, "nebula_mask_bg_clip_maxiters", 5))

    require_scipy = bool(getattr(cfg, "nebula_mask_require_scipy", True))
    downsample_factor = int(getattr(cfg, "nebula_mask_components_downsample", 1))
    if downsample_factor < 1:
        downsample_factor = 1

    try:
        px = load_pixels(event.file_path)  # 2D float32 (or luminance proxy)
        arr2d = np.asarray(px, dtype=np.float64)

        # Defensive normalization
        if arr2d.ndim == 3 and arr2d.shape[0] == 1:
            arr2d = arr2d[0]
        if arr2d.ndim != 2 or arr2d.size == 0:
            raise ValueError(
                f"unsupported_image_data_shape:{getattr(arr2d, 'shape', None)}"
            )

        nan_fraction, inf_fraction = _nan_inf_fractions(arr2d)

        bg_median_adu, bg_madstd_adu = _robust_bg_stats(
            arr2d, sigma=clip_sigma, maxiters=clip_maxiters
        )
        if bg_median_adu is None or bg_madstd_adu is None or bg_madstd_adu <= 0:
            raise ValueError("unable_to_estimate_background")

        threshold_adu = _safe_float(bg_median_adu + threshold_sigma * bg_madstd_adu)
        if threshold_adu is None:
            raise ValueError("threshold_failed")

        if smooth_sigma_px > 0:
            try:
                sm = _maybe_gaussian_smooth(arr2d, smooth_sigma_px)
            except Exception as e:
                if require_scipy:
                    raise
                logger.log_failure(
                    module=MODULE_NAME,
                    file=str(event.file_path),
                    action=cfg.on_metric_failure,
                    reason=f"scipy_missing_for_smoothing:{type(e).__name__}:{e}",
                    duration_s=(time.monotonic() - t0),
                )
                sm = arr2d
        else:
            sm = arr2d

        mask = np.isfinite(sm) & (sm >= threshold_adu)

        total_px = int(arr2d.shape[0]) * int(arr2d.shape[1])
        total_px = max(1, total_px)
        mask_px = int(mask.sum())
        mask_coverage_frac = float(mask_px) / float(total_px)

        n_components, largest_component_frac, largest_bbox = _mask_components_stats(
            mask,
            require_scipy=require_scipy,
            downsample_factor=downsample_factor,
        )

        with Database().transaction() as db:
            image_id = _get_image_id(db, event.file_path)
            if image_id is None:
                raise RuntimeError("image_id_not_found_for_file_path")

            exptime_s = _get_exptime_s(db, int(image_id))

            fields = {
                "exptime_s": exptime_s,
                "threshold_sigma": threshold_sigma,
                "smooth_sigma_px": smooth_sigma_px,
                "bg_clip_sigma": clip_sigma,
                "bg_clip_maxiters": clip_maxiters,
                "bg_median_adu": bg_median_adu,
                "bg_madstd_adu": bg_madstd_adu,
                "threshold_adu": threshold_adu,
                "mask_pixel_count": mask_px,
                "mask_coverage_frac": mask_coverage_frac,
                "n_components": n_components,
                "largest_component_frac": largest_component_frac,
                "largest_component_bbox_json": (
                    json.dumps(largest_bbox) if largest_bbox is not None else None
                ),
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
                INSERT OR REPLACE INTO nebula_mask_metrics (
                  image_id,
                  expected_fields, read_fields, written_fields,
                  exptime_s,
                  threshold_sigma, smooth_sigma_px,
                  bg_clip_sigma, bg_clip_maxiters,
                  bg_median_adu, bg_madstd_adu, threshold_adu,
                  mask_pixel_count, mask_coverage_frac,
                  n_components, largest_component_frac, largest_component_bbox_json,
                  nan_fraction, inf_fraction,
                  parse_warnings, db_written_utc,
                  usable, reason
                ) VALUES (
                  ?,
                  ?, ?, ?,
                  ?,
                  ?, ?,
                  ?, ?,
                  ?, ?, ?,
                  ?, ?,
                  ?, ?, ?,
                  ?, ?,
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
                    fields["threshold_sigma"],
                    fields["smooth_sigma_px"],
                    fields["bg_clip_sigma"],
                    fields["bg_clip_maxiters"],
                    fields["bg_median_adu"],
                    fields["bg_madstd_adu"],
                    fields["threshold_adu"],
                    fields["mask_pixel_count"],
                    fields["mask_coverage_frac"],
                    fields["n_components"],
                    fields["largest_component_frac"],
                    fields["largest_component_bbox_json"],
                    fields["nan_fraction"],
                    fields["inf_fraction"],
                    None,
                    utc_now(),
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
