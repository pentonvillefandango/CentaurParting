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
from centaur.watcher import FileReadyEvent
from centaur.pipeline import ModuleRunRecord

# Shared pixel loader (FITS + XISF)
from centaur.io.frame_loader import load_pixels

MODULE_NAME = "roi_signal_worker"


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


def _central_bbox(h: int, w: int, frac: float) -> Tuple[int, int, int, int]:
    frac = float(frac)
    frac = max(0.05, min(1.0, frac))
    rh = max(1, int(h * frac))
    rw = max(1, int(w * frac))
    y0 = (h - rh) // 2
    x0 = (w - rw) // 2
    y1 = y0 + rh
    x1 = x0 + rw
    return x0, y0, x1, y1


def _extract_bbox(arr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return arr[y0:y1, x0:x1]


def _compute_median_madstd_and_clipfrac(
    values_1d: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if values_1d.size == 0:
        return None, None, None

    finite = values_1d[np.isfinite(values_1d)]
    if finite.size == 0:
        return None, None, None

    clipped = sigma_clip(finite, sigma=3.0, maxiters=5, masked=True)
    mask = np.asarray(clipped.mask)
    kept = finite[~mask] if mask.size else finite

    clipped_fraction = (
        _safe_float(float(mask.sum()) / float(mask.size)) if mask.size else 0.0
    )
    if kept.size == 0:
        return None, None, clipped_fraction

    med = float(np.median(kept))
    mad_val = float(np.median(np.abs(kept - med)))
    madstd = _safe_float(1.4826 * mad_val)
    return _safe_float(med), madstd, clipped_fraction


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
    t0 = time.monotonic()

    expected_fields = 0
    read_fields = 0
    written_fields = 0
    image_id: Optional[int] = None
    fields: Dict[str, Any] = {}

    roi_kind = "target_v1"

    # Deterministic defaults (configurable via AppConfig)
    obj_frac = float(getattr(cfg, "roi_signal_obj_fraction", 0.20))
    bg_outer_frac = float(getattr(cfg, "roi_signal_bg_outer_fraction", 0.45))

    try:
        # Load pixels (FITS + XISF)
        px = load_pixels(event.file_path)  # expected 2D float32
        arr2d = np.asarray(px, dtype=np.float64)

        # Defensive shape normalization
        if arr2d.ndim == 3 and arr2d.shape[0] == 1:
            arr2d = arr2d[0]
        if arr2d.ndim != 2 or arr2d.size == 0:
            raise ValueError(
                f"unsupported_image_data_shape:{getattr(arr2d, 'shape', None)}"
            )

        h, w = arr2d.shape[:2]

        obj_bbox = _central_bbox(h, w, obj_frac)
        bg_outer_bbox = _central_bbox(h, w, bg_outer_frac)

        obj = _extract_bbox(arr2d, obj_bbox)
        bg_outer = _extract_bbox(arr2d, bg_outer_bbox)

        # Background ring = outer box minus inner object box
        x0o, y0o, x1o, y1o = obj_bbox
        x0b, y0b, x1b, y1b = bg_outer_bbox

        inner_x0 = max(0, x0o - x0b)
        inner_y0 = max(0, y0o - y0b)
        inner_x1 = min(bg_outer.shape[1], x1o - x0b)
        inner_y1 = min(bg_outer.shape[0], y1o - y0b)

        bg_vals = bg_outer.reshape(-1)
        if inner_x1 > inner_x0 and inner_y1 > inner_y0:
            mask = np.ones(bg_outer.shape, dtype=bool)
            mask[inner_y0:inner_y1, inner_x0:inner_x1] = False
            bg_vals = bg_outer[mask].reshape(-1)

        obj_vals = obj.reshape(-1)

        obj_median_adu, obj_madstd_adu, clipped_fraction_obj = (
            _compute_median_madstd_and_clipfrac(obj_vals)
        )
        bg_median_adu, bg_madstd_adu, clipped_fraction_bg = (
            _compute_median_madstd_and_clipfrac(bg_vals)
        )

        obj_minus_bg_adu = (
            _safe_float(obj_median_adu - bg_median_adu)
            if (obj_median_adu is not None and bg_median_adu is not None)
            else None
        )

        with Database().transaction() as db:
            image_id = _get_image_id(db, event.file_path)
            if image_id is None:
                raise RuntimeError("image_id_not_found_for_file_path")

            exptime_s = _get_exptime_s(db, int(image_id))
            obj_minus_bg_adu_s = (
                _safe_float(obj_minus_bg_adu / exptime_s)
                if (exptime_s and exptime_s > 0 and obj_minus_bg_adu is not None)
                else None
            )

            obj_area = int(obj.shape[0]) * int(obj.shape[1])
            roi_fraction = float(obj_area) / float(max(1, h * w))

            fields = {
                "exptime_s": exptime_s,
                "roi_kind": roi_kind,
                "roi_fraction": roi_fraction,
                "roi_bbox_json": json.dumps(
                    {
                        "x0": obj_bbox[0],
                        "y0": obj_bbox[1],
                        "x1": obj_bbox[2],
                        "y1": obj_bbox[3],
                    }
                ),
                "bg_bbox_json": json.dumps(
                    {
                        "x0": bg_outer_bbox[0],
                        "y0": bg_outer_bbox[1],
                        "x1": bg_outer_bbox[2],
                        "y1": bg_outer_bbox[3],
                    }
                ),
                "obj_median_adu": obj_median_adu,
                "obj_madstd_adu": obj_madstd_adu,
                "bg_median_adu": bg_median_adu,
                "bg_madstd_adu": bg_madstd_adu,
                "obj_minus_bg_adu": obj_minus_bg_adu,
                "obj_minus_bg_adu_s": obj_minus_bg_adu_s,
                "clipped_fraction_obj": clipped_fraction_obj,
                "clipped_fraction_bg": clipped_fraction_bg,
                "usable": 1,
                "reason": None,
            }

            expected_fields = len(fields)
            read_fields = expected_fields
            written_fields = sum(1 for v in fields.values() if v is not None)

            db.execute(
                """
                INSERT OR REPLACE INTO roi_signal_metrics (
                  image_id,
                  expected_fields, read_fields, written_fields,
                  exptime_s,
                  roi_kind, roi_fraction, roi_bbox_json, bg_bbox_json,
                  obj_median_adu, obj_madstd_adu, bg_median_adu, bg_madstd_adu,
                  obj_minus_bg_adu, obj_minus_bg_adu_s,
                  clipped_fraction_obj, clipped_fraction_bg,
                  parse_warnings, db_written_utc,
                  usable, reason
                ) VALUES (
                  ?,
                  ?, ?, ?,
                  ?,
                  ?, ?, ?, ?,
                  ?, ?, ?, ?,
                  ?, ?,
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
                    fields["roi_kind"],
                    fields["roi_fraction"],
                    fields["roi_bbox_json"],
                    fields["bg_bbox_json"],
                    fields["obj_median_adu"],
                    fields["obj_madstd_adu"],
                    fields["bg_median_adu"],
                    fields["bg_madstd_adu"],
                    fields["obj_minus_bg_adu"],
                    fields["obj_minus_bg_adu_s"],
                    fields["clipped_fraction_obj"],
                    fields["clipped_fraction_bg"],
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
