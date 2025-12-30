from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent
from centaur.pipeline import ModuleRunRecord, PipelineContext

# V1b: shared pixels loader (FITS + XISF)
from centaur.io.frame_loader import load_pixels

MODULE_NAME = "star_headroom_worker"


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


def _flatten_image_data(data: Any) -> np.ndarray:
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


def _get_header_core(db: Database, image_id: int) -> Dict[str, Any]:
    row = db.execute(
        """
        SELECT exptime, datamax, bitpix, bzero
        FROM fits_header_core
        WHERE image_id=?
        """,
        (image_id,),
    ).fetchone()
    return dict(row) if row else {}


def _get_star_xy_json(db: Database, image_id: int) -> Optional[str]:
    row = db.execute(
        "SELECT star_xy_json FROM psf_basic_metrics WHERE image_id=?",
        (image_id,),
    ).fetchone()
    return str(row["star_xy_json"]) if row and row["star_xy_json"] else None


def _get_saturation_metrics(
    db: Database, image_id: int
) -> Tuple[Optional[float], Optional[float]]:
    row = db.execute(
        "SELECT saturation_adu, max_pixel_adu FROM saturation_metrics WHERE image_id=?",
        (image_id,),
    ).fetchone()
    if not row:
        return None, None
    return _safe_float(row["saturation_adu"]), _safe_float(row["max_pixel_adu"])


def _bitpix_vmax(core: Dict[str, Any]) -> Optional[float]:
    bitpix = core.get("bitpix")
    try:
        bp = int(bitpix) if bitpix is not None else 0
        bp = int(abs(bp))
        if bp <= 0 or bp > 32:
            return None
        return float((2**bp) - 1)
    except Exception:
        return None


def _derive_saturation_adu(
    core: Dict[str, Any],
    sat_adu: Optional[float],
    max_pixel_adu: Optional[float],
) -> Tuple[Optional[float], str]:
    """
    Choose the saturation ceiling we should use for *star headroom*.

    Priority:
      1) header datamax (if present and plausible)
      2) saturation_metrics.saturation_adu (if present and plausible)
      3) max_pixel_adu IF it's near the representable top end (common real-world "effective clip")
      4) bitpix-derived vmax fallback
      5) raw max_pixel_adu fallback
    """
    datamax = _safe_float(core.get("datamax"))
    if datamax is not None and datamax > 0:
        return datamax, "header_datamax"

    if sat_adu is not None and sat_adu > 0:
        return sat_adu, "saturation_metrics"

    vmax = _bitpix_vmax(core)
    if (
        vmax is not None
        and vmax > 0
        and max_pixel_adu is not None
        and max_pixel_adu > 0
    ):
        if max_pixel_adu >= 0.98 * vmax:
            return max_pixel_adu, "max_pixel_near_fullscale"

    if vmax is not None and vmax > 0:
        return vmax, "bitpix_vmax"

    if max_pixel_adu is not None and max_pixel_adu > 0:
        return max_pixel_adu, "fallback_max_pixel"

    return None, "unknown"


def _extract_star_peaks(
    arr: np.ndarray, stars: List[Tuple[int, int]], r: int, peak_q: float
) -> np.ndarray:
    h, w = arr.shape
    peaks: List[float] = []
    rr = int(max(1, r))
    q = float(peak_q)

    for x, y in stars:
        x = int(x)
        y = int(y)
        x0 = max(0, x - rr)
        x1 = min(w, x + rr + 1)
        y0 = max(0, y - rr)
        y1 = min(h, y + rr + 1)

        cut = arr[y0:y1, x0:x1].reshape(-1)
        cut = cut[np.isfinite(cut)]
        if cut.size < 4:
            continue

        p = _safe_float(np.percentile(cut, q))
        if p is not None:
            peaks.append(float(p))

    return np.asarray(peaks, dtype=np.float64)


def process_file_event(
    cfg: AppConfig, logger: Logger, event: FileReadyEvent, ctx: PipelineContext
) -> Any:
    t0 = time.monotonic()

    expected_fields = 0
    read_fields = 0
    written_fields = 0
    image_id: Optional[int] = None
    fields: Dict[str, Any] = {}

    max_stars = int(getattr(cfg, "star_headroom_max_stars", 200))
    sample_r = int(getattr(cfg, "star_headroom_sample_radius_px", 2))
    peak_q = float(getattr(cfg, "star_headroom_peak_percentile", 99.5))

    sat_eps_adu = float(getattr(cfg, "star_headroom_sat_eps_adu", 0.5))

    try:
        # V1b: load pixels via shared loader (FITS + XISF)
        arr2d = _flatten_image_data(load_pixels(event.file_path))
        if arr2d.size == 0:
            raise ValueError("unsupported_image_data_shape")

        with Database().transaction() as db:
            image_id = _get_image_id(db, event.file_path)
            if image_id is None:
                raise RuntimeError("image_id_not_found_for_file_path")

            core = _get_header_core(db, int(image_id))
            exptime_s = _safe_float(core.get("exptime"))

            sat_adu, max_pixel_adu = _get_saturation_metrics(db, int(image_id))
            saturation_adu_used, saturation_source = _derive_saturation_adu(
                core, sat_adu, max_pixel_adu
            )

            # Get stars: ctx first, fallback to psf_basic_metrics.star_xy_json
            stars_xy: List[Tuple[int, int]] = []
            if ctx.psf1 and isinstance(ctx.psf1.get("star_xy"), list):
                try:
                    stars_xy = [
                        (int(p["x"]), int(p["y"]))
                        for p in ctx.psf1["star_xy"]
                        if isinstance(p, dict) and "x" in p and "y" in p
                    ]
                except Exception:
                    stars_xy = []

            if not stars_xy:
                star_xy_json = _get_star_xy_json(db, int(image_id))
                if star_xy_json:
                    pts = json.loads(star_xy_json)
                    stars_xy = [
                        (int(p["x"]), int(p["y"])) for p in pts if "x" in p and "y" in p
                    ]

            if not stars_xy:
                fields = {
                    "exptime_s": exptime_s,
                    "n_stars_used": 0,
                    "saturation_adu_used": saturation_adu_used,
                    "saturation_source": saturation_source,
                    "usable": 0,
                    "reason": "no_stars_available",
                }
            else:
                if max_stars > 0 and len(stars_xy) > max_stars:
                    stars_xy = stars_xy[:max_stars]

                peaks = _extract_star_peaks(arr2d, stars_xy, r=sample_r, peak_q=peak_q)

                if peaks.size == 0:
                    fields = {
                        "exptime_s": exptime_s,
                        "n_stars_used": 0,
                        "saturation_adu_used": saturation_adu_used,
                        "saturation_source": saturation_source,
                        "usable": 0,
                        "reason": "no_valid_star_peaks",
                    }
                else:
                    p50 = _safe_float(np.percentile(peaks, 50))
                    p90 = _safe_float(np.percentile(peaks, 90))
                    p99 = _safe_float(np.percentile(peaks, 99))

                    def headroom(p: Optional[float]) -> Optional[float]:
                        if (
                            p is None
                            or saturation_adu_used is None
                            or saturation_adu_used <= 0
                        ):
                            return None
                        return _safe_float(
                            max(0.0, min(1.0, 1.0 - (p / saturation_adu_used)))
                        )

                    h50 = headroom(p50)
                    h90 = headroom(p90)
                    h99 = headroom(p99)

                    sat_star_frac = None
                    if saturation_adu_used is not None and saturation_adu_used > 0:
                        sat_star_frac = _safe_float(
                            float(np.mean(peaks >= (saturation_adu_used - sat_eps_adu)))
                        )

                    fields = {
                        "exptime_s": exptime_s,
                        "n_stars_used": int(peaks.size),
                        "saturation_adu_used": saturation_adu_used,
                        "saturation_source": saturation_source,
                        "star_peak_p50_adu": p50,
                        "star_peak_p90_adu": p90,
                        "star_peak_p99_adu": p99,
                        "headroom_p50": h50,
                        "headroom_p90": h90,
                        "headroom_p99": h99,
                        "saturated_star_fraction": sat_star_frac,
                        "usable": 1,
                        "reason": None,
                    }

            expected_fields = len(fields)
            read_fields = expected_fields
            written_fields = sum(1 for v in fields.values() if v is not None)

            db.execute(
                """
                INSERT OR REPLACE INTO star_headroom_metrics (
                  image_id,
                  expected_fields, read_fields, written_fields,
                  exptime_s, parse_warnings, db_written_utc,

                  n_stars_used,

                  saturation_adu_used, saturation_source,

                  star_peak_p50_adu, star_peak_p90_adu, star_peak_p99_adu,
                  headroom_p50, headroom_p90, headroom_p99,
                  saturated_star_fraction,

                  usable, reason
                ) VALUES (
                  ?,
                  ?, ?, ?,
                  ?, ?, ?,
                  ?,
                  ?, ?,
                  ?, ?, ?,
                  ?, ?, ?,
                  ?,
                  ?, ?
                )
                """,
                (
                    int(image_id),
                    int(expected_fields),
                    int(read_fields),
                    int(written_fields),
                    fields.get("exptime_s"),
                    None,
                    utc_now(),
                    fields.get("n_stars_used"),
                    fields.get("saturation_adu_used"),
                    fields.get("saturation_source"),
                    fields.get("star_peak_p50_adu"),
                    fields.get("star_peak_p90_adu"),
                    fields.get("star_peak_p99_adu"),
                    fields.get("headroom_p50"),
                    fields.get("headroom_p90"),
                    fields.get("headroom_p99"),
                    fields.get("saturated_star_fraction"),
                    fields.get("usable"),
                    fields.get("reason"),
                ),
            )

        logger.log_module_result(
            module=MODULE_NAME,
            file=str(event.file_path),
            expected_read=expected_fields,
            read=read_fields,
            expected_written=expected_fields,
            written=written_fields,
            status="OK",
            duration_s=time.monotonic() - t0,
            verbose_fields=fields,
        )

        return ModuleRunRecord(
            image_id=int(image_id),
            expected_read=int(expected_fields),
            read=int(read_fields),
            expected_written=int(expected_fields),
            written=int(written_fields),
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
                image_id=int(image_id),
                expected_read=int(expected_fields or 0),
                read=int(read_fields or 0),
                expected_written=int(expected_fields or 0),
                written=int(written_fields or 0),
                status="FAILED",
                message=f"{type(e).__name__}:{e}",
            )
        return False
