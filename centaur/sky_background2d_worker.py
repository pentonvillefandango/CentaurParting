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

MODULE_NAME = "sky_background2d_worker"


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


def _get_exptime_s(db: Database, image_id: int) -> Optional[float]:
    row = db.execute(
        "SELECT exptime FROM fits_header_core WHERE image_id = ?",
        (image_id,),
    ).fetchone()
    if not row:
        return None
    return _safe_float(row["exptime"])


def _insert_module_run(
    db: Database,
    image_id: int,
    *,
    expected_read: int,
    read: int,
    written: int,
    status: str,
    message: Optional[str],
    started_utc: str,
    ended_utc: str,
    duration_ms: int,
) -> None:
    db.execute(
        """
        INSERT INTO module_runs
        (image_id, module_name, expected_fields, read_fields, written_fields,
         status, message, started_utc, ended_utc, duration_ms, db_written_utc)
        VALUES (?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?)
        """,
        (
            image_id,
            MODULE_NAME,
            expected_read,
            read,
            written,
            status,
            message,
            started_utc,
            ended_utc,
            duration_ms,
            utc_now(),
        ),
    )


@dataclass(frozen=True)
class TileResult:
    median: Optional[float]
    clipped_fraction: Optional[float]


def _tile_sigma_clipped_median(tile: np.ndarray) -> TileResult:
    """
    Robust per-tile background estimate:
    - remove non-finite
    - sigma-clip (reject stars)
    - take median of kept pixels
    """
    v = tile.reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return TileResult(median=None, clipped_fraction=None)

    clipped = sigma_clip(v, sigma=3.0, maxiters=5, masked=True)
    mask = np.asarray(clipped.mask)
    kept = v[~mask] if mask.size else v

    if kept.size == 0:
        med = None
    else:
        med = _safe_float(np.median(kept))

    frac = _safe_float(float(mask.sum()) / float(mask.size)) if mask.size else 0.0
    return TileResult(median=med, clipped_fraction=frac)


def _build_background_map(arr2d: np.ndarray, tile_size: int) -> Tuple[np.ndarray, float]:
    """
    Returns:
      bkg_map: (ny, nx) float array with NaNs where tiles failed
      clipped_fraction_mean: mean clipped fraction across tiles (finite only)
    """
    h, w = arr2d.shape
    tile_size = int(max(16, tile_size))

    nx = w // tile_size
    ny = h // tile_size
    if nx < 2 or ny < 2:
        raise ValueError("image_too_small_for_tile_grid")

    bkg = np.full((ny, nx), np.nan, dtype=np.float64)
    clip_fracs: list[float] = []

    for ty in range(ny):
        for tx in range(nx):
            y0 = ty * tile_size
            x0 = tx * tile_size
            tile = arr2d[y0 : y0 + tile_size, x0 : x0 + tile_size]
            r = _tile_sigma_clipped_median(tile)
            if r.median is not None:
                bkg[ty, tx] = float(r.median)
            if r.clipped_fraction is not None:
                clip_fracs.append(float(r.clipped_fraction))

    clipped_fraction_mean = float(np.mean(clip_fracs)) if clip_fracs else 0.0
    return bkg, clipped_fraction_mean


def _nanpercentile(x: np.ndarray, q: float) -> Optional[float]:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return None
    return _safe_float(np.percentile(finite, q))


def _plane_fit_slopes(bkg: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Fit z = a*x + b*y + c over tile indices, using finite tiles only.
    Returns slopes a (x), b (y), and magnitude sqrt(a^2+b^2).
    """
    ny, nx = bkg.shape
    ys, xs = np.mgrid[0:ny, 0:nx]
    z = bkg

    m = np.isfinite(z)
    if int(m.sum()) < 6:
        return None, None, None

    X = np.column_stack([xs[m].ravel(), ys[m].ravel(), np.ones(int(m.sum()))])
    y = z[m].ravel()

    # least squares
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = _safe_float(coef[0])
    b = _safe_float(coef[1])
    if a is None or b is None:
        return a, b, None
    mag = _safe_float(np.sqrt(a * a + b * b))
    return a, b, mag


def _gradient_stats(bkg: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Gradient magnitude computed on tile grid.
    """
    finite = np.isfinite(bkg)
    if int(finite.sum()) < 6:
        return None, None

    # Fill NaNs with median for gradient computation (so gaps don't explode)
    med = np.nanmedian(bkg)
    filled = np.where(np.isfinite(bkg), bkg, med)

    gy, gx = np.gradient(filled)  # per tile index
    gmag = np.sqrt(gx * gx + gy * gy)

    gmag_f = gmag[finite]
    if gmag_f.size == 0:
        return None, None

    gmean = _safe_float(np.mean(gmag_f))
    gp95 = _safe_float(np.percentile(gmag_f, 95))
    return gmean, gp95


def _corner_delta(bkg: np.ndarray) -> Optional[float]:
    ny, nx = bkg.shape
    corners = np.array(
        [bkg[0, 0], bkg[0, nx - 1], bkg[ny - 1, 0], bkg[ny - 1, nx - 1]],
        dtype=np.float64,
    )
    corners = corners[np.isfinite(corners)]
    if corners.size == 0:
        return None
    return _safe_float(float(np.max(corners) - np.min(corners)))


def process_file_event(cfg: AppConfig, logger: Logger, event: FileReadyEvent) -> Optional[bool]:
    """
    Compute a tiled 2D background map and gradient metrics.

    Returns:
      - None on success (pipeline treats None as OK)
      - False on failure
    """
    t0 = time.monotonic()
    started_utc = utc_now()

    # Sensible default for large astro frames; we can make this configurable later.
    tile_size_px = 64

    expected_fields = 0

    try:
        with fits.open(event.file_path, memmap=False) as hdul:
            data = hdul[0].data

        arr2d = _flatten_image_data(data)
        if arr2d.size == 0:
            raise ValueError("unsupported_fits_data_shape")

        bkg, clipped_fraction_mean = _build_background_map(arr2d, tile_size=tile_size_px)
        ny, nx = bkg.shape

        # Summary stats of the map
        bkg2d_median = _safe_float(np.nanmedian(bkg))
        bkg2d_min = _safe_float(np.nanmin(bkg)) if np.isfinite(bkg).any() else None
        bkg2d_max = _safe_float(np.nanmax(bkg)) if np.isfinite(bkg).any() else None
        bkg2d_range = _safe_float(bkg2d_max - bkg2d_min) if (bkg2d_min is not None and bkg2d_max is not None) else None
        p5 = _nanpercentile(bkg, 5)
        p95 = _nanpercentile(bkg, 95)
        p95_minus_p5 = _safe_float(p95 - p5) if (p5 is not None and p95 is not None) else None

        # RMS of map (spatial variation)
        finite_vals = bkg[np.isfinite(bkg)]
        bkg2d_rms = _safe_float(np.std(finite_vals)) if finite_vals.size else None

        # Plane fit slopes
        slope_x, slope_y, slope_mag = _plane_fit_slopes(bkg)

        # Gradient stats
        grad_mean, grad_p95 = _gradient_stats(bkg)

        # Corner delta
        corner_delta = _corner_delta(bkg)

        with Database().transaction() as db:
            image_id = _get_image_id(db, event.file_path)
            if image_id is None:
                raise RuntimeError("image_id_not_found_for_file_path")

            exptime_s = _get_exptime_s(db, image_id)

            # Per-second variants (only where meaningful)
            if exptime_s is not None and exptime_s > 0:
                bkg2d_median_s = _safe_float(bkg2d_median / exptime_s) if bkg2d_median is not None else None
                slope_mag_s = _safe_float(slope_mag / exptime_s) if slope_mag is not None else None
            else:
                bkg2d_median_s = None
                slope_mag_s = None

            fields: Dict[str, Any] = {
                "exptime_s": exptime_s,
                "tile_size_px": tile_size_px,
                "grid_nx": nx,
                "grid_ny": ny,
                "clipped_fraction_mean": clipped_fraction_mean,
                "bkg2d_median_adu": bkg2d_median,
                "bkg2d_min_adu": bkg2d_min,
                "bkg2d_max_adu": bkg2d_max,
                "bkg2d_range_adu": bkg2d_range,
                "bkg2d_p95_minus_p5_adu": p95_minus_p5,
                "bkg2d_rms_of_map_adu": bkg2d_rms,
                "plane_slope_x_adu_per_tile": slope_x,
                "plane_slope_y_adu_per_tile": slope_y,
                "plane_slope_mag_adu_per_tile": slope_mag,
                "grad_mean_adu_per_tile": grad_mean,
                "grad_p95_adu_per_tile": grad_p95,
                "corner_delta_adu": corner_delta,
                "bkg2d_median_adu_s": bkg2d_median_s,
                "plane_slope_mag_adu_per_tile_s": slope_mag_s,
            }

            expected_fields = len(fields)
            read_fields = expected_fields
            written_fields = sum(1 for v in fields.values() if v is not None)

            db.execute(
                """
                INSERT OR REPLACE INTO sky_background2d_metrics (
                  image_id,
                  expected_fields, read_fields, written_fields,
                  exptime_s, tile_size_px, grid_nx, grid_ny,
                  clipped_fraction_mean, parse_warnings, db_written_utc,
                  bkg2d_median_adu, bkg2d_min_adu, bkg2d_max_adu, bkg2d_range_adu,
                  bkg2d_p95_minus_p5_adu, bkg2d_rms_of_map_adu,
                  plane_slope_x_adu_per_tile, plane_slope_y_adu_per_tile, plane_slope_mag_adu_per_tile,
                  grad_mean_adu_per_tile, grad_p95_adu_per_tile,
                  corner_delta_adu,
                  bkg2d_median_adu_s, plane_slope_mag_adu_per_tile_s
                ) VALUES (
                  ?,
                  ?, ?, ?,
                  ?, ?, ?, ?,
                  ?, ?, ?,
                  ?, ?, ?, ?,
                  ?, ?,
                  ?, ?, ?,
                  ?, ?,
                  ?,
                  ?, ?
                )
                """,
                (
                    image_id,
                    expected_fields, read_fields, written_fields,
                    exptime_s, tile_size_px, nx, ny,
                    clipped_fraction_mean, None, utc_now(),
                    bkg2d_median, bkg2d_min, bkg2d_max, bkg2d_range,
                    p95_minus_p5, bkg2d_rms,
                    slope_x, slope_y, slope_mag,
                    grad_mean, grad_p95,
                    corner_delta,
                    bkg2d_median_s, slope_mag_s,
                ),
            )

            ended_utc = utc_now()
            duration_ms = int((time.monotonic() - t0) * 1000)
            _insert_module_run(
                db,
                image_id,
                expected_read=expected_fields,
                read=read_fields,
                written=written_fields,
                status="ok",
                message=None,
                started_utc=started_utc,
                ended_utc=ended_utc,
                duration_ms=duration_ms,
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

        return None

    except Exception as e:
        duration_s = time.monotonic() - t0
        logger.log_failure(
            module=MODULE_NAME,
            file=str(event.file_path),
            action=cfg.on_metric_failure,
            reason=f"{type(e).__name__}:{e}",
            duration_s=duration_s,
        )
        return False

