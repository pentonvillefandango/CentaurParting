from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits  # type: ignore

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent

MODULE_NAME = "psf_model_worker"


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _pct(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    return _safe_float(np.percentile(np.asarray(vals, dtype=np.float64), p))


def _get_image_id(db: Database, file_path: Path) -> Optional[int]:
    row = db.execute(
        "SELECT image_id FROM images WHERE file_path = ?",
        (str(file_path),),
    ).fetchone()
    return int(row["image_id"]) if row else None


def _fetch_psf1_summary(db: Database, image_id: int) -> Tuple[int, int]:
    row = db.execute(
        "SELECT n_measured, usable FROM psf_basic_metrics WHERE image_id = ?",
        (image_id,),
    ).fetchone()
    if not row:
        return 0, 0
    return int(row["n_measured"] or 0), int(row["usable"] or 0)


def _fetch_psf1_star_xy(db: Database, image_id: int) -> List[Tuple[int, int]]:
    """
    Returns list of (x,y) integer positions from psf_basic_metrics.star_xy_json.
    If missing/empty/unparseable, returns [].
    """
    row = db.execute(
        "SELECT star_xy_json FROM psf_basic_metrics WHERE image_id = ?",
        (image_id,),
    ).fetchone()
    if not row:
        return []

    s = row["star_xy_json"]
    if s is None:
        return []
    if not isinstance(s, str) or not s.strip():
        return []

    try:
        data = json.loads(s)
        out: List[Tuple[int, int]] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                x = item.get("x", None)
                y = item.get("y", None)
                if x is None or y is None:
                    continue
                try:
                    xi = int(x)
                    yi = int(y)
                except Exception:
                    continue
                out.append((xi, yi))
        return out
    except Exception:
        return []


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


def _moment_gaussian_equiv(data: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Gaussian-equivalent sigma_x, sigma_y using second moments.
    Assumes background-subtracted-ish; uses positive flux only.
    """
    data_pos = np.clip(data, 0.0, None)
    flux = float(np.sum(data_pos))
    if not np.isfinite(flux) or flux <= 0:
        return None

    yy, xx = np.mgrid[:data_pos.shape[0], :data_pos.shape[1]]
    x_c = float(np.sum(xx * data_pos) / flux)
    y_c = float(np.sum(yy * data_pos) / flux)

    dx = xx - x_c
    dy = yy - y_c

    var_x = float(np.sum((dx * dx) * data_pos) / flux)
    var_y = float(np.sum((dy * dy) * data_pos) / flux)

    if not (np.isfinite(var_x) and np.isfinite(var_y)) or var_x <= 0 or var_y <= 0:
        return None

    return math.sqrt(var_x), math.sqrt(var_y)


def process_file_event(cfg: AppConfig, logger: Logger, event: FileReadyEvent) -> Optional[bool]:
    """
    PSF-2: model summary from PSF-1 star positions.
    """
    t0 = time.monotonic()
    started_utc = utc_now()

    # Config
    max_stars_cfg = int(getattr(cfg, "psf2_max_stars", 1000))  # 0 = unlimited
    fit_radius = int(getattr(cfg, "psf2_fit_radius_px", 8))
    edge_margin = int(getattr(cfg, "psf2_edge_margin_px", 16))
    min_good_fits = int(getattr(cfg, "psf2_min_good_fits", 50))

    # Interpret max_stars=0 as unlimited
    max_stars = 0 if max_stars_cfg <= 0 else max_stars_cfg

    try:
        # Resolve image_id and PSF-1 usability + star list
        with Database().transaction() as db:
            image_id = _get_image_id(db, event.file_path)
            if image_id is None:
                raise RuntimeError("image_id_not_found")

            n_input, psf1_usable = _fetch_psf1_summary(db, image_id)
            star_xy = _fetch_psf1_star_xy(db, image_id)

        # Gate on PSF-1
        expected_fields = 10
        if psf1_usable != 1 or n_input < 25 or not star_xy:
            # If PSF-1 didn't give us a usable list, skip cleanly
            fields = {
                "n_input_stars": n_input,
                "n_modeled": 0,
                "gauss_fwhm_px_median": None,
                "gauss_fwhm_px_iqr": None,
                "gauss_fwhm_px_p90": None,
                "gauss_ecc_median": None,
                "gauss_residual_median": None,
                "moffat_fwhm_px_median": None,
                "moffat_beta_median": None,
                "moffat_residual_median": None,
            }

            with Database().transaction() as db:
                db.execute(
                    """
                    INSERT OR REPLACE INTO psf_model_metrics (
                      image_id,
                      expected_fields, read_fields, written_fields,
                      db_written_utc,
                      n_input_stars, n_modeled,
                      gauss_fwhm_px_median, gauss_fwhm_px_iqr, gauss_fwhm_px_p90,
                      gauss_ecc_median, gauss_residual_median,
                      moffat_fwhm_px_median, moffat_beta_median,
                      moffat_residual_median,
                      usable, reason
                    ) VALUES (
                      ?, ?, ?, ?,
                      ?,
                      ?, ?,
                      ?, ?, ?,
                      ?, ?,
                      ?, ?,
                      ?,
                      ?, ?
                    )
                    """,
                    (
                        image_id,
                        expected_fields, expected_fields, 2,
                        utc_now(),
                        n_input, 0,
                        None, None, None,
                        None, None,
                        None, None,
                        None,
                        0, "psf1_not_usable_or_no_star_list",
                    ),
                )

                _insert_module_run(
                    db,
                    image_id,
                    expected_read=expected_fields,
                    read=expected_fields,
                    written=2,
                    status="ok",
                    message="skipped_psf2",
                    started_utc=started_utc,
                    ended_utc=utc_now(),
                    duration_ms=int((time.monotonic() - t0) * 1000),
                )

            logger.log_module_result(
                module=MODULE_NAME,
                file=str(event.file_path),
                expected_read=expected_fields,
                read=expected_fields,
                expected_written=expected_fields,
                written=2,
                status="OK",
                duration_s=time.monotonic() - t0,
                verbose_fields=fields if cfg.logging.is_verbose(MODULE_NAME) else None,
            )
            return None

        # Optionally cap number of stars (but still obey PSF-1 selection)
        if max_stars > 0 and len(star_xy) > max_stars:
            star_xy = star_xy[:max_stars]

        # Load image
        with fits.open(str(event.file_path), memmap=False) as hdul:
            img = np.asarray(hdul[0].data, dtype=np.float64)

        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        if img.ndim != 2:
            raise ValueError("unsupported_fits_data_shape")

        h, w = img.shape

        gauss_fwhm: List[float] = []
        gauss_ecc: List[float] = []

        n_fit_attempted = 0
        n_gauss_ok = 0

        for (x, y) in star_xy:
            # Edge margin (extra safety)
            if x < edge_margin or y < edge_margin or x >= (w - edge_margin) or y >= (h - edge_margin):
                continue

            y0, y1 = y - fit_radius, y + fit_radius + 1
            x0, x1 = x - fit_radius, x + fit_radius + 1
            if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
                continue

            stamp = img[y0:y1, x0:x1]
            if stamp.size < 25 or not np.isfinite(stamp).any():
                continue

            # subtract local background
            local_bg = float(np.nanmedian(stamp))
            data = stamp - local_bg
            if not np.isfinite(data).all():
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            if float(np.nanmax(data)) <= 0:
                continue

            n_fit_attempted += 1

            m = _moment_gaussian_equiv(data)
            if m is None:
                continue

            sx, sy = m
            if sx <= 0 or sy <= 0:
                continue

            fwhm_g = 2.355 * 0.5 * (sx + sy)
            ecc = 1.0 - (min(sx, sy) / max(sx, sy))

            if not (np.isfinite(fwhm_g) and np.isfinite(ecc)):
                continue

            gauss_fwhm.append(float(fwhm_g))
            gauss_ecc.append(float(ecc))
            n_gauss_ok += 1

        n_modeled = int(len(gauss_fwhm))

        gauss_fwhm_med = _pct(gauss_fwhm, 50)
        gauss_fwhm_iqr = (
            _safe_float(np.percentile(gauss_fwhm, 75) - np.percentile(gauss_fwhm, 25))
            if gauss_fwhm
            else None
        )
        gauss_fwhm_p90 = _pct(gauss_fwhm, 90)
        gauss_ecc_med = _pct(gauss_ecc, 50)

        usable = int(n_modeled >= min_good_fits)
        reason = "ok" if usable else "too_few_fits"

        fields: Dict[str, Any] = {
            "n_input_stars": n_input,
            "n_modeled": n_modeled,
            "gauss_fwhm_px_median": gauss_fwhm_med,
            "gauss_fwhm_px_iqr": gauss_fwhm_iqr,
            "gauss_fwhm_px_p90": gauss_fwhm_p90,
            "gauss_ecc_median": gauss_ecc_med,
            "gauss_residual_median": None,
            "moffat_fwhm_px_median": None,
            "moffat_beta_median": None,
            "moffat_residual_median": None,
            # logged-only debug
            "n_fit_attempted": n_fit_attempted,
            "n_gauss_ok": n_gauss_ok,
            "psf2_max_stars_used": max_stars,
            "psf2_min_good_fits": min_good_fits,
        }

        written_fields = 2  # n_input_stars, n_modeled
        for k in ("gauss_fwhm_px_median", "gauss_fwhm_px_iqr", "gauss_fwhm_px_p90", "gauss_ecc_median"):
            if fields[k] is not None:
                written_fields += 1

        with Database().transaction() as db:
            db.execute(
                """
                INSERT OR REPLACE INTO psf_model_metrics (
                  image_id,
                  expected_fields, read_fields, written_fields,
                  db_written_utc,
                  n_input_stars, n_modeled,
                  gauss_fwhm_px_median, gauss_fwhm_px_iqr, gauss_fwhm_px_p90,
                  gauss_ecc_median, gauss_residual_median,
                  moffat_fwhm_px_median, moffat_beta_median,
                  moffat_residual_median,
                  usable, reason
                ) VALUES (
                  ?, ?, ?, ?,
                  ?,
                  ?, ?,
                  ?, ?, ?,
                  ?, ?,
                  ?, ?,
                  ?,
                  ?, ?
                )
                """,
                (
                    image_id,
                    expected_fields, expected_fields, written_fields,
                    utc_now(),
                    n_input, n_modeled,
                    gauss_fwhm_med, gauss_fwhm_iqr, gauss_fwhm_p90,
                    gauss_ecc_med, None,
                    None, None,
                    None,
                    usable, reason,
                ),
            )

            _insert_module_run(
                db,
                image_id,
                expected_read=expected_fields,
                read=expected_fields,
                written=written_fields,
                status="ok",
                message=None,
                started_utc=started_utc,
                ended_utc=utc_now(),
                duration_ms=int((time.monotonic() - t0) * 1000),
            )

        logger.log_module_result(
            module=MODULE_NAME,
            file=str(event.file_path),
            expected_read=expected_fields,
            read=expected_fields,
            expected_written=expected_fields,
            written=written_fields,
            status="OK",
            duration_s=time.monotonic() - t0,
            verbose_fields=fields if cfg.logging.is_verbose(MODULE_NAME) else None,
        )
        return None

    except Exception as e:
        logger.log_failure(
            module=MODULE_NAME,
            file=str(event.file_path),
            action=cfg.on_metric_failure,
            reason=f"{type(e).__name__}:{e}",
            duration_s=time.monotonic() - t0,
        )
        return False
