
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent
from centaur.pipeline import ModuleRunRecord

MODULE_NAME = "signal_structure_worker"


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


def _get_image_id(db: Database, file_path: Path) -> Optional[int]:
    row = db.execute(
        "SELECT image_id FROM images WHERE file_path = ?",
        (str(file_path),),
    ).fetchone()
    return int(row["image_id"]) if row else None


def process_file_event(cfg: AppConfig, logger: Logger, event: FileReadyEvent) -> Any:
    """
    Model A: structure-based efficiency metrics for full-frame nebula/small sensors.

    Reads inputs from existing metric tables:
      - sky_basic_metrics: percentiles + ff_madstd_adu_s
      - sky_background2d_metrics: plane_slope_mag_adu_per_tile_s
      - saturation_metrics: saturated_pixel_fraction (optional)
      - psf_basic_metrics: fwhm/ecc/usable (optional)

    Writes one row per image into signal_structure_metrics.
    Pipeline owns module_runs.
    """
    t0 = time.monotonic()

    expected_fields = 0
    read_fields = 0
    written_fields = 0
    image_id: Optional[int] = None
    fields: Dict[str, Any] = {}

    try:
        with Database().transaction() as db:
            image_id = _get_image_id(db, event.file_path)
            if image_id is None:
                raise RuntimeError("image_id_not_found_for_file_path")

            # --- sky_basic inputs (required for this model) ---
            sb = db.execute(
                """
                SELECT
                  exptime_s,
                  ff_p10_adu, ff_p50_adu, ff_p90_adu, ff_p99_adu, ff_p999_adu,
                  ff_madstd_adu_s
                FROM sky_basic_metrics
                WHERE image_id = ?
                """,
                (image_id,),
            ).fetchone()
            if not sb:
                raise RuntimeError("missing_sky_basic_metrics")

            exptime_s = _safe_float(sb["exptime_s"])
            ff_p10 = _safe_float(sb["ff_p10_adu"])
            ff_p50 = _safe_float(sb["ff_p50_adu"])
            ff_p90 = _safe_float(sb["ff_p90_adu"])
            ff_p99 = _safe_float(sb["ff_p99_adu"])
            ff_p999 = _safe_float(sb["ff_p999_adu"])
            ff_madstd_s = _safe_float(sb["ff_madstd_adu_s"])

            ff_p90m_p10 = _safe_float(ff_p90 - ff_p10) if (ff_p90 is not None and ff_p10 is not None) else None
            ff_p99m_p50 = _safe_float(ff_p99 - ff_p50) if (ff_p99 is not None and ff_p50 is not None) else None
            ff_p999m_p50 = _safe_float(ff_p999 - ff_p50) if (ff_p999 is not None and ff_p50 is not None) else None

            if exptime_s is not None and exptime_s > 0:
                ff_p90m_p10_s = _safe_float(ff_p90m_p10 / exptime_s) if ff_p90m_p10 is not None else None
                ff_p99m_p50_s = _safe_float(ff_p99m_p50 / exptime_s) if ff_p99m_p50 is not None else None
                ff_p999m_p50_s = _safe_float(ff_p999m_p50 / exptime_s) if ff_p999m_p50 is not None else None
            else:
                ff_p90m_p10_s = None
                ff_p99m_p50_s = None
                ff_p999m_p50_s = None

            # --- sky_background2d inputs (optional but strongly preferred) ---
            b2 = db.execute(
                """
                SELECT plane_slope_mag_adu_per_tile_s
                FROM sky_background2d_metrics
                WHERE image_id = ?
                """,
                (image_id,),
            ).fetchone()
            grad_s = _safe_float(b2["plane_slope_mag_adu_per_tile_s"]) if b2 else None

            # --- saturation inputs (optional) ---
            sat = db.execute(
                """
                SELECT saturated_pixel_fraction
                FROM saturation_metrics
                WHERE image_id = ?
                """,
                (image_id,),
            ).fetchone()
            sat_frac = _safe_float(sat["saturated_pixel_fraction"]) if sat else None

            # --- PSF inputs (optional) ---
            pb = db.execute(
                """
                SELECT fwhm_px_median, ecc_median, usable
                FROM psf_basic_metrics
                WHERE image_id = ?
                """,
                (image_id,),
            ).fetchone()
            psf_fwhm = _safe_float(pb["fwhm_px_median"]) if pb else None
            psf_ecc = _safe_float(pb["ecc_median"]) if pb else None
            psf_usable = int(pb["usable"]) if (pb and pb["usable"] is not None) else None

            # --- derived: efficiency + time weight ---
            # Choose p99-p50 as the primary structure term (as per your validated SQL)
            structure = ff_p99m_p50_s  # ADU/s
            noise = ff_madstd_s        # ADU/s
            grad = grad_s              # ADU/tile/s (proxy)

            eps = 1e-12
            eff_score: Optional[float] = None
            time_weight: Optional[float] = None

            if structure is not None and noise is not None:
                denom = (noise * noise) + ((grad * grad) if grad is not None else 0.0) + eps
                eff_score = _safe_float((structure * structure) / denom)
                time_weight = _safe_float(1.0 / (eff_score + eps)) if eff_score is not None else None

            fields = {
                "exptime_s": exptime_s,

                "ff_p90_minus_p10_adu": ff_p90m_p10,
                "ff_p99_minus_p50_adu": ff_p99m_p50,
                "ff_p999_minus_p50_adu": ff_p999m_p50,

                "ff_p90_minus_p10_adu_s": ff_p90m_p10_s,
                "ff_p99_minus_p50_adu_s": ff_p99m_p50_s,
                "ff_p999_minus_p50_adu_s": ff_p999m_p50_s,

                "ff_madstd_adu_s": ff_madstd_s,
                "plane_slope_mag_adu_per_tile_s": grad_s,

                "saturated_pixel_fraction": sat_frac,
                "psf_fwhm_px_median": psf_fwhm,
                "psf_ecc_median": psf_ecc,
                "psf_usable": psf_usable,

                "eff_score": eff_score,
                "time_weight": time_weight,
            }

            expected_fields = len(fields)
            read_fields = expected_fields
            written_fields = sum(1 for v in fields.values() if v is not None)

            db.execute(
                """
                INSERT OR REPLACE INTO signal_structure_metrics (
                  image_id,
                  expected_fields, read_fields, written_fields,
                  parse_warnings, db_written_utc,

                  exptime_s,

                  ff_p90_minus_p10_adu, ff_p99_minus_p50_adu, ff_p999_minus_p50_adu,
                  ff_p90_minus_p10_adu_s, ff_p99_minus_p50_adu_s, ff_p999_minus_p50_adu_s,

                  ff_madstd_adu_s,
                  plane_slope_mag_adu_per_tile_s,

                  saturated_pixel_fraction,
                  psf_fwhm_px_median,
                  psf_ecc_median,
                  psf_usable,

                  eff_score,
                  time_weight
                ) VALUES (
                  ?,
                  ?, ?, ?,
                  ?, ?,
                  ?,
                  ?, ?, ?,
                  ?, ?, ?,
                  ?,
                  ?,
                  ?,
                  ?, ?, ?,
                  ?, ?
                )
                """,
                (
                    int(image_id),
                    int(expected_fields), int(read_fields), int(written_fields),
                    None, utc_now(),

                    fields["exptime_s"],

                    fields["ff_p90_minus_p10_adu"], fields["ff_p99_minus_p50_adu"], fields["ff_p999_minus_p50_adu"],
                    fields["ff_p90_minus_p10_adu_s"], fields["ff_p99_minus_p50_adu_s"], fields["ff_p999_minus_p50_adu_s"],

                    fields["ff_madstd_adu_s"],
                    fields["plane_slope_mag_adu_per_tile_s"],

                    fields["saturated_pixel_fraction"],
                    fields["psf_fwhm_px_median"],
                    fields["psf_ecc_median"],
                    fields["psf_usable"],

                    fields["eff_score"],
                    fields["time_weight"],
                ),
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
