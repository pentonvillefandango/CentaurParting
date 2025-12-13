from __future__ import annotations

import time
from typing import Optional

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent


def _extract_gain_setting(db: Database, image_id: int) -> Optional[int]:
    """
    Extract integer GAIN from fits_header_full.header_json.

    header_json is a JSON array like:
      [{"keyword":"GAIN","value":252,"comment":"..."}, ...]
    """
    row = db.execute(
        """
        SELECT CAST(json_extract(je.value, '$.value') AS INTEGER) AS gain_setting
        FROM fits_header_full ff,
             json_each(ff.header_json) AS je
        WHERE ff.image_id = ?
          AND upper(json_extract(je.value, '$.keyword')) = 'GAIN'
        LIMIT 1;
        """,
        (image_id,),
    ).fetchone()

    if row is None:
        return None

    return row["gain_setting"]


def process_file_event(cfg: AppConfig, logger: Logger, event: FileReadyEvent) -> bool:
    start = time.monotonic()

    try:
        with Database().transaction() as db:
            # Pull prerequisites
            row = db.execute(
                """
                SELECT
                    i.image_id,
                    lower(f.instrume) AS camera_name,
                    f.exptime AS exptime_s,
                    sb.roi_median_adu_s AS sky_rate_adu_s,
                    b.grad_p95_adu_per_tile AS grad_p95
                FROM images i
                JOIN fits_header_core f USING(image_id)
                JOIN sky_basic_metrics sb USING(image_id)
                JOIN sky_background2d_metrics b USING(image_id)
                WHERE i.file_path = ?
                """,
                (str(event.file_path),),
            ).fetchone()

            if row is None:
                logger.log_failure(
                    module="exposure_advice_worker",
                    file=str(event.file_path),
                    action="continue",
                    reason="missing_prerequisites",
                )
                return False

            image_id = int(row["image_id"])
            camera_name = row["camera_name"] or ""
            sky_rate_adu_s = float(row["sky_rate_adu_s"] or 0.0)
            grad_p95 = float(row["grad_p95"] or 0.0)

            # Extract GAIN from full FITS header
            gain_setting = _extract_gain_setting(db, image_id)

            # Look up camera constants using (camera_name, gain_setting)
            const = db.execute(
                """
                SELECT gain_e_per_adu, read_noise_e, is_osc
                FROM camera_constants
                WHERE camera_name = ?
                  AND (
                        (gain_setting IS NULL AND ? IS NULL)
                     OR gain_setting = ?
                  )
                LIMIT 1;
                """,
                (camera_name, gain_setting, gain_setting),
            ).fetchone()

            if const is None:
                db.execute(
                    """
                    INSERT OR REPLACE INTO exposure_advice
                    (image_id, decision_reason)
                    VALUES (?, ?);
                    """,
                    (image_id, f"missing_camera_constants_for_gain:{gain_setting}"),
                )

                logger.log_module_summary(
                    module="exposure_advice_worker",
                    file=str(event.file_path),
                    expected_read=5,
                    read=5,
                    expected_written=1,
                    written=1,
                    status=f"OK (no constants for gain={gain_setting})",
                    duration_s=time.monotonic() - start,
                )
                return True

            gain_e_per_adu = float(const["gain_e_per_adu"])
            read_noise_e = float(const["read_noise_e"])

            # Convert sky rate to electrons/sec
            sky_e_s = sky_rate_adu_s * gain_e_per_adu
            if sky_e_s <= 0:
                db.execute(
                    """
                    INSERT OR REPLACE INTO exposure_advice
                    (image_id, decision_reason)
                    VALUES (?, ?);
                    """,
                    (image_id, "invalid_sky_rate"),
                )

                logger.log_module_summary(
                    module="exposure_advice_worker",
                    file=str(event.file_path),
                    expected_read=5,
                    read=5,
                    expected_written=1,
                    written=1,
                    status="OK (invalid sky rate)",
                    duration_s=time.monotonic() - start,
                )
                return True

            # Sky-limited minimum exposure
            def min_exp(k: float) -> float:
                return ((k * read_noise_e) ** 2) / sky_e_s

            min_k3 = min_exp(3.0)
            min_k5 = min_exp(5.0)

            # Gradient-limited maximum exposure (simple heuristic)
            grad_cap_s = None
            if grad_p95 > 0:
                grad_cap_s = max(30.0, 300.0 * (20.0 / grad_p95))

            rec_min = max(min_k3, 10.0)
            rec_max = grad_cap_s if grad_cap_s is not None else rec_min * 4.0

            db.execute(
                """
                INSERT OR REPLACE INTO exposure_advice
                (image_id,
                 sky_limited_min_s_k3,
                 sky_limited_min_s_k5,
                 gradient_limited_max_s,
                 recommended_min_s,
                 recommended_max_s,
                 decision_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    image_id,
                    round(min_k3, 1),
                    round(min_k5, 1),
                    round(grad_cap_s, 1) if grad_cap_s is not None else None,
                    round(rec_min, 1),
                    round(rec_max, 1),
                    "ok",
                ),
            )

        logger.log_module_summary(
            module="exposure_advice_worker",
            file=str(event.file_path),
            expected_read=6,
            read=6,
            expected_written=7,
            written=7,
            status="OK",
            duration_s=time.monotonic() - start,
        )
        return True

    except Exception as e:
        logger.log_failure(
            module="exposure_advice_worker",
            file=str(event.file_path),
            action="continue",
            reason=f"{type(e).__name__}:{e}",
        )
        return False
