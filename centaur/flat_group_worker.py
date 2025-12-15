from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

from zoneinfo import ZoneInfo

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent

MODULE_NAME = "flat_group_worker"
LOCAL_TZ = ZoneInfo("Europe/London")


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


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


def _is_flat(imagetyp: Optional[str]) -> bool:
    return (imagetyp or "").strip().upper() == "FLAT"


def _night_key_from_date_obs(date_obs: Optional[str]) -> Optional[str]:
    """
    Local observing night:
      - convert DATE-OBS to Europe/London
      - if local time < 12:00, assign to previous date
    Returns YYYY-MM-DD or None.
    """
    if not date_obs:
        return None
    try:
        dt = datetime.fromisoformat(date_obs.replace("Z", "+00:00"))
    except Exception:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    local = dt.astimezone(LOCAL_TZ)
    night_date = local.date()
    if local.hour < 12:
        night_date = (local - timedelta(days=1)).date()
    return night_date.isoformat()


@dataclass(frozen=True)
class _FlatProfileKey:
    camera: str
    filt: Optional[str]
    binning: Optional[str]
    telescope: Optional[str]
    focallen: Optional[float]
    naxis1: Optional[int]
    naxis2: Optional[int]


def process_file_event(cfg: AppConfig, logger: Logger, event: FileReadyEvent) -> Optional[bool]:
    """
    Create/attach:
      - flat_profiles row (physical compatibility)
      - flat_capture_sets row (night/target/filter/exptime session grouping)
      - flat_frame_links row linking image -> profile + capture set

    Runs only when IMAGETYP=FLAT.
    """
    t0 = time.monotonic()
    started_utc = utc_now()
    file_path = str(event.file_path)

    try:
        with Database().transaction() as db:
            row = db.execute(
                """
                SELECT
                  i.image_id,
                  f.imagetyp,
                  f.object AS target,
                  f.filter,
                  f.exptime,
                  f.instrume AS camera,
                  f.telescop AS telescope,
                  f.focallen,
                  f.naxis1,
                  f.naxis2,
                  f.date_obs,
                  f.xbinning,
                  f.ybinning
                FROM images i
                JOIN fits_header_core f USING(image_id)
                WHERE i.file_path = ?
                """,
                (file_path,),
            ).fetchone()

            if row is None:
                logger.log_failure(
                    module=MODULE_NAME,
                    file=file_path,
                    action="continue",
                    reason="missing_fits_header_core",
                )
                return False

            image_id = int(row["image_id"])

            if not _is_flat(row["imagetyp"]):
                # Not applicable, but log a clean module_run so sanity tools can see it ran.
                ended_utc = utc_now()
                _insert_module_run(
                    db,
                    image_id,
                    expected_read=1,
                    read=1,
                    written=0,
                    status="ok",
                    message="not_applicable_non_flat",
                    started_utc=started_utc,
                    ended_utc=ended_utc,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                )
                return True

            camera = (row["camera"] or "").strip()
            filt = (row["filter"] or None)
            target = (row["target"] or None)
            exptime = row["exptime"]
            telescope = (row["telescope"] or None)
            focallen = row["focallen"]
            naxis1 = row["naxis1"]
            naxis2 = row["naxis2"]

            xb = row["xbinning"]
            yb = row["ybinning"]
            binning = None
            if xb is not None and yb is not None:
                binning = f"{int(xb)}x{int(yb)}"

            night = _night_key_from_date_obs(row["date_obs"]) or "unknown"

            profile_key = _FlatProfileKey(
                camera=camera.lower(),
                filt=(filt.lower() if isinstance(filt, str) else filt),
                binning=binning,
                telescope=(telescope.lower() if isinstance(telescope, str) else telescope),
                focallen=(float(focallen) if focallen is not None else None),
                naxis1=(int(naxis1) if naxis1 is not None else None),
                naxis2=(int(naxis2) if naxis2 is not None else None),
            )

            # 1) Find or create flat_profile
            prof = db.execute(
                """
                SELECT flat_profile_id
                FROM flat_profiles
                WHERE lower(camera) = lower(?)
                  AND (filter IS NULL AND ? IS NULL OR lower(filter) = lower(?))
                  AND (binning IS NULL AND ? IS NULL OR binning = ?)
                  AND (telescope IS NULL AND ? IS NULL OR lower(telescope) = lower(?))
                  AND (focallen IS NULL AND ? IS NULL OR focallen = ?)
                  AND (naxis1 IS NULL AND ? IS NULL OR naxis1 = ?)
                  AND (naxis2 IS NULL AND ? IS NULL OR naxis2 = ?)
                LIMIT 1;
                """,
                (
                    profile_key.camera,
                    profile_key.filt, profile_key.filt,
                    profile_key.binning, profile_key.binning,
                    profile_key.telescope, profile_key.telescope,
                    profile_key.focallen, profile_key.focallen,
                    profile_key.naxis1, profile_key.naxis1,
                    profile_key.naxis2, profile_key.naxis2,
                ),
            ).fetchone()

            if prof is None:
                db.execute(
                    """
                    INSERT INTO flat_profiles
                    (camera, filter, binning, telescope, focallen, naxis1, naxis2, created_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        camera,
                        filt,
                        binning,
                        telescope,
                        float(focallen) if focallen is not None else None,
                        int(naxis1) if naxis1 is not None else None,
                        int(naxis2) if naxis2 is not None else None,
                        utc_now(),
                    ),
                )
                flat_profile_id = int(db.execute("SELECT last_insert_rowid() AS id;").fetchone()["id"])
            else:
                flat_profile_id = int(prof["flat_profile_id"])

            # 2) Find or create capture set
            cap = db.execute(
                """
                SELECT flat_capture_set_id
                FROM flat_capture_sets
                WHERE night = ?
                  AND (target IS NULL AND ? IS NULL OR lower(target) = lower(?))
                  AND lower(camera) = lower(?)
                  AND (filter IS NULL AND ? IS NULL OR lower(filter) = lower(?))
                  AND (binning IS NULL AND ? IS NULL OR binning = ?)
                  AND (exptime IS NULL AND ? IS NULL OR exptime = ?)
                  AND flat_profile_id = ?
                LIMIT 1;
                """,
                (
                    night,
                    target, target,
                    camera,
                    filt, filt,
                    binning, binning,
                    exptime, exptime,
                    flat_profile_id,
                ),
            ).fetchone()

            if cap is None:
                db.execute(
                    """
                    INSERT INTO flat_capture_sets
                    (night, target, camera, filter, binning, exptime, flat_profile_id, created_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        night,
                        target,
                        camera,
                        filt,
                        binning,
                        float(exptime) if exptime is not None else None,
                        flat_profile_id,
                        utc_now(),
                    ),
                )
                flat_capture_set_id = int(db.execute("SELECT last_insert_rowid() AS id;").fetchone()["id"])
            else:
                flat_capture_set_id = int(cap["flat_capture_set_id"])

            # 3) Link image -> profile + capture set
            db.execute(
                """
                INSERT OR REPLACE INTO flat_frame_links
                (image_id, flat_profile_id, flat_capture_set_id, created_utc)
                VALUES (?, ?, ?, ?)
                """,
                (
                    image_id,
                    flat_profile_id,
                    flat_capture_set_id,
                    utc_now(),
                ),
            )

            ended_utc = utc_now()
            duration_ms = int((time.monotonic() - t0) * 1000)

            # module_runs audit (so sanity can mark FLAT modules OK)
            _insert_module_run(
                db,
                image_id,
                expected_read=1,
                read=1,
                written=1,
                status="ok",
                message=None,
                started_utc=started_utc,
                ended_utc=ended_utc,
                duration_ms=duration_ms,
            )

        logger.log_module_summary(
            module=MODULE_NAME,
            file=file_path,
            expected_read=1,
            read=1,
            expected_written=1,
            written=1,
            status="OK",
            duration_s=time.monotonic() - t0,
        )
        return True

    except Exception as e:
        logger.log_failure(
            module=MODULE_NAME,
            file=file_path,
            action="continue",
            reason=f"{type(e).__name__}:{e}",
        )
        return False
