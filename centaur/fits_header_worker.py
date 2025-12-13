from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from astropy.io import fits  # type: ignore

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent

MODULE_NAME = "fits_header_worker"


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _first_present(header: fits.Header, keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in header:
            return header.get(k)
    return None


# Core field -> possible FITS keywords (best-effort, non-destructive)
CORE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    # Provenance
    "creator": ("CREATOR",),
    "origin": ("ORIGIN",),
    "software": ("SWCREATE", "SOFTWARE", "PROC", "PIPELINE"),
    "observer": ("OBSERVER",),
    "project": ("PROJID", "OBSID"),

    # Time & target
    "date_obs": ("DATE-OBS", "DATEOBS"),
    "date_end": ("DATE-END", "DATEEND"),
    "jd": ("JD",),
    "mjd_obs": ("MJD-OBS", "MJDOBS"),
    "object": ("OBJECT",),
    "ra": ("RA", "OBJRA"),
    "dec": ("DEC", "OBJDEC"),
    "equinox": ("EQUINOX",),

    # Exposure & capture
    "exptime": ("EXPTIME", "EXPOSURE"),
    "exp_total": ("EXPTOTAL", "TOTEXP"),
    "nsubexp": ("NSUBEXP", "NCOMBINE"),
    "imagetyp": ("IMAGETYP", "FRAME", "OBSTYPE"),
    "filter": ("FILTER", "FILTNAME", "FILTERID"),
    "seqnum": ("SEQNUM", "FRAMENO", "IMAGENO"),
    "gain": ("GAIN",),
    "offset": ("OFFSET", "BLACKLVL"),

    # Camera / detector
    "instrume": ("INSTRUME",),
    "detector": ("DETECTOR", "CAMERA", "SENSOR"),
    "ccd_temp": ("CCD-TEMP", "CCDTEMP", "CCD_TEMP", "SENSORTMP", "TEMPCCD"),
    "set_temp": ("SET-TEMP", "TEMPSET", "SETTEMP"),
    "xbinning": ("XBINNING", "XBIN"),
    "ybinning": ("YBINNING", "YBIN"),
    "readmode": ("READMODE",),
    "bayerpat": ("BAYERPAT",),
    "xpixsz": ("XPIXSZ",),
    "ypixsz": ("YPIXSZ",),

    # Optics / rig
    "telescop": ("TELESCOP",),
    "focallen": ("FOCALLEN",),
    "f_ratio": ("F_RATIO", "FRATIO"),
    "aperture": ("APERTURE", "DIAMETER"),
    "rotator": ("ROTATOR", "ROTPOS"),
    "focuspos": ("FOCUSPOS",),

    # Site / environment
    "sitename": ("SITENAME", "OBSERVAT"),
    "latitude": ("LAT", "OBS-LAT", "SITELAT"),
    "longitude": ("LON", "OBS-LONG", "SITELONG"),
    "elevation_m": ("ALT", "ELEVATION", "SITEELEV"),

    # Geometry / scaling
    "naxis1": ("NAXIS1",),
    "naxis2": ("NAXIS2",),
    "bitpix": ("BITPIX",),
    "bzero": ("BZERO",),
    "bscale": ("BSCALE",),
    "datamin": ("DATAMIN",),
    "datamax": ("DATAMAX",),

    # WCS
    "ctype1": ("CTYPE1",),
    "ctype2": ("CTYPE2",),
    "crval1": ("CRVAL1",),
    "crval2": ("CRVAL2",),
    "crpix1": ("CRPIX1",),
    "crpix2": ("CRPIX2",),
    "cdelt1": ("CDELT1",),
    "cdelt2": ("CDELT2",),

    "cd1_1": ("CD1_1",),
    "cd1_2": ("CD1_2",),
    "cd2_1": ("CD2_1",),
    "cd2_2": ("CD2_2",),

    "pc1_1": ("PC1_1",),
    "pc1_2": ("PC1_2",),
    "pc2_1": ("PC2_1",),
    "pc2_2": ("PC2_2",),
}

CORE_FIELDS = list(CORE_KEYWORDS.keys())


def _coerce_value(v: Any) -> Any:
    if v is None:
        return None
    try:
        import numpy as np  # type: ignore
        if isinstance(v, np.generic):
            v = v.item()
    except Exception:
        pass
    return v


def _header_to_json(header: fits.Header) -> str:
    # Preserve duplicates (HISTORY/COMMENT) by storing as list-of-cards
    cards = []
    for card in header.cards:
        cards.append(
            {"keyword": str(card.keyword), "value": card.value, "comment": card.comment}
        )
    return json.dumps(cards, default=str)


def _upsert_watch_root(db: Database, root_path: Path, root_label: str) -> int:
    now = utc_now()
    db.execute(
        """
        INSERT OR IGNORE INTO watch_roots (root_path, root_label, created_utc)
        VALUES (?, ?, ?)
        """,
        (str(root_path), root_label, now),
    )
    db.execute(
        "UPDATE watch_roots SET root_label = ? WHERE root_path = ?",
        (root_label, str(root_path)),
    )
    row = db.execute(
        "SELECT watch_root_id FROM watch_roots WHERE root_path = ?",
        (str(root_path),),
    ).fetchone()
    return int(row["watch_root_id"])


def _insert_image_row(db: Database, event: FileReadyEvent, watch_root_id: int) -> int:
    p = event.file_path
    st = p.stat()
    now = utc_now()

    rel = str(event.relative_path) if event.relative_path else None
    mtime_utc = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()

    db.execute(
        """
        INSERT OR IGNORE INTO images
        (file_path, relative_path, file_name, watch_root_id,
         file_size_bytes, file_mtime_utc,
         status, ignore_reason,
         stable_check_seconds, stable_check_passed,
         db_created_utc, db_updated_utc)
        VALUES (?, ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?,
                ?, ?)
        """,
        (
            str(p),
            rel,
            p.name,
            watch_root_id,
            int(st.st_size),
            mtime_utc,
            "processed",
            None,
            None,
            1,
            now,
            now,
        ),
    )
    row = db.execute("SELECT image_id FROM images WHERE file_path = ?", (str(p),)).fetchone()
    return int(row["image_id"])


def _find_setup_id(db: Database, telescop: Any, instrume: Any, detector: Any) -> Optional[int]:
    row = db.execute(
        """
        SELECT setup_id FROM optical_setups
        WHERE telescop IS ? AND instrume IS ? AND detector IS ?
        ORDER BY setup_id ASC
        LIMIT 1
        """,
        (telescop, instrume, detector),
    ).fetchone()
    return int(row["setup_id"]) if row else None


def _create_setup_id(db: Database, telescop: Any, instrume: Any, detector: Any) -> int:
    now = utc_now()
    db.execute(
        """
        INSERT INTO optical_setups (telescop, instrume, detector, site_name, site_lat, site_lon, created_utc)
        VALUES (?, ?, ?, NULL, NULL, NULL, ?)
        """,
        (telescop, instrume, detector, now),
    )
    row = db.execute("SELECT last_insert_rowid() AS id").fetchone()
    return int(row["id"])


def _link_image_setup(db: Database, image_id: int, setup_id: int) -> None:
    now = utc_now()
    db.execute(
        """
        INSERT OR REPLACE INTO image_setups (image_id, setup_id, method, confidence, db_written_utc)
        VALUES (?, ?, ?, ?, ?)
        """,
        (image_id, setup_id, "header", None, now),
    )


def _insert_module_run(
    db: Database,
    image_id: int,
    *,
    expected_read: int,
    read: int,
    expected_written: int,
    written: int,
    status: str,
    message: Optional[str],
    started_utc: str,
    ended_utc: str,
    duration_ms: int,
) -> None:
    now = utc_now()
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
            now,
        ),
    )


def process_file_event(config: AppConfig, logger: Logger, event: FileReadyEvent) -> None:
    t0 = time.monotonic()
    started_utc = utc_now()

    expected_read = len(CORE_FIELDS)
    expected_written = len(CORE_FIELDS)

    try:
        with fits.open(event.file_path, memmap=False) as hdul:
            header = hdul[0].header
            header_json = _header_to_json(header)
            header_bytes = len(header_json.encode("utf-8"))

            core: Dict[str, Any] = {}
            read_count = 0
            written_count = 0

            for field in CORE_FIELDS:
                raw = _first_present(header, CORE_KEYWORDS[field])
                val = _coerce_value(raw)
                core[field] = val
                if raw is not None:
                    read_count += 1
                if val is not None:
                    written_count += 1

        with Database().transaction() as db:
            watch_root_id = _upsert_watch_root(db, event.watch_root_path, event.watch_root_label)
            image_id = _insert_image_row(db, event, watch_root_id)

            # fits_header_core
            now = utc_now()
            cols = ["image_id"] + CORE_FIELDS + ["expected_fields", "read_fields", "parse_warnings", "db_written_utc"]
            vals = [image_id] + [core.get(k) for k in CORE_FIELDS] + [expected_read, read_count, None, now]
            placeholders = ",".join(["?"] * len(cols))
            col_list = ",".join(cols)

            db.execute(
                f"INSERT OR REPLACE INTO fits_header_core ({col_list}) VALUES ({placeholders})",
                tuple(vals),
            )

            # fits_header_full
            db.execute(
                """
                INSERT OR REPLACE INTO fits_header_full (image_id, header_json, header_bytes, db_written_utc)
                VALUES (?, ?, ?, ?)
                """,
                (image_id, header_json, header_bytes, utc_now()),
            )

            # setup mapping
            telescop = core.get("telescop")
            instrume = core.get("instrume")
            detector = core.get("detector")

            setup_id = _find_setup_id(db, telescop, instrume, detector)
            if setup_id is None:
                setup_id = _create_setup_id(db, telescop, instrume, detector)
            _link_image_setup(db, image_id, setup_id)

            ended_utc = utc_now()
            duration_ms = int((time.monotonic() - t0) * 1000)

            _insert_module_run(
                db,
                image_id,
                expected_read=expected_read,
                read=read_count,
                expected_written=expected_written,
                written=written_count,
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
            expected_read=expected_read,
            read=read_count,
            expected_written=expected_written,
            written=written_count,
            status="OK",
            duration_s=duration_s,
            verbose_fields={k: core.get(k) for k in CORE_FIELDS},
        )

    except Exception as e:
        duration_s = time.monotonic() - t0
        logger.log_failure(
            module=MODULE_NAME,
            file=str(event.file_path),
            action=config.on_metric_failure,
            reason=f"{type(e).__name__}:{e}",
            duration_s=duration_s,
        )

