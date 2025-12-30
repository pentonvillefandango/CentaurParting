# centaur/fits_header_worker.py
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent
from centaur.pipeline import ModuleRunRecord

# Unified loader (FITS + XISF)
from centaur.io.frame_loader import load_frame

MODULE_NAME = "fits_header_worker"


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# Core field -> possible keywords (best-effort, non-destructive)
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


def _header_cards_to_json(cards: list[dict]) -> str:
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
    row = db.execute(
        "SELECT image_id FROM images WHERE file_path = ?", (str(p),)
    ).fetchone()
    return int(row["image_id"])


def _find_setup_id(
    db: Database, telescop: Any, instrume: Any, detector: Any
) -> Optional[int]:
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


def _write_core_and_full(
    db: Database,
    *,
    image_id: int,
    header_source: str,
    core: Dict[str, Any],
    header_cards: list[dict],
    parse_warnings: Optional[str],
    expected_read: int,
    read_count: int,
) -> None:
    header_json = _header_cards_to_json(header_cards)
    header_bytes = len(header_json.encode("utf-8"))
    now = utc_now()

    cols = (
        ["image_id", "header_source"]
        + CORE_FIELDS
        + ["expected_fields", "read_fields", "parse_warnings", "db_written_utc"]
    )
    vals = (
        [image_id, header_source]
        + [core.get(k) for k in CORE_FIELDS]
        + [expected_read, read_count, parse_warnings, now]
    )
    placeholders = ",".join(["?"] * len(cols))
    col_list = ",".join(cols)

    db.execute(
        f"INSERT OR REPLACE INTO fits_header_core ({col_list}) VALUES ({placeholders})",
        tuple(vals),
    )

    db.execute(
        """
        INSERT OR REPLACE INTO fits_header_full (image_id, header_json, header_bytes, db_written_utc)
        VALUES (?, ?, ?, ?)
        """,
        (image_id, header_json, header_bytes, utc_now()),
    )


def _xisf_imagetype_from_xml(
    path: Path, *, max_scan_bytes: int = 16 * 1024 * 1024
) -> Optional[str]:
    """
    Best-effort extract of PixInsight XISF <Image ... imageType="FLAT|LIGHT|...">.

    Your current XISF metadata parser may not emit this as a header card, so we parse it here
    to ensure fits_header_core.imagetyp is populated for XISF (needed by flat workers, etc).
    """
    try:
        blob = path.read_bytes()[:max_scan_bytes]
        start = blob.find(b"<xisf")
        if start < 0:
            start = blob.find(b"<XISF")
        if start < 0:
            return None
        end = blob.find(b"</xisf>")
        if end < 0:
            end = blob.find(b"</XISF>")
        if end < 0:
            return None
        end = end + len(b"</xisf>")
        xml = blob[start:end].decode("utf-8", errors="replace")
    except Exception:
        return None

    # Find the first <Image ...> tag and read imageType="..."
    m_img = re.search(r"<Image\b[^>]*>", xml, flags=re.IGNORECASE)
    if not m_img:
        return None

    tag = m_img.group(0)
    m = re.search(r'\bimageType\s*=\s*"([^"]+)"', tag, flags=re.IGNORECASE)
    if not m:
        return None

    v = (m.group(1) or "").strip()
    if not v:
        return None

    u = v.upper()
    # Keep it conservative, but map common composites
    if u in ("LIGHT", "FLAT", "DARK", "BIAS"):
        return u
    if "FLAT" in u:
        return "FLAT"
    if "DARK" in u:
        return "DARK"
    if "BIAS" in u:
        return "BIAS"
    if "LIGHT" in u:
        return "LIGHT"
    return u


def process_file_event(config: AppConfig, logger: Logger, event: FileReadyEvent) -> Any:
    t0 = time.monotonic()

    expected_read = len(CORE_FIELDS)
    expected_written = len(CORE_FIELDS)

    image_id: Optional[int] = None
    core: Dict[str, Any] = {}
    read_count = 0
    written_count = 0
    parse_warnings: Optional[str] = None

    # Always create the images row first so we can ALWAYS return a ModuleRunRecord.
    try:
        with Database().transaction() as db:
            watch_root_id = _upsert_watch_root(
                db, event.watch_root_path, event.watch_root_label
            )
            image_id = _insert_image_row(db, event, watch_root_id)

    except Exception as e:
        logger.log_failure(
            module=MODULE_NAME,
            file=str(event.file_path),
            action=config.on_metric_failure,
            reason=f"db_insert_image_failed:{type(e).__name__}:{e}",
            duration_s=time.monotonic() - t0,
        )
        return False  # no image_id -> cannot produce ModuleRunRecord safely

    try:
        # Load metadata (FITS or XISF)
        frame = load_frame(
            Path(event.file_path),
            pixels=False,
            core_fields=CORE_FIELDS,
            core_keywords=CORE_KEYWORDS,
        )

        # Start with what the loader provided
        core.update(frame.core or {})

        # XISF-only: ensure imagetyp exists (needed for flat workers, etc.)
        p = Path(event.file_path)
        if (p.suffix.lower() == ".xisf") and (core.get("imagetyp") is None):
            it = _xisf_imagetype_from_xml(p)
            if it:
                core["imagetyp"] = it
                parse_warnings = (
                    "imagetyp_from_xisf_imagetype"
                    if not parse_warnings
                    else (parse_warnings + ";imagetyp_from_xisf_imagetype")
                )

        # Count "read" as “any candidate keyword present in extracted raw keys”
        raw_keys_present = set()
        for c in frame.header_cards:
            k = str(c.get("keyword") or "").strip().upper()
            if k:
                raw_keys_present.add(k)

        for field in CORE_FIELDS:
            # Normal read_count logic from keyword presence
            if any(
                str(k).strip().upper() in raw_keys_present for k in CORE_KEYWORDS[field]
            ):
                read_count += 1

            # If we synthesized imagetyp from XML, count that as "read" too (so audit matches reality)
            if field == "imagetyp" and core.get("imagetyp") is not None:
                # only bump if none of the usual keywords were present
                if not any(
                    str(k).strip().upper() in raw_keys_present
                    for k in CORE_KEYWORDS["imagetyp"]
                ):
                    read_count += 1

            val = core.get(field)
            if val is not None:
                written_count += 1

        # Write header tables + setup mapping
        with Database().transaction() as db:
            _write_core_and_full(
                db,
                image_id=int(image_id),
                header_source=str(frame.source),
                core=core,
                header_cards=frame.header_cards,
                parse_warnings=parse_warnings,
                expected_read=int(expected_read),
                read_count=int(read_count),
            )

            telescop = core.get("telescop")
            instrume = core.get("instrume")
            detector = core.get("detector")

            setup_id = _find_setup_id(db, telescop, instrume, detector)
            if setup_id is None:
                setup_id = _create_setup_id(db, telescop, instrume, detector)
            _link_image_setup(db, int(image_id), setup_id)

        logger.log_module_result(
            module=MODULE_NAME,
            file=str(event.file_path),
            expected_read=expected_read,
            read=read_count,
            expected_written=expected_written,
            written=written_count,
            status="OK",
            duration_s=time.monotonic() - t0,
            verbose_fields={k: core.get(k) for k in CORE_FIELDS},
        )

        return ModuleRunRecord(
            image_id=int(image_id),
            expected_read=int(expected_read),
            read=int(read_count),
            expected_written=int(expected_written),
            written=int(written_count),
            status="OK",
            message=parse_warnings,
        )

    except Exception as e:
        # Persist a minimal header row so downstream can see this frame existed + why it failed
        warn = f"{type(e).__name__}:{e}"
        try:
            with Database().transaction() as db:
                _write_core_and_full(
                    db,
                    image_id=int(image_id),
                    header_source="unknown",
                    core={k: None for k in CORE_FIELDS},
                    header_cards=[
                        {
                            "keyword": "PARSE_ERROR",
                            "value": warn,
                            "comment": "fits_header_worker failure",
                        }
                    ],
                    parse_warnings=warn,
                    expected_read=int(expected_read),
                    read_count=0,
                )
        except Exception:
            pass

        logger.log_failure(
            module=MODULE_NAME,
            file=str(event.file_path),
            action=config.on_metric_failure,
            reason=warn,
            duration_s=time.monotonic() - t0,
        )

        return ModuleRunRecord(
            image_id=int(image_id),
            expected_read=int(expected_read),
            read=int(read_count),
            expected_written=int(expected_written),
            written=int(written_count),
            status="FAILED",
            message=warn,
        )
