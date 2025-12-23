from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, Optional, Tuple

from centaur.config import AppConfig
from centaur.logging import Logger, utc_now
from centaur.pipeline import ModuleRunRecord
from centaur.watcher import FileReadyEvent

MODULE_NAME = "observing_conditions_worker"

# Reads: images + fits_header_core (optional astropy calls)
EXPECTED_READ = 2
EXPECTED_WRITTEN = 1


def _lookup_image_id(conn: sqlite3.Connection, file_path: str) -> Optional[int]:
    row = conn.execute(
        "SELECT image_id FROM images WHERE file_path = ?", (file_path,)
    ).fetchone()
    return int(row[0]) if row else None


def _row_as_dict(row: Optional[sqlite3.Row]) -> Dict[str, Any]:
    if row is None:
        return {}
    return {k: row[k] for k in row.keys()}


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return {r[1] for r in rows}


def _insert_dynamic(
    conn: sqlite3.Connection,
    table: str,
    values: Dict[str, Any],
    pk: str = "image_id",
) -> Tuple[int, int]:
    """
    Insert/replace only the keys that exist as columns in the table.
    Returns: (written_fields_ex_pk, expected_fields_ex_pk_attempted)
    """
    cols = _table_columns(conn, table)

    payload: Dict[str, Any] = {}
    for k, v in values.items():
        if k in cols:
            payload[k] = v

    # Ensure PK exists if the table has it and caller provided it
    if pk in cols and pk not in payload and pk in values:
        payload[pk] = values[pk]

    if not payload:
        return (0, 0)

    keys = list(payload.keys())
    placeholders = ", ".join(["?"] * len(keys))
    col_list = ", ".join(keys)

    sql = f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})"
    conn.execute(sql, tuple(payload[k] for k in keys))

    written_ex_pk = len([k for k in keys if k != pk])
    expected_attempted_ex_pk = len([k for k in values.keys() if k != pk])
    return (written_ex_pk, expected_attempted_ex_pk)


def _compute_audit_fields(
    conn: sqlite3.Connection, table: str, payload: Dict[str, Any]
) -> Tuple[int, int, int]:
    """
    Confirmed rule:
      expected_fields = columns - 1 (excluding image_id)
      read_fields = count of non-NULL produced values (excluding image_id)
      written_fields = payload keys (excluding image_id)
    """
    cols = _table_columns(conn, table)
    expected_fields = max(0, len(cols) - 1)

    # Only count what we will actually insert (i.e., in-table columns)
    produced_items = [(k, v) for (k, v) in payload.items() if k != "image_id"]
    read_fields = sum(1 for (_k, v) in produced_items if v is not None)
    written_fields = len(produced_items)

    return expected_fields, read_fields, written_fields


def process_file_event(
    cfg: AppConfig, logger: Logger, event: FileReadyEvent, ctx: Any = None
) -> ModuleRunRecord:
    t0 = time.perf_counter()
    image_id: int = -1

    conn = sqlite3.connect(str(cfg.db_path))
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA foreign_keys = ON;")

        resolved = _lookup_image_id(conn, str(event.file_path))
        if resolved is None:
            msg = "could_not_resolve_image_id (images.file_path lookup failed)"
            logger.log_failure(
                MODULE_NAME,
                str(event.file_path),
                action=cfg.on_metric_failure,
                reason=msg,
            )
            return ModuleRunRecord(
                image_id=-1,
                expected_read=EXPECTED_READ,
                read=0,
                expected_written=EXPECTED_WRITTEN,
                written=0,
                status="FAILED",
                message=msg,
            )
        image_id = int(resolved)

        h = _row_as_dict(
            conn.execute(
                "SELECT * FROM fits_header_core WHERE image_id = ?", (image_id,)
            ).fetchone()
        )
        read = 1 + (1 if h else 0)

        # Defaults
        usable = 1
        reason = "ok"
        parse_warnings: Optional[str] = None

        # Inputs from header (may be missing)
        date_obs = h.get("date_obs")
        ra = h.get("ra")
        dec = h.get("dec")
        lat = h.get("latitude")
        lon = h.get("longitude")
        elev_m = h.get("elevation_m")

        # pass-through identity if you want it later (schema has these)
        filt = h.get("filter")
        exptime_s = h.get("exptime")
        camera_name = h.get("instrume")

        airmass = None
        alt_deg = None
        az_deg = None

        # Compute airmass/alt/az if we have enough header data and astropy is available.
        try:
            if date_obs and ra and dec and (lat is not None) and (lon is not None):
                from astropy.time import Time
                from astropy import units as u
                from astropy.coordinates import SkyCoord, EarthLocation, AltAz

                t = Time(
                    str(date_obs), format="isot", scale="utc", out_subfmt="date_hms"
                )
                loc = EarthLocation(
                    lat=float(lat) * u.deg,
                    lon=float(lon) * u.deg,
                    height=(float(elev_m) if elev_m is not None else 0.0) * u.m,
                )
                sc = SkyCoord(
                    str(ra), str(dec), unit=(u.hourangle, u.deg), frame="icrs"
                )
                altaz = sc.transform_to(AltAz(obstime=t, location=loc))

                alt_deg = float(altaz.alt.to(u.deg).value)
                az_deg = float(altaz.az.to(u.deg).value)

                if altaz.secz is not None and altaz.secz.value == altaz.secz.value:
                    airmass = float(altaz.secz.value)
        except Exception:
            parse_warnings = "astropy_altaz_failed"

        # If we couldn't compute anything and don't even have basics, mark unusable.
        if airmass is None and (lat is None or lon is None or date_obs is None):
            usable = 0
            reason = "missing_header_for_conditions"

        # Build row payload (audit fields filled AFTER we know payload + schema)
        out: Dict[str, Any] = {
            "image_id": image_id,
            # audit fields: set later
            "expected_fields": None,
            "read_fields": None,
            "written_fields": None,
            "parse_warnings": parse_warnings,
            "db_written_utc": utc_now(),
            # Schema columns
            "date_obs": date_obs,
            "latitude": lat,
            "longitude": lon,
            "elevation_m": elev_m,
            "airmass": airmass,
            "alt_deg": alt_deg,
            "az_deg": az_deg,
            # optional “night context”
            "filter": filt,
            "exptime_s": exptime_s,
            "camera_name": camera_name,
            "usable": usable,
            "reason": reason,
        }

        # Filter to actual table columns first
        table = "observing_conditions"
        cols = _table_columns(conn, table)
        payload = {k: v for k, v in out.items() if k in cols}

        # Compute & set audit fields (confirmed rule)
        expected_fields, read_fields, written_fields = _compute_audit_fields(
            conn, table, payload
        )
        out["expected_fields"] = expected_fields
        out["read_fields"] = read_fields
        out["written_fields"] = written_fields

        # Rebuild payload including audit fields now filled
        payload = {k: v for k, v in out.items() if k in cols}

        # Insert
        written_ex_pk, _attempted_ex_pk = _insert_dynamic(
            conn, table, payload, pk="image_id"
        )
        conn.commit()

        duration_s = time.perf_counter() - t0

        logger.log_module_result(
            MODULE_NAME,
            str(event.file_path),
            expected_read=int(EXPECTED_READ),
            read=int(read),
            expected_written=int(EXPECTED_WRITTEN),
            written=1 if written_ex_pk > 0 else 0,
            status="OK",
            duration_s=duration_s,
            verbose_fields={
                "__inputs__": {
                    "image_id": image_id,
                    "have_date_obs": bool(date_obs),
                    "have_radec": bool(ra) and bool(dec),
                    "have_site": (lat is not None) and (lon is not None),
                },
                "__outputs__": {
                    "usable": usable,
                    "reason": reason,
                    "airmass": airmass,
                    "alt_deg": alt_deg,
                    "az_deg": az_deg,
                    "expected_fields": expected_fields,
                    "read_fields": read_fields,
                    "written_fields": written_fields,
                },
            },
        )

        return ModuleRunRecord(
            image_id=image_id,
            expected_read=int(EXPECTED_READ),
            read=int(read),
            expected_written=int(EXPECTED_WRITTEN),
            written=1 if written_ex_pk > 0 else 0,
            status="OK",
            message=f"{reason} airmass={airmass}",
        )

    except Exception as e:
        duration_s = time.perf_counter() - t0
        logger.log_failure(
            MODULE_NAME,
            str(event.file_path),
            action=cfg.on_metric_failure,
            reason=repr(e),
            duration_s=duration_s,
        )
        return ModuleRunRecord(
            image_id=int(image_id),
            expected_read=int(EXPECTED_READ),
            read=0,
            expected_written=int(EXPECTED_WRITTEN),
            written=0,
            status="FAILED",
            message=repr(e),
        )
    finally:
        conn.close()
