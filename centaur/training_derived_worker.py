from __future__ import annotations

import math
import sqlite3
import time
from typing import Any, Dict, Optional, Tuple

from centaur.config import AppConfig
from centaur.logging import Logger, utc_now
from centaur.pipeline import ModuleRunRecord
from centaur.watcher import FileReadyEvent

MODULE_NAME = "training_derived_worker"

# Upstream: fits_header_core, camera_constants, plus many optional metrics tables.
EXPECTED_READ = 8
EXPECTED_WRITTEN = 1


def _lookup_image_id(conn: sqlite3.Connection, file_path: str) -> Optional[int]:
    row = conn.execute(
        "SELECT image_id FROM images WHERE file_path = ?", (file_path,)
    ).fetchone()
    return int(row[0]) if row else None


def _fetch_row_dict(
    conn: sqlite3.Connection, table: str, image_id: int
) -> Dict[str, Any]:
    try:
        row = conn.execute(
            f"SELECT * FROM {table} WHERE image_id = ?", (image_id,)
        ).fetchone()
        if row is None:
            return {}
        return {k: row[k] for k in row.keys()}
    except sqlite3.Error:
        # table missing or other issue: treat as absent
        return {}


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def _insert_dynamic(
    conn: sqlite3.Connection, table: str, values: Dict[str, Any], pk: str = "image_id"
) -> Tuple[int, int]:
    """
    Insert/replace only keys that exist as columns.
    Returns: (written_fields_ex_pk, expected_attempted_ex_pk)
    """
    cols = _table_columns(conn, table)

    payload: Dict[str, Any] = {}
    for k, v in values.items():
        if k in cols:
            payload[k] = v

    if pk in cols and pk not in payload and pk in values:
        payload[pk] = values[pk]

    if not payload:
        return (0, 0)

    keys = list(payload.keys())
    sql = f"INSERT OR REPLACE INTO {table} ({', '.join(keys)}) VALUES ({', '.join(['?'] * len(keys))})"
    conn.execute(sql, tuple(payload[k] for k in keys))

    written_ex_pk = len([k for k in keys if k != pk])
    expected_attempted_ex_pk = len([k for k in values.keys() if k != pk])
    return (written_ex_pk, expected_attempted_ex_pk)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


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

        h = _fetch_row_dict(conn, "fits_header_core", image_id)

        # optional upstream
        psf = _fetch_row_dict(conn, "psf_basic_metrics", image_id)
        sat = _fetch_row_dict(conn, "saturation_metrics", image_id)
        sky = _fetch_row_dict(conn, "sky_basic_metrics", image_id)
        bkg2d = _fetch_row_dict(conn, "sky_background2d_metrics", image_id)
        roi = _fetch_row_dict(conn, "roi_signal_metrics", image_id)
        msig = _fetch_row_dict(conn, "masked_signal_metrics", image_id)
        head = _fetch_row_dict(conn, "star_headroom_metrics", image_id)
        neb = _fetch_row_dict(conn, "nebula_mask_metrics", image_id)
        struct = _fetch_row_dict(conn, "signal_structure_metrics", image_id)
        cond = _fetch_row_dict(conn, "observing_conditions", image_id)

        # camera constants lookup
        camera_name = str(h.get("instrume") or "").strip().lower()
        gain_setting = _safe_int(h.get("gain"))

        cc: Dict[str, Any] = {}
        try:
            if camera_name:
                if gain_setting is None:
                    r = conn.execute(
                        "SELECT * FROM camera_constants WHERE camera_name = ? AND gain_setting IS NULL",
                        (camera_name,),
                    ).fetchone()
                else:
                    r = conn.execute(
                        "SELECT * FROM camera_constants WHERE camera_name = ? AND gain_setting = ?",
                        (camera_name, int(gain_setting)),
                    ).fetchone()
                if r is not None:
                    cc = {k: r[k] for k in r.keys()}
        except sqlite3.Error:
            cc = {}

        # Rough read counter for log summary (not the audit read_fields)
        read = 1  # images lookup implicit
        for d in (h, psf, sat, sky, bkg2d, roi, msig, head, neb, struct, cc, cond):
            if d:
                read += 1

        exptime_s = (
            _safe_float(h.get("exptime"))
            or _safe_float(h.get("exptime_s"))
            or _safe_float(h.get("exptime_sec"))
        )
        filt = h.get("filter")

        usable = 1
        reason = "ok"
        parse_warnings: Optional[str] = None

        if exptime_s is None or exptime_s <= 0:
            usable = 0
            reason = "missing_exptime"

        # Gain/read noise from camera constants if present
        gain_e_per_adu = _safe_float(cc.get("gain_e_per_adu"))
        read_noise_e = _safe_float(cc.get("read_noise_e"))

        # Background (ADU and ADU/s) from sky_basic_metrics if present
        sky_ff_median_adu = _safe_float(sky.get("ff_median_adu")) or _safe_float(
            sky.get("sky_ff_median_adu")
        )
        sky_ff_median_adu_s = None
        if sky_ff_median_adu is not None and exptime_s and exptime_s > 0:
            sky_ff_median_adu_s = sky_ff_median_adu / exptime_s

        # Nebula signal (prefer masked) in ADU/s
        nebula_minus_bg_adu_s = _safe_float(msig.get("nebula_minus_bg_adu_s"))
        if nebula_minus_bg_adu_s is None:
            # fallback to roi if needed
            nebula_minus_bg_adu_s = _safe_float(roi.get("obj_minus_bg_adu_s"))

        # Simple transparency proxy: use sky rate
        transparency_proxy = sky_ff_median_adu_s

        # Star headroom and star peak rates (if present)
        headroom_p99 = _safe_float(head.get("headroom_p99"))
        star_peak_p99_adu = _safe_float(head.get("star_peak_p99_adu")) or _safe_float(
            head.get("peak_p99_adu")
        )
        star_peak_rate_p99_adu_s = None
        if star_peak_p99_adu is not None and exptime_s and exptime_s > 0:
            star_peak_rate_p99_adu_s = star_peak_p99_adu / exptime_s

        # Effective ceiling (linearity fraction default 0.90)
        linearity_fraction = _safe_float(cc.get("linearity_fraction"))
        if linearity_fraction is None:
            linearity_fraction = 0.90

        # pick a saturation ADU reference (best effort)
        saturation_adu_used = (
            _safe_float(h.get("datamax"))
            or _safe_float(sat.get("saturation_adu_used"))
            or _safe_float(sat.get("max_pixel_adu"))
            or _safe_float(sat.get("datamax"))
        )

        effective_ceiling_adu = None
        if saturation_adu_used is not None and linearity_fraction is not None:
            effective_ceiling_adu = float(saturation_adu_used) * float(
                linearity_fraction
            )

        # p99 over ceiling
        p99_over_linear_ceiling = None
        linear_headroom_p99 = None
        if (
            star_peak_p99_adu is not None
            and effective_ceiling_adu
            and effective_ceiling_adu > 0
        ):
            p99_over_linear_ceiling = float(star_peak_p99_adu) / float(
                effective_ceiling_adu
            )
            linear_headroom_p99 = 1.0 - p99_over_linear_ceiling

        # Sky-limited ratio in electrons (bg_e / rn^2)
        bg_e_per_pix = None
        bg_e_per_pix_s = None
        sky_limited_ratio = None
        min_exptime_sky_limited_s = None

        if (sky_ff_median_adu is not None) and (gain_e_per_adu is not None):
            bg_e_per_pix = float(sky_ff_median_adu) * float(gain_e_per_adu)
            if exptime_s and exptime_s > 0:
                bg_e_per_pix_s = bg_e_per_pix / float(exptime_s)

        if (
            (bg_e_per_pix is not None)
            and (read_noise_e is not None)
            and (read_noise_e > 0)
        ):
            sky_limited_ratio = bg_e_per_pix / (
                float(read_noise_e) * float(read_noise_e)
            )
            if bg_e_per_pix_s and bg_e_per_pix_s > 0:
                min_exptime_sky_limited_s = (
                    float(read_noise_e) * float(read_noise_e)
                ) / float(bg_e_per_pix_s)

        # Conditions passthrough
        airmass = _safe_float(cond.get("airmass"))
        moon_alt_deg = _safe_float(cond.get("moon_alt_deg"))
        moon_illum_frac = _safe_float(cond.get("moon_illum_frac"))
        moon_sep_deg = _safe_float(cond.get("moon_sep_deg"))

        # Quick safety
        nan_fraction = _safe_float(sat.get("nan_fraction"))
        inf_fraction = _safe_float(sat.get("inf_fraction"))
        if (nan_fraction and nan_fraction > 0) or (inf_fraction and inf_fraction > 0):
            usable = 0
            reason = "nan_or_inf"

        # Ratio proxies
        nebula_over_sky = None
        eff_proxy = None
        if (
            nebula_minus_bg_adu_s is not None
            and sky_ff_median_adu_s
            and sky_ff_median_adu_s > 0
        ):
            nebula_over_sky = nebula_minus_bg_adu_s / sky_ff_median_adu_s
            eff_proxy = nebula_minus_bg_adu_s / math.sqrt(sky_ff_median_adu_s)

        out: Dict[str, Any] = {
            "image_id": image_id,
            # audit fields: set later
            "expected_fields": None,
            "read_fields": None,
            "written_fields": None,
            "parse_warnings": parse_warnings,
            "db_written_utc": utc_now(),
            "filter": filt,
            "exptime_s": exptime_s,
            "camera_name": camera_name or None,
            "gain_setting": gain_setting,
            "transparency_proxy": transparency_proxy,
            "headroom_p99": headroom_p99,
            "star_peak_rate_p99_adu_s": star_peak_rate_p99_adu_s,
            "nebula_minus_bg_adu_s": nebula_minus_bg_adu_s,
            "sky_ff_median_adu_s": sky_ff_median_adu_s,
            "sky_limited_ratio": sky_limited_ratio,
            "nebula_over_sky": nebula_over_sky,
            "eff_proxy": eff_proxy,
            "effective_ceiling_adu": effective_ceiling_adu,
            "star_peak_p99_adu": star_peak_p99_adu,
            "p99_over_linear_ceiling": p99_over_linear_ceiling,
            "linear_headroom_p99": linear_headroom_p99,
            "airmass": airmass,
            "moon_alt_deg": moon_alt_deg,
            "moon_illum_frac": moon_illum_frac,
            "moon_sep_deg": moon_sep_deg,
            "usable": usable,
            "reason": reason,
        }

        table = "training_derived_metrics"
        cols = _table_columns(conn, table)
        payload = {k: v for k, v in out.items() if k in cols}

        expected_fields, read_fields, written_fields = _compute_audit_fields(
            conn, table, payload
        )
        out["expected_fields"] = expected_fields
        out["read_fields"] = read_fields
        out["written_fields"] = written_fields

        payload = {k: v for k, v in out.items() if k in cols}

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
                    "exptime_s": exptime_s,
                    "filter": filt,
                    "camera_name": camera_name,
                    "gain_setting": gain_setting,
                },
                "__outputs__": {
                    "usable": usable,
                    "reason": reason,
                    "transparency_proxy": transparency_proxy,
                    "nebula_minus_bg_adu_s": nebula_minus_bg_adu_s,
                    "sky_ff_median_adu_s": sky_ff_median_adu_s,
                    "sky_limited_ratio": sky_limited_ratio,
                    "min_exptime_sky_limited_s": min_exptime_sky_limited_s,
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
            message=f"{reason} sky_limited_ratio={sky_limited_ratio}",
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
