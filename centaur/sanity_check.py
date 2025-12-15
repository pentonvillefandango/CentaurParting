
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Helpers
# -----------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    r = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return bool(r)


def _col_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    if not _table_exists(conn, table):
        return False
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return any(str(r["name"]) == col for r in rows)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        if v == float("inf") or v == float("-inf"):
            return None
        return v
    except Exception:
        return None


def _clamp01(v: Optional[float]) -> bool:
    if v is None:
        return True
    return 0.0 <= v <= 1.0


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


# -----------------------------
# Data model for report rows
# -----------------------------

@dataclass
class ImageRow:
    image_id: int
    file_name: str
    imagetyp: str
    naxis1: Optional[int]
    naxis2: Optional[int]
    status: str

    # module ok flags (tri-state: 1 ok, 0 missing/failed, blank if not applicable)
    fits_ok: int = 0
    sky_basic_ok: str = ""
    sky_bkg2d_ok: str = ""
    exposure_ok: str = ""
    psf0_ok: str = ""
    psf1_ok: str = ""
    psf2_ok: str = ""
    flat_group_ok: str = ""
    flat_basic_ok: str = ""

    # key metrics (optional)
    sky_roi_median_adu: Optional[float] = None
    sky_roi_madstd_adu: Optional[float] = None

    psf0_n_peaks_good: Optional[int] = None
    psf1_n_measured: Optional[int] = None
    psf2_n_modeled: Optional[int] = None
    psf2_gauss_fwhm_med: Optional[float] = None
    psf2_gauss_fwhm_p90: Optional[float] = None
    psf2_gauss_ecc_med: Optional[float] = None

    # performance
    psf2_duration_ms: Optional[int] = None
    psf2_ms_per_star: Optional[float] = None

    # anomalies
    anomalies: str = ""


# -----------------------------
# Sanity checks
# -----------------------------

def _fetch_images(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    # images has file_name/status; fits_header_core has imagetyp/naxis1/naxis2
    sql = """
    SELECT
      i.image_id,
      i.file_name,
      i.status,
      COALESCE(UPPER(TRIM(h.imagetyp)), '') AS imagetyp,
      h.naxis1,
      h.naxis2
    FROM images i
    LEFT JOIN fits_header_core h USING (image_id)
    ORDER BY i.image_id ASC;
    """
    return conn.execute(sql).fetchall()


def _module_run_ok_map(conn: sqlite3.Connection) -> Dict[Tuple[int, str], int]:
    """
    Map (image_id, module_name) -> 1 if latest run status == 'ok' else 0.
    If multiple runs exist, we pick the latest by ended_utc if present, else rowid.
    """
    if not _table_exists(conn, "module_runs"):
        return {}

    rows = conn.execute(
        """
        SELECT image_id, module_name, status
        FROM module_runs
        """
    ).fetchall()

    # If multiple exist, last-write-wins by iteration order; OK for sanity report
    out: Dict[Tuple[int, str], int] = {}
    for r in rows:
        key = (int(r["image_id"]), str(r["module_name"]))
        st = str(r["status"] or "").lower()
        out[key] = 1 if st == "ok" else 0
    return out


def _append_anom(anoms: List[str], msg: str) -> None:
    if msg and msg not in anoms:
        anoms.append(msg)


def _check_fits_header(row: ImageRow, anoms: List[str]) -> None:
    if row.naxis1 is None or row.naxis2 is None:
        _append_anom(anoms, "fits_header_missing_geometry")
    else:
        if row.naxis1 <= 0 or row.naxis2 <= 0:
            _append_anom(anoms, "fits_header_bad_geometry")


def _check_sky_basic(conn: sqlite3.Connection, row: ImageRow, anoms: List[str]) -> None:
    if not _table_exists(conn, "sky_basic_metrics"):
        return
    r = conn.execute(
        "SELECT * FROM sky_basic_metrics WHERE image_id=?",
        (row.image_id,),
    ).fetchone()
    if not r:
        _append_anom(anoms, "sky_basic_missing_row")
        return

    # light touch: health fractions in [0,1]
    nanf = _safe_float(r["nan_fraction"])
    inff = _safe_float(r["inf_fraction"])
    if not _clamp01(nanf):
        _append_anom(anoms, "sky_basic_nan_fraction_out_of_range")
    if not _clamp01(inff):
        _append_anom(anoms, "sky_basic_inf_fraction_out_of_range")

    # store a couple of helpful values in report row
    row.sky_roi_median_adu = _safe_float(r["roi_median_adu"])
    row.sky_roi_madstd_adu = _safe_float(r["roi_madstd_adu"])

    # consistency checks if values exist
    vmin = _safe_float(r["roi_min_adu"])
    vmax = _safe_float(r["roi_max_adu"])
    vmed = row.sky_roi_median_adu
    if vmin is not None and vmax is not None and vmin > vmax:
        _append_anom(anoms, "sky_basic_roi_min_gt_max")
    if vmin is not None and vmax is not None and vmed is not None:
        if not (vmin <= vmed <= vmax):
            _append_anom(anoms, "sky_basic_roi_median_outside_minmax")

    cf_ff = _safe_float(r["clipped_fraction_ff"])
    cf_roi = _safe_float(r["clipped_fraction_roi"])
    if not _clamp01(cf_ff):
        _append_anom(anoms, "sky_basic_clipped_fraction_ff_out_of_range")
    if not _clamp01(cf_roi):
        _append_anom(anoms, "sky_basic_clipped_fraction_roi_out_of_range")


def _check_sky_bkg2d(conn: sqlite3.Connection, row: ImageRow, anoms: List[str]) -> None:
    if not _table_exists(conn, "sky_background2d_metrics"):
        return
    r = conn.execute(
        "SELECT * FROM sky_background2d_metrics WHERE image_id=?",
        (row.image_id,),
    ).fetchone()
    if not r:
        _append_anom(anoms, "sky_background2d_missing_row")
        return
    # not assuming exact schema; just check audit if present
    if "expected_fields" in r.keys() and "written_fields" in r.keys():
        ef = int(r["expected_fields"] or 0)
        wf = int(r["written_fields"] or 0)
        if ef > 0 and wf == 0:
            _append_anom(anoms, "sky_background2d_wrote_zero_fields")


def _check_psf0(conn: sqlite3.Connection, row: ImageRow, anoms: List[str]) -> None:
    if not _table_exists(conn, "psf_detect_metrics"):
        return
    r = conn.execute(
        "SELECT n_peaks_total, n_peaks_good, usable, reason FROM psf_detect_metrics WHERE image_id=?",
        (row.image_id,),
    ).fetchone()
    if not r:
        _append_anom(anoms, "psf0_missing_row")
        return

    n_total = int(r["n_peaks_total"] or 0)
    n_good = int(r["n_peaks_good"] or 0)
    row.psf0_n_peaks_good = n_good

    if n_good > n_total:
        _append_anom(anoms, "psf0_good_gt_total")

    usable = int(r["usable"] or 0)
    reason = str(r["reason"] or "")
    if usable == 1 and n_good < 25:
        _append_anom(anoms, "psf0_usable_but_too_few_peaks")
    if usable == 0 and reason == "ok":
        _append_anom(anoms, "psf0_not_usable_but_reason_ok")


def _check_psf1(conn: sqlite3.Connection, row: ImageRow, anoms: List[str]) -> None:
    if not _table_exists(conn, "psf_basic_metrics"):
        return
    r = conn.execute(
        "SELECT n_peaks_good, n_measured, usable, reason, star_xy_json FROM psf_basic_metrics WHERE image_id=?",
        (row.image_id,),
    ).fetchone()
    if not r:
        _append_anom(anoms, "psf1_missing_row")
        return

    n_good = int(r["n_peaks_good"] or 0)
    n_meas = int(r["n_measured"] or 0)
    row.psf1_n_measured = n_meas

    if n_meas > n_good:
        _append_anom(anoms, "psf1_measured_gt_good")

    usable = int(r["usable"] or 0)
    reason = str(r["reason"] or "")
    if usable == 1 and n_meas < 25:
        _append_anom(anoms, "psf1_usable_but_too_few_measured")
    if usable == 0 and reason == "ok":
        _append_anom(anoms, "psf1_not_usable_but_reason_ok")

    # JSON sanity (if present)
    star_json = r["star_xy_json"]
    if star_json is not None:
        try:
            arr = json.loads(star_json)
            if not isinstance(arr, list):
                _append_anom(anoms, "psf1_star_xy_json_not_list")
            else:
                if len(arr) != n_meas:
                    _append_anom(anoms, "psf1_star_xy_json_len_mismatch")
                # spot-check first few shapes
                for j in arr[:5]:
                    if not isinstance(j, dict) or "x" not in j or "y" not in j:
                        _append_anom(anoms, "psf1_star_xy_json_bad_items")
                        break
        except Exception:
            _append_anom(anoms, "psf1_star_xy_json_parse_failed")


def _check_psf2(conn: sqlite3.Connection, row: ImageRow, anoms: List[str]) -> None:
    if not _table_exists(conn, "psf_model_metrics"):
        return
    r = conn.execute(
        """
        SELECT
          n_input_stars,
          n_modeled,
          usable,
          reason,
          gauss_fwhm_px_median,
          gauss_fwhm_px_p90,
          gauss_ecc_median
        FROM psf_model_metrics
        WHERE image_id=?
        """,
        (row.image_id,),
    ).fetchone()
    if not r:
        _append_anom(anoms, "psf2_missing_row")
        return

    n_input = int(r["n_input_stars"] or 0)
    n_mod = int(r["n_modeled"] or 0)
    row.psf2_n_modeled = n_mod
    row.psf2_gauss_fwhm_med = _safe_float(r["gauss_fwhm_px_median"])
    row.psf2_gauss_fwhm_p90 = _safe_float(r["gauss_fwhm_px_p90"])
    row.psf2_gauss_ecc_med = _safe_float(r["gauss_ecc_median"])

    # The key invariant (for your final design): modeled should never exceed PSF-1 input.
    # If this triggers, something re-detected / duplicated.
    if n_input > 0 and n_mod > n_input:
        _append_anom(anoms, f"psf2_modeled_gt_input({n_mod}>{n_input})")

    usable = int(r["usable"] or 0)
    reason = str(r["reason"] or "")
    if usable == 1 and n_mod < 25:
        _append_anom(anoms, "psf2_usable_but_too_few_modeled")
    if usable == 0 and reason == "ok":
        _append_anom(anoms, "psf2_not_usable_but_reason_ok")


def _psf2_duration(conn: sqlite3.Connection, image_id: int) -> Optional[int]:
    if not _table_exists(conn, "module_runs"):
        return None
    r = conn.execute(
        """
        SELECT duration_ms
        FROM module_runs
        WHERE image_id=?
          AND module_name='psf_model_worker'
        ORDER BY rowid DESC
        LIMIT 1
        """,
        (image_id,),
    ).fetchone()
    if not r:
        return None
    try:
        return int(r["duration_ms"])
    except Exception:
        return None


# -----------------------------
# Main report generation
# -----------------------------

def run(db_path: Path, out_csv: Path, out_json: Path) -> int:
    conn = _connect(db_path)

    images = _fetch_images(conn)
    mod_ok = _module_run_ok_map(conn)

    # quick summary counters
    summary: Dict[str, Any] = {
        "db_path": str(db_path),
        "n_images": len(images),
        "status_counts": {},
        "imagetyp_counts": {},
        "tables_present": {},
        "anomaly_counts": {},
    }

    # detect available tables
    for t in [
        "images",
        "fits_header_core",
        "sky_basic_metrics",
        "sky_background2d_metrics",
        "exposure_advice",
        "psf_detect_metrics",
        "psf_basic_metrics",
        "psf_model_metrics",
        "module_runs",
        "flat_metrics",
    ]:
        summary["tables_present"][t] = _table_exists(conn, t)

    rows_out: List[ImageRow] = []

    for r in images:
        image_id = int(r["image_id"])
        file_name = str(r["file_name"] or "")
        status = str(r["status"] or "")
        imagetyp = str(r["imagetyp"] or "")
        naxis1 = r["naxis1"]
        naxis2 = r["naxis2"]
        na1 = int(naxis1) if naxis1 is not None else None
        na2 = int(naxis2) if naxis2 is not None else None

        # counts
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
        summary["imagetyp_counts"][imagetyp] = summary["imagetyp_counts"].get(imagetyp, 0) + 1

        row = ImageRow(
            image_id=image_id,
            file_name=file_name,
            imagetyp=imagetyp,
            naxis1=na1,
            naxis2=na2,
            status=status,
        )

        anoms: List[str] = []

        # module ok flags (based on module_runs if present)
        row.fits_ok = mod_ok.get((image_id, "fits_header_worker"), 0)

        # applicability routing
        is_flat = (imagetyp == "FLAT")
        is_light = (imagetyp == "LIGHT" or imagetyp == "")

        # For readability in CSV: "" means N/A; "1"/"0" means applicable
        def set_flag(attr: str, applicable: bool, module_name: str) -> None:
            if not applicable:
                setattr(row, attr, "")
                return
            setattr(row, attr, "1" if mod_ok.get((image_id, module_name), 0) == 1 else "0")

        set_flag("flat_group_ok", is_flat, "flat_group_worker")
        set_flag("flat_basic_ok", is_flat, "flat_basic_worker")

        set_flag("sky_basic_ok", (not is_flat), "sky_basic_worker")
        set_flag("sky_bkg2d_ok", (not is_flat), "sky_background2d_worker")
        set_flag("exposure_ok", (not is_flat), "exposure_advice_worker")

        set_flag("psf0_ok", (not is_flat), "psf_detect_worker")
        set_flag("psf1_ok", (not is_flat), "psf_basic_worker")
        set_flag("psf2_ok", (not is_flat), "psf_model_worker")

        # core checks
        _check_fits_header(row, anoms)

        # metrics checks (only if not flat)
        if not is_flat:
            _check_sky_basic(conn, row, anoms)
            _check_sky_bkg2d(conn, row, anoms)
            _check_psf0(conn, row, anoms)
            _check_psf1(conn, row, anoms)
            _check_psf2(conn, row, anoms)

            # perf
            d = _psf2_duration(conn, image_id)
            row.psf2_duration_ms = d
            if d is not None and row.psf2_n_modeled and row.psf2_n_modeled > 0:
                row.psf2_ms_per_star = float(d) / float(row.psf2_n_modeled)

        # anomaly bookkeeping
        row.anomalies = ";".join(anoms)
        for a in anoms:
            summary["anomaly_counts"][a] = summary["anomaly_counts"].get(a, 0) + 1

        rows_out.append(row)

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "image_id", "file_name", "status", "imagetyp", "naxis1", "naxis2",
            "fits_ok",
            "sky_basic_ok", "sky_background2d_ok", "exposure_ok",
            "psf0_ok", "psf1_ok", "psf2_ok",
            "flat_group_ok", "flat_basic_ok",
            "sky_roi_median_adu", "sky_roi_madstd_adu",
            "psf0_n_peaks_good",
            "psf1_n_measured",
            "psf2_n_modeled",
            "psf2_gauss_fwhm_med", "psf2_gauss_fwhm_p90", "psf2_gauss_ecc_med",
            "psf2_duration_ms", "psf2_ms_per_star",
            "anomalies",
        ])
        for r in rows_out:
            w.writerow([
                r.image_id, r.file_name, r.status, r.imagetyp, r.naxis1, r.naxis2,
                r.fits_ok,
                r.sky_basic_ok, r.sky_bkg2d_ok, r.exposure_ok,
                r.psf0_ok, r.psf1_ok, r.psf2_ok,
                r.flat_group_ok, r.flat_basic_ok,
                r.sky_roi_median_adu, r.sky_roi_madstd_adu,
                r.psf0_n_peaks_good,
                r.psf1_n_measured,
                r.psf2_n_modeled,
                r.psf2_gauss_fwhm_med, r.psf2_gauss_fwhm_p90, r.psf2_gauss_ecc_med,
                r.psf2_duration_ms, r.psf2_ms_per_star,
                r.anomalies,
            ])

    # Write JSON summary
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    # Print a small console summary (kept short on purpose)
    n_anom_total = sum(int(v) for v in summary["anomaly_counts"].values())
    print(f"[sanity_check] images={summary['n_images']} total_anomalies={n_anom_total}")
    if summary["anomaly_counts"]:
        top = sorted(summary["anomaly_counts"].items(), key=lambda kv: kv[1], reverse=True)[:10]
        for k, v in top:
            print(f"  {k}: {v}")

    conn.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Centaur sanity checker (DB -> CSV + JSON).")
    ap.add_argument("--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB")
    ap.add_argument("--out", type=str, default="data/sanity_report.csv", help="Output CSV path")
    ap.add_argument("--summary", type=str, default="data/sanity_summary.json", help="Output JSON summary path")
    args = ap.parse_args()

    return run(Path(args.db), Path(args.out), Path(args.summary))


if __name__ == "__main__":
    raise SystemExit(main())
