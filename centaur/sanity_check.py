# ---------------------------------------------
# how to run
# python3 centaur/sanity_check.py --db data/centaurparting.db
#
# (optional explicit paths)
# python3 centaur/sanity_check.py --db data/centaurparting.db \
#   --out data/sanity_report.csv \
#   --summary data/sanity_summary.json \
#   --perf data/sanity_performance.csv
#
# outputs (default if you omit --out/--summary/--perf):
#   data/sanity_report_YYYYMMDD_HHMMSS.csv
#   data/sanity_summary_YYYYMMDD_HHMMSS.json
#   data/sanity_performance_YYYYMMDD_HHMMSS.csv
# ---------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Helpers
# -----------------------------

def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def _view_exists(conn: sqlite3.Connection, name: str) -> bool:
    r = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='view' AND name=?",
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


def _append_anom(anoms: List[str], msg: str) -> None:
    if msg and msg not in anoms:
        anoms.append(msg)


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
    psf_grid_ok: str = ""
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

    # PSF grid key rollups
    psf_grid_n_input_stars: Optional[int] = None
    psf_grid_n_cells_with_data: Optional[int] = None
    psf_grid_center_fwhm_px: Optional[float] = None
    psf_grid_corner_fwhm_px_median: Optional[float] = None
    psf_grid_center_to_corner_ratio: Optional[float] = None

    # performance (from module_runs)
    fits_duration_ms: Optional[int] = None
    sky_basic_duration_ms: Optional[int] = None
    sky_bkg2d_duration_ms: Optional[int] = None
    exposure_duration_ms: Optional[int] = None
    flat_group_duration_ms: Optional[int] = None
    flat_basic_duration_ms: Optional[int] = None
    psf0_duration_ms: Optional[int] = None
    psf1_duration_ms: Optional[int] = None
    psf_grid_duration_ms: Optional[int] = None
    psf2_duration_ms: Optional[int] = None

    psf2_ms_per_star: Optional[float] = None

    # anomalies
    anomalies: str = ""


# -----------------------------
# Sanity checks (metrics tables)
# -----------------------------

def _fetch_images(conn: sqlite3.Connection) -> List[sqlite3.Row]:
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
    if not _table_exists(conn, "module_runs"):
        return {}

    rows = conn.execute(
        """
        SELECT image_id, module_name, status
        FROM module_runs
        """
    ).fetchall()

    out: Dict[Tuple[int, str], int] = {}
    for r in rows:
        key = (int(r["image_id"]), str(r["module_name"]))
        st = str(r["status"] or "").lower()
        out[key] = 1 if st == "ok" else 0
    return out


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

    nanf = _safe_float(r["nan_fraction"])
    inff = _safe_float(r["inf_fraction"])
    if not _clamp01(nanf):
        _append_anom(anoms, "sky_basic_nan_fraction_out_of_range")
    if not _clamp01(inff):
        _append_anom(anoms, "sky_basic_inf_fraction_out_of_range")

    row.sky_roi_median_adu = _safe_float(r["roi_median_adu"])
    row.sky_roi_madstd_adu = _safe_float(r["roi_madstd_adu"])

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

    # light validation of star_xy_json shape
    star_json = r["star_xy_json"]
    if star_json is not None:
        try:
            arr = json.loads(star_json)
            if not isinstance(arr, list):
                _append_anom(anoms, "psf1_star_xy_json_not_list")
            else:
                if len(arr) != n_meas:
                    _append_anom(anoms, "psf1_star_xy_json_len_mismatch")
                for j in arr[:5]:
                    if not isinstance(j, dict) or "x" not in j or "y" not in j:
                        _append_anom(anoms, "psf1_star_xy_json_bad_items")
                        break
        except Exception:
            _append_anom(anoms, "psf1_star_xy_json_parse_failed")


def _check_psf_grid(conn: sqlite3.Connection, row: ImageRow, anoms: List[str]) -> None:
    if not _table_exists(conn, "psf_grid_metrics"):
        return
    r = conn.execute(
        """
        SELECT
          grid_rows,
          grid_cols,
          grid_min_stars_per_cell,
          n_input_stars,
          n_cells_with_data,
          center_fwhm_px,
          corner_fwhm_px_median,
          center_to_corner_fwhm_ratio,
          left_right_fwhm_ratio,
          top_bottom_fwhm_ratio,
          usable,
          reason,

          cell_r0c0_n, cell_r0c1_n, cell_r0c2_n,
          cell_r1c0_n, cell_r1c1_n, cell_r1c2_n,
          cell_r2c0_n, cell_r2c1_n, cell_r2c2_n
        FROM psf_grid_metrics
        WHERE image_id=?
        """,
        (row.image_id,),
    ).fetchone()
    if not r:
        _append_anom(anoms, "psf_grid_missing_row")
        return

    grid_rows = int(r["grid_rows"] or 0)
    grid_cols = int(r["grid_cols"] or 0)
    if grid_rows != 3 or grid_cols != 3:
        _append_anom(anoms, f"psf_grid_unexpected_grid_size({grid_rows}x{grid_cols})")

    n_input = int(r["n_input_stars"] or 0)
    n_cells = int(r["n_cells_with_data"] or 0)
    row.psf_grid_n_input_stars = n_input
    row.psf_grid_n_cells_with_data = n_cells

    row.psf_grid_center_fwhm_px = _safe_float(r["center_fwhm_px"])
    row.psf_grid_corner_fwhm_px_median = _safe_float(r["corner_fwhm_px_median"])
    row.psf_grid_center_to_corner_ratio = _safe_float(r["center_to_corner_fwhm_ratio"])

    # Invariant: sum of cell counts should not exceed n_input_stars
    cell_counts = [
        int(r["cell_r0c0_n"] or 0), int(r["cell_r0c1_n"] or 0), int(r["cell_r0c2_n"] or 0),
        int(r["cell_r1c0_n"] or 0), int(r["cell_r1c1_n"] or 0), int(r["cell_r1c2_n"] or 0),
        int(r["cell_r2c0_n"] or 0), int(r["cell_r2c1_n"] or 0), int(r["cell_r2c2_n"] or 0),
    ]
    sum_cells = sum(cell_counts)
    if n_input > 0 and sum_cells > n_input:
        _append_anom(anoms, f"psf_grid_sum_cell_n_gt_input({sum_cells}>{n_input})")

    # Ratios should be positive if present
    for col, name in [
        ("center_to_corner_fwhm_ratio", "psf_grid_center_to_corner_ratio_nonpos"),
        ("left_right_fwhm_ratio", "psf_grid_left_right_ratio_nonpos"),
        ("top_bottom_fwhm_ratio", "psf_grid_top_bottom_ratio_nonpos"),
    ]:
        v = _safe_float(r[col])
        if v is not None and v <= 0:
            _append_anom(anoms, name)

    usable = int(r["usable"] or 0)
    reason = str(r["reason"] or "")
    if usable == 1 and n_cells < 3:
        _append_anom(anoms, "psf_grid_usable_but_too_few_cells")
    if usable == 0 and reason == "ok":
        _append_anom(anoms, "psf_grid_not_usable_but_reason_ok")


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

    if n_input > 0 and n_mod > n_input:
        _append_anom(anoms, f"psf2_modeled_gt_input({n_mod}>{n_input})")

    usable = int(r["usable"] or 0)
    reason = str(r["reason"] or "")
    if usable == 1 and n_mod < 25:
        _append_anom(anoms, "psf2_usable_but_too_few_modeled")
    if usable == 0 and reason == "ok":
        _append_anom(anoms, "psf2_not_usable_but_reason_ok")


# -----------------------------
# Performance helpers
# -----------------------------

def _module_duration(conn: sqlite3.Connection, image_id: int, module_name: str) -> Optional[int]:
    if not _table_exists(conn, "module_runs"):
        return None
    r = conn.execute(
        """
        SELECT duration_ms
        FROM module_runs
        WHERE image_id=?
          AND module_name=?
        ORDER BY rowid DESC
        LIMIT 1
        """,
        (image_id, module_name),
    ).fetchone()
    if not r:
        return None
    try:
        if r["duration_ms"] is None:
            return None
        return int(r["duration_ms"])
    except Exception:
        return None


def _export_perf_csv_from_views(conn: sqlite3.Connection, out_csv: Path, summary: Dict[str, Any]) -> None:
    """
    Export two sections into one CSV:
      1) v_perf_module_rollup
      2) v_perf_total_rollup
    Also sanity-checks that the views exist + return rows.
    """
    views = ["v_perf_module_rollup", "v_perf_total_rollup"]
    summary.setdefault("views_present", {})
    for v in views:
        summary["views_present"][v] = _view_exists(conn, v)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)

        # Section 1
        w.writerow(["section", "v_perf_module_rollup"])
        if not _view_exists(conn, "v_perf_module_rollup"):
            w.writerow(["error", "view_missing"])
        else:
            rows = conn.execute("SELECT * FROM v_perf_module_rollup;").fetchall()
            if not rows:
                w.writerow(["warning", "no_rows"])
            else:
                cols = list(rows[0].keys())
                w.writerow(cols)
                for r in rows:
                    w.writerow([r[c] for c in cols])

        w.writerow([])

        # Section 2
        w.writerow(["section", "v_perf_total_rollup"])
        if not _view_exists(conn, "v_perf_total_rollup"):
            w.writerow(["error", "view_missing"])
        else:
            rows = conn.execute("SELECT * FROM v_perf_total_rollup;").fetchall()
            if not rows:
                w.writerow(["warning", "no_rows"])
            else:
                cols = list(rows[0].keys())
                w.writerow(cols)
                for r in rows:
                    w.writerow([r[c] for c in cols])


def _check_perf_views(conn: sqlite3.Connection, summary: Dict[str, Any]) -> None:
    """
    Add view-related anomalies into summary['anomaly_counts'] if anything looks wrong.
    We keep this lightweight: existence + basic duration sanity.
    """
    def bump(msg: str) -> None:
        summary["anomaly_counts"][msg] = summary["anomaly_counts"].get(msg, 0) + 1

    if not _view_exists(conn, "v_perf_module_rollup"):
        bump("view_missing:v_perf_module_rollup")
        return

    # Basic checks: non-negative durations where present
    try:
        rows = conn.execute("SELECT * FROM v_perf_module_rollup LIMIT 500;").fetchall()
        for r in rows:
            # "avg_ms" name depends on your view; tolerate variants.
            for key in ("avg_ms", "avg_duration_ms", "avg_processing_ms"):
                if key in r.keys():
                    v = _safe_float(r[key])
                    if v is not None and v < 0:
                        bump("v_perf_module_rollup_negative_avg_ms")
                    break
    except Exception:
        bump("v_perf_module_rollup_query_failed")

    if not _view_exists(conn, "v_perf_total_rollup"):
        bump("view_missing:v_perf_total_rollup")
        return

    try:
        rows = conn.execute("SELECT * FROM v_perf_total_rollup LIMIT 500;").fetchall()
        for r in rows:
            for key in ("avg_total_ms", "avg_ms", "avg_duration_ms", "avg_processing_ms"):
                if key in r.keys():
                    v = _safe_float(r[key])
                    if v is not None and v < 0:
                        bump("v_perf_total_rollup_negative_avg_total_ms")
                    break
    except Exception:
        bump("v_perf_total_rollup_query_failed")


# -----------------------------
# Main report generation
# -----------------------------

def run(db_path: Path, out_csv: Optional[Path], out_json: Optional[Path], out_perf_csv: Optional[Path]) -> int:
    conn = _connect(db_path)

    stamp = _now_stamp()
    out_csv = out_csv or (Path("data") / f"sanity_report_{stamp}.csv")
    out_json = out_json or (Path("data") / f"sanity_summary_{stamp}.json")
    out_perf_csv = out_perf_csv or (Path("data") / f"sanity_performance_{stamp}.csv")

    images = _fetch_images(conn)
    mod_ok = _module_run_ok_map(conn)

    summary: Dict[str, Any] = {
        "db_path": str(db_path),
        "n_images": len(images),
        "status_counts": {},
        "imagetyp_counts": {},
        "tables_present": {},
        "views_present": {},
        "anomaly_counts": {},
        "outputs": {
            "report_csv": str(out_csv),
            "summary_json": str(out_json),
            "performance_csv": str(out_perf_csv),
        },
    }

    for t in [
        "images",
        "fits_header_core",
        "sky_basic_metrics",
        "sky_background2d_metrics",
        "exposure_advice",
        "psf_detect_metrics",
        "psf_basic_metrics",
        "psf_grid_metrics",
        "psf_model_metrics",
        "module_runs",
        "flat_metrics",
        "flat_profiles",
        "flat_capture_sets",
        "flat_frame_links",
    ]:
        summary["tables_present"][t] = _table_exists(conn, t)

    # View checks + view-export
    _check_perf_views(conn, summary)
    _export_perf_csv_from_views(conn, out_perf_csv, summary)

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

        row.fits_ok = mod_ok.get((image_id, "fits_header_worker"), 0)

        is_flat = (imagetyp == "FLAT")
        is_light = (imagetyp == "LIGHT" or imagetyp == "")

        def set_flag(attr: str, applicable: bool, module_name: str) -> None:
            if not applicable:
                setattr(row, attr, "")
                return
            setattr(row, attr, "1" if mod_ok.get((image_id, module_name), 0) == 1 else "0")

        # flags
        set_flag("flat_group_ok", is_flat, "flat_group_worker")
        set_flag("flat_basic_ok", is_flat, "flat_basic_worker")

        set_flag("sky_basic_ok", (not is_flat), "sky_basic_worker")
        set_flag("sky_bkg2d_ok", (not is_flat), "sky_background2d_worker")
        set_flag("exposure_ok", (not is_flat), "exposure_advice_worker")

        set_flag("psf0_ok", (not is_flat), "psf_detect_worker")
        set_flag("psf1_ok", (not is_flat), "psf_basic_worker")
        set_flag("psf_grid_ok", (not is_flat), "psf_grid_worker")
        set_flag("psf2_ok", (not is_flat), "psf_model_worker")

        # checks
        _check_fits_header(row, anoms)

        if not is_flat:
            _check_sky_basic(conn, row, anoms)
            _check_sky_bkg2d(conn, row, anoms)
            _check_psf0(conn, row, anoms)
            _check_psf1(conn, row, anoms)
            _check_psf_grid(conn, row, anoms)
            _check_psf2(conn, row, anoms)
        else:
            # For flats, we mostly just want to know the flat tables got rows
            if _table_exists(conn, "flat_metrics"):
                fr = conn.execute("SELECT 1 FROM flat_metrics WHERE image_id=?", (image_id,)).fetchone()
                if not fr and row.flat_basic_ok == "1":
                    _append_anom(anoms, "flat_metrics_missing_row")

        # per-image durations (from module_runs)
        row.fits_duration_ms = _module_duration(conn, image_id, "fits_header_worker")
        row.sky_basic_duration_ms = _module_duration(conn, image_id, "sky_basic_worker")
        row.sky_bkg2d_duration_ms = _module_duration(conn, image_id, "sky_background2d_worker")
        row.exposure_duration_ms = _module_duration(conn, image_id, "exposure_advice_worker")
        row.flat_group_duration_ms = _module_duration(conn, image_id, "flat_group_worker")
        row.flat_basic_duration_ms = _module_duration(conn, image_id, "flat_basic_worker")
        row.psf0_duration_ms = _module_duration(conn, image_id, "psf_detect_worker")
        row.psf1_duration_ms = _module_duration(conn, image_id, "psf_basic_worker")
        row.psf_grid_duration_ms = _module_duration(conn, image_id, "psf_grid_worker")
        row.psf2_duration_ms = _module_duration(conn, image_id, "psf_model_worker")

        if row.psf2_duration_ms is not None and row.psf2_n_modeled and row.psf2_n_modeled > 0:
            row.psf2_ms_per_star = float(row.psf2_duration_ms) / float(row.psf2_n_modeled)

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
            "psf0_ok", "psf1_ok", "psf_grid_ok", "psf2_ok",
            "flat_group_ok", "flat_basic_ok",

            "sky_roi_median_adu", "sky_roi_madstd_adu",

            "psf0_n_peaks_good",
            "psf1_n_measured",

            "psf_grid_n_input_stars",
            "psf_grid_n_cells_with_data",
            "psf_grid_center_fwhm_px",
            "psf_grid_corner_fwhm_px_median",
            "psf_grid_center_to_corner_ratio",

            "psf2_n_modeled",
            "psf2_gauss_fwhm_med", "psf2_gauss_fwhm_p90", "psf2_gauss_ecc_med",

            # durations
            "fits_duration_ms",
            "sky_basic_duration_ms",
            "sky_bkg2d_duration_ms",
            "exposure_duration_ms",
            "flat_group_duration_ms",
            "flat_basic_duration_ms",
            "psf0_duration_ms",
            "psf1_duration_ms",
            "psf_grid_duration_ms",
            "psf2_duration_ms",
            "psf2_ms_per_star",

            "anomalies",
        ])
        for r in rows_out:
            w.writerow([
                r.image_id, r.file_name, r.status, r.imagetyp, r.naxis1, r.naxis2,
                r.fits_ok,
                r.sky_basic_ok, r.sky_bkg2d_ok, r.exposure_ok,
                r.psf0_ok, r.psf1_ok, r.psf_grid_ok, r.psf2_ok,
                r.flat_group_ok, r.flat_basic_ok,

                r.sky_roi_median_adu, r.sky_roi_madstd_adu,

                r.psf0_n_peaks_good,
                r.psf1_n_measured,

                r.psf_grid_n_input_stars,
                r.psf_grid_n_cells_with_data,
                r.psf_grid_center_fwhm_px,
                r.psf_grid_corner_fwhm_px_median,
                r.psf_grid_center_to_corner_ratio,

                r.psf2_n_modeled,
                r.psf2_gauss_fwhm_med, r.psf2_gauss_fwhm_p90, r.psf2_gauss_ecc_med,

                r.fits_duration_ms,
                r.sky_basic_duration_ms,
                r.sky_bkg2d_duration_ms,
                r.exposure_duration_ms,
                r.flat_group_duration_ms,
                r.flat_basic_duration_ms,
                r.psf0_duration_ms,
                r.psf1_duration_ms,
                r.psf_grid_duration_ms,
                r.psf2_duration_ms,
                r.psf2_ms_per_star,

                r.anomalies,
            ])

    # Write JSON summary
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    n_anom_total = sum(int(v) for v in summary["anomaly_counts"].values())
    print(f"[sanity_check] images={summary['n_images']} total_anomalies={n_anom_total}")
    print(f"[sanity_check] report={out_csv}")
    print(f"[sanity_check] summary={out_json}")
    print(f"[sanity_check] performance={out_perf_csv}")
    if summary["anomaly_counts"]:
        top = sorted(summary["anomaly_counts"].items(), key=lambda kv: kv[1], reverse=True)[:10]
        for k, v in top:
            print(f"  {k}: {v}")

    conn.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Centaur sanity checker (DB -> CSV + JSON + performance CSV).")
    ap.add_argument("--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB")

    # Optional: if omitted, we auto-name with timestamp.
    ap.add_argument("--out", type=str, default="", help="Output CSV path (blank = auto timestamp)")
    ap.add_argument("--summary", type=str, default="", help="Output JSON summary path (blank = auto timestamp)")
    ap.add_argument("--perf", type=str, default="", help="Output performance CSV path (blank = auto timestamp)")

    args = ap.parse_args()

    out_csv = Path(args.out) if args.out.strip() else None
    out_json = Path(args.summary) if args.summary.strip() else None
    out_perf = Path(args.perf) if args.perf.strip() else None

    return run(Path(args.db), out_csv, out_json, out_perf)


if __name__ == "__main__":
    raise SystemExit(main())
