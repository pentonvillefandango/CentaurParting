# ---------------------------------------------
# how to run
# python3 centaur/sanity_check.py --db data/centaurparting.db
#
# (optional explicit paths)
# python3 centaur/sanity_check.py --db data/centaurparting.db \
#   --out data/sanity_report.csv \
#   --summary data/sanity_summary.json \
#   --perf data/sanity_performance.csv \
#   --runs data/sanity_module_runs_checks.csv
#
# outputs (default if you omit --out/--summary/--perf/--runs):
#   data/sanity_report_YYYYMMDD_HHMMSS.csv
#   data/sanity_summary_YYYYMMDD_HHMMSS.json
#   data/sanity_performance_YYYYMMDD_HHMMSS.csv
#   data/sanity_module_runs_checks_YYYYMMDD_HHMMSS.csv
# ---------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
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


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _clamp01(v: Optional[float]) -> bool:
    if v is None:
        return True
    return 0.0 <= v <= 1.0


def _append_anom(anoms: List[str], msg: str) -> None:
    if msg and msg not in anoms:
        anoms.append(msg)


def _parse_iso_utc(s: Any) -> Optional[datetime]:
    """
    Best-effort parse of ISO timestamps we write (with timezone).
    Returns aware datetime in UTC, or None.
    """
    if s is None:
        return None
    if not isinstance(s, str):
        return None
    ss = s.strip()
    if not ss:
        return None
    try:
        dt = datetime.fromisoformat(ss)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


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

    signal_structure_ok: str = ""
    saturation_ok: str = ""
    roi_signal_ok: str = ""

    nebula_mask_ok: str = ""
    masked_signal_ok: str = ""
    star_headroom_ok: str = ""

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

    # NEW: key rollups for today’s 4 models (+ saturation/roi which already existed)
    sat_frac: Optional[float] = None
    sat_max_pixel_adu: Optional[float] = None
    sat_saturation_adu: Optional[float] = None

    roi_signal_adu_s: Optional[float] = None

    struct_p99m_p50_adu_s: Optional[float] = None

    nebula_mask_frac: Optional[float] = None

    masked_signal_adu_s: Optional[float] = None
    masked_snr: Optional[float] = None

    headroom_p90: Optional[float] = None
    headroom_p99: Optional[float] = None
    sat_star_frac: Optional[float] = None
    n_stars_used: Optional[int] = None
    star_peak_p90_adu: Optional[float] = None
    saturation_adu_used: Optional[float] = None

    # performance (from module_runs) (ms)
    fits_duration_ms: Optional[int] = None
    sky_basic_duration_ms: Optional[int] = None
    sky_bkg2d_duration_ms: Optional[int] = None
    signal_structure_duration_ms: Optional[int] = None
    saturation_duration_ms: Optional[int] = None
    roi_signal_duration_ms: Optional[int] = None
    nebula_mask_duration_ms: Optional[int] = None
    masked_signal_duration_ms: Optional[int] = None
    star_headroom_duration_ms: Optional[int] = None

    exposure_duration_ms: Optional[int] = None
    flat_group_duration_ms: Optional[int] = None
    flat_basic_duration_ms: Optional[int] = None
    psf0_duration_ms: Optional[int] = None
    psf1_duration_ms: Optional[int] = None
    psf_grid_duration_ms: Optional[int] = None
    psf2_duration_ms: Optional[int] = None

    # NEW: duration_us (pipeline-owned)
    fits_duration_us: Optional[int] = None
    sky_basic_duration_us: Optional[int] = None
    sky_bkg2d_duration_us: Optional[int] = None
    signal_structure_duration_us: Optional[int] = None
    saturation_duration_us: Optional[int] = None
    roi_signal_duration_us: Optional[int] = None
    nebula_mask_duration_us: Optional[int] = None
    masked_signal_duration_us: Optional[int] = None
    star_headroom_duration_us: Optional[int] = None

    exposure_duration_us: Optional[int] = None
    flat_group_duration_us: Optional[int] = None
    flat_basic_duration_us: Optional[int] = None
    psf0_duration_us: Optional[int] = None
    psf1_duration_us: Optional[int] = None
    psf_grid_duration_us: Optional[int] = None
    psf2_duration_us: Optional[int] = None

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

    nanf = _safe_float(r["nan_fraction"]) if "nan_fraction" in r.keys() else None
    inff = _safe_float(r["inf_fraction"]) if "inf_fraction" in r.keys() else None
    if not _clamp01(nanf):
        _append_anom(anoms, "sky_basic_nan_fraction_out_of_range")
    if not _clamp01(inff):
        _append_anom(anoms, "sky_basic_inf_fraction_out_of_range")

    row.sky_roi_median_adu = (
        _safe_float(r["roi_median_adu"]) if "roi_median_adu" in r.keys() else None
    )
    row.sky_roi_madstd_adu = (
        _safe_float(r["roi_madstd_adu"]) if "roi_madstd_adu" in r.keys() else None
    )

    vmin = _safe_float(r["roi_min_adu"]) if "roi_min_adu" in r.keys() else None
    vmax = _safe_float(r["roi_max_adu"]) if "roi_max_adu" in r.keys() else None
    vmed = row.sky_roi_median_adu
    if vmin is not None and vmax is not None and vmin > vmax:
        _append_anom(anoms, "sky_basic_roi_min_gt_max")
    if vmin is not None and vmax is not None and vmed is not None:
        if not (vmin <= vmed <= vmax):
            _append_anom(anoms, "sky_basic_roi_median_outside_minmax")

    cf_ff = (
        _safe_float(r["clipped_fraction_ff"])
        if "clipped_fraction_ff" in r.keys()
        else None
    )
    cf_roi = (
        _safe_float(r["clipped_fraction_roi"])
        if "clipped_fraction_roi" in r.keys()
        else None
    )
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


def _check_signal_structure(
    conn: sqlite3.Connection, row: ImageRow, anoms: List[str]
) -> None:
    if not _table_exists(conn, "signal_structure_metrics"):
        return
    r = conn.execute(
        "SELECT * FROM signal_structure_metrics WHERE image_id=?",
        (row.image_id,),
    ).fetchone()
    if not r:
        _append_anom(anoms, "signal_structure_missing_row")
        return

    # do NOT assume usable column exists (you hit that earlier in queries)
    if "expected_fields" in r.keys() and "written_fields" in r.keys():
        ef = int(r["expected_fields"] or 0)
        wf = int(r["written_fields"] or 0)
        if ef > 0 and wf == 0:
            _append_anom(anoms, "signal_structure_wrote_zero_fields")

    # capture the main “structure per second” if present
    if "ff_p99_minus_p50_adu_s" in r.keys():
        row.struct_p99m_p50_adu_s = _safe_float(r["ff_p99_minus_p50_adu_s"])
        if row.struct_p99m_p50_adu_s is not None and row.struct_p99m_p50_adu_s < 0:
            _append_anom(anoms, "signal_structure_negative_struct_s")


def _check_saturation(
    conn: sqlite3.Connection, row: ImageRow, anoms: List[str]
) -> None:
    if not _table_exists(conn, "saturation_metrics"):
        return
    r = conn.execute(
        "SELECT * FROM saturation_metrics WHERE image_id=?",
        (row.image_id,),
    ).fetchone()
    if not r:
        _append_anom(anoms, "saturation_missing_row")
        return

    # sat_frac should be [0..1] if present
    if "saturated_pixel_fraction" in r.keys():
        row.sat_frac = _safe_float(r["saturated_pixel_fraction"])
        if not _clamp01(row.sat_frac):
            _append_anom(anoms, "saturation_sat_frac_out_of_range")

    row.sat_max_pixel_adu = (
        _safe_float(r["max_pixel_adu"]) if "max_pixel_adu" in r.keys() else None
    )
    row.sat_saturation_adu = (
        _safe_float(r["saturation_adu"]) if "saturation_adu" in r.keys() else None
    )

    if row.sat_saturation_adu is not None and row.sat_max_pixel_adu is not None:
        if row.sat_max_pixel_adu > row.sat_saturation_adu + 1e-6:
            _append_anom(anoms, "saturation_max_pixel_gt_saturation_adu")


def _check_roi_signal(
    conn: sqlite3.Connection, row: ImageRow, anoms: List[str]
) -> None:
    if not _table_exists(conn, "roi_signal_metrics"):
        return
    r = conn.execute(
        "SELECT * FROM roi_signal_metrics WHERE image_id=?",
        (row.image_id,),
    ).fetchone()
    if not r:
        _append_anom(anoms, "roi_signal_missing_row")
        return

    if "obj_minus_bg_adu_s" in r.keys():
        row.roi_signal_adu_s = _safe_float(r["obj_minus_bg_adu_s"])


def _check_nebula_mask(
    conn: sqlite3.Connection, row: ImageRow, anoms: List[str]
) -> None:
    if not _table_exists(conn, "nebula_mask_metrics"):
        return
    r = conn.execute(
        "SELECT * FROM nebula_mask_metrics WHERE image_id=?",
        (row.image_id,),
    ).fetchone()
    if not r:
        _append_anom(anoms, "nebula_mask_missing_row")
        return

    if "mask_coverage_frac" in r.keys():
        row.nebula_mask_frac = _safe_float(r["mask_coverage_frac"])
        if not _clamp01(row.nebula_mask_frac):
            _append_anom(anoms, "nebula_mask_frac_out_of_range")

    if "nan_fraction" in r.keys() and not _clamp01(_safe_float(r["nan_fraction"])):
        _append_anom(anoms, "nebula_mask_nan_fraction_out_of_range")
    if "inf_fraction" in r.keys() and not _clamp01(_safe_float(r["inf_fraction"])):
        _append_anom(anoms, "nebula_mask_inf_fraction_out_of_range")

    if "usable" in r.keys():
        usable = int(r["usable"] or 0)
        reason = str(r["reason"] or "")
        if usable == 0 and reason == "ok":
            _append_anom(anoms, "nebula_mask_not_usable_but_reason_ok")


def _check_masked_signal(
    conn: sqlite3.Connection, row: ImageRow, anoms: List[str]
) -> None:
    if not _table_exists(conn, "masked_signal_metrics"):
        return
    r = conn.execute(
        "SELECT * FROM masked_signal_metrics WHERE image_id=?",
        (row.image_id,),
    ).fetchone()
    if not r:
        _append_anom(anoms, "masked_signal_missing_row")
        return

    if "masked_signal_adu_s" in r.keys():
        row.masked_signal_adu_s = _safe_float(r["masked_signal_adu_s"])
    if "masked_snr" in r.keys():
        row.masked_snr = _safe_float(r["masked_snr"])
        if row.masked_snr is not None and row.masked_snr <= 0:
            _append_anom(anoms, "masked_signal_nonpositive_snr")

    if "nebula_frac" in r.keys():
        nf = _safe_float(r["nebula_frac"])
        if not _clamp01(nf):
            _append_anom(anoms, "masked_signal_nebula_frac_out_of_range")

    if "usable" in r.keys():
        usable = int(r["usable"] or 0)
        reason = str(r["reason"] or "")
        if usable == 0 and reason == "ok":
            _append_anom(anoms, "masked_signal_not_usable_but_reason_ok")


def _check_star_headroom(
    conn: sqlite3.Connection, row: ImageRow, anoms: List[str]
) -> None:
    if not _table_exists(conn, "star_headroom_metrics"):
        return
    r = conn.execute(
        "SELECT * FROM star_headroom_metrics WHERE image_id=?",
        (row.image_id,),
    ).fetchone()
    if not r:
        _append_anom(anoms, "star_headroom_missing_row")
        return

    row.n_stars_used = (
        _safe_int(r["n_stars_used"]) if "n_stars_used" in r.keys() else None
    )
    row.star_peak_p90_adu = (
        _safe_float(r["star_peak_p90_adu"]) if "star_peak_p90_adu" in r.keys() else None
    )

    # optional columns (depending on schema revisions)
    if "headroom_p90" in r.keys():
        row.headroom_p90 = _safe_float(r["headroom_p90"])
        if not _clamp01(row.headroom_p90):
            _append_anom(anoms, "star_headroom_headroom_p90_out_of_range")
    if "headroom_p99" in r.keys():
        row.headroom_p99 = _safe_float(r["headroom_p99"])
        if not _clamp01(row.headroom_p99):
            _append_anom(anoms, "star_headroom_headroom_p99_out_of_range")
    if "saturated_star_fraction" in r.keys():
        row.sat_star_frac = _safe_float(r["saturated_star_fraction"])
        if not _clamp01(row.sat_star_frac):
            _append_anom(anoms, "star_headroom_sat_star_frac_out_of_range")

    # your worker writes these names (current)
    if "saturation_adu_used" in r.keys():
        row.saturation_adu_used = _safe_float(r["saturation_adu_used"])
    elif "saturation_adu" in r.keys():
        row.saturation_adu_used = _safe_float(r["saturation_adu"])

    if row.n_stars_used is not None and row.n_stars_used < 0:
        _append_anom(anoms, "star_headroom_negative_n_stars_used")

    if "usable" in r.keys():
        usable = int(r["usable"] or 0)
        reason = str(r["reason"] or "")
        if usable == 1 and (row.n_stars_used is not None) and row.n_stars_used <= 0:
            _append_anom(anoms, "star_headroom_usable_but_zero_stars")
        if usable == 0 and reason == "ok":
            _append_anom(anoms, "star_headroom_not_usable_but_reason_ok")


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

    cell_counts = [
        int(r["cell_r0c0_n"] or 0),
        int(r["cell_r0c1_n"] or 0),
        int(r["cell_r0c2_n"] or 0),
        int(r["cell_r1c0_n"] or 0),
        int(r["cell_r1c1_n"] or 0),
        int(r["cell_r1c2_n"] or 0),
        int(r["cell_r2c0_n"] or 0),
        int(r["cell_r2c1_n"] or 0),
        int(r["cell_r2c2_n"] or 0),
    ]
    sum_cells = sum(cell_counts)
    if n_input > 0 and sum_cells > n_input:
        _append_anom(anoms, f"psf_grid_sum_cell_n_gt_input({sum_cells}>{n_input})")

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


def _module_duration_ms(
    conn: sqlite3.Connection, image_id: int, module_name: str
) -> Optional[int]:
    """
    Backward-compatible: prefer duration_us if present, else duration_ms.
    """
    if not _table_exists(conn, "module_runs"):
        return None

    has_us = _col_exists(conn, "module_runs", "duration_us")
    cols = "duration_us, duration_ms" if has_us else "duration_ms"

    r = conn.execute(
        f"""
        SELECT {cols}
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

    if has_us:
        dus = _safe_int(r["duration_us"])
        if dus is not None:
            return int(dus // 1000)

    return _safe_int(r["duration_ms"])


def _module_duration_us(
    conn: sqlite3.Connection, image_id: int, module_name: str
) -> Optional[int]:
    if not _table_exists(conn, "module_runs"):
        return None
    if not _col_exists(conn, "module_runs", "duration_us"):
        return None
    r = conn.execute(
        """
        SELECT duration_us
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
    return _safe_int(r["duration_us"])


def _export_perf_csv_from_views(
    conn: sqlite3.Connection, out_csv: Path, summary: Dict[str, Any]
) -> None:
    """
    Export two sections into one CSV:
      1) v_perf_module_rollup
      2) v_perf_total_rollup
    """
    views = ["v_perf_module_rollup", "v_perf_total_rollup"]
    summary.setdefault("views_present", {})
    for v in views:
        summary["views_present"][v] = _view_exists(conn, v)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)

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
    def bump(msg: str) -> None:
        summary["anomaly_counts"][msg] = summary["anomaly_counts"].get(msg, 0) + 1

    if not _view_exists(conn, "v_perf_module_rollup"):
        bump("view_missing:v_perf_module_rollup")
        return

    try:
        rows = conn.execute("SELECT * FROM v_perf_module_rollup LIMIT 500;").fetchall()
        for r in rows:
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
            for key in (
                "avg_total_ms",
                "avg_ms",
                "avg_duration_ms",
                "avg_processing_ms",
            ):
                if key in r.keys():
                    v = _safe_float(r[key])
                    if v is not None and v < 0:
                        bump("v_perf_total_rollup_negative_avg_total_ms")
                    break
    except Exception:
        bump("v_perf_total_rollup_query_failed")


# -----------------------------
# module_runs + routing checks (pipeline-owned)
# -----------------------------


def _write_module_runs_checks_csv(
    conn: sqlite3.Connection, out_csv: Path, summary: Dict[str, Any]
) -> None:
    summary.setdefault("module_runs_checks", {})
    summary["module_runs_checks"]["enabled"] = False

    if not _table_exists(conn, "module_runs") or not _table_exists(
        conn, "fits_header_core"
    ):
        summary["module_runs_checks"][
            "error"
        ] = "missing_table:module_runs_or_fits_header_core"
        return

    has_duration_us = _col_exists(conn, "module_runs", "duration_us")
    summary["module_runs_checks"]["has_duration_us"] = bool(has_duration_us)

    rows = conn.execute(
        """
        SELECT
          mr.image_id,
          mr.module_name,
          mr.status,
          mr.expected_fields,
          mr.read_fields,
          mr.written_fields,
          mr.started_utc,
          mr.ended_utc,
          mr.duration_us,
          mr.duration_ms,
          mr.db_written_utc,
          UPPER(COALESCE(TRIM(f.imagetyp), '')) AS imagetyp
        FROM module_runs mr
        JOIN fits_header_core f USING(image_id)
        ORDER BY mr.image_id, mr.started_utc, mr.module_name;
        """
    ).fetchall()

    # Current pipeline routing (LIGHT)
    light_modules = [
        "fits_header_worker",
        "sky_basic_worker",
        "sky_background2d_worker",
        "nebula_mask_worker",
        "saturation_worker",
        "signal_structure_worker",
        "roi_signal_worker",
        "exposure_advice_worker",
        "psf_detect_worker",
        "psf_basic_worker",
        "star_headroom_worker",
        "psf_grid_worker",
        "psf_model_worker",
        "masked_signal_worker",
    ]
    flat_modules = [
        "fits_header_worker",
        "flat_group_worker",
        "flat_basic_worker",
    ]

    def expected_modules_for(imagetyp: str) -> List[str]:
        if imagetyp == "FLAT":
            return flat_modules
        return light_modules

    violations: List[Dict[str, Any]] = []
    by_image: Dict[int, Dict[str, Any]] = {}
    dup_seen: Dict[Tuple[int, str], int] = {}

    def add_violation(kind: str, image_id: int, module_name: str, detail: str) -> None:
        violations.append(
            {
                "kind": kind,
                "image_id": image_id,
                "module_name": module_name,
                "detail": detail,
            }
        )

    for r in rows:
        image_id = int(r["image_id"])
        module_name = str(r["module_name"])
        imagetyp = str(r["imagetyp"] or "")

        k = (image_id, module_name)
        dup_seen[k] = dup_seen.get(k, 0) + 1
        if dup_seen[k] == 2:
            add_violation(
                "duplicate_image_module",
                image_id,
                module_name,
                "COUNT(image_id,module_name) > 1",
            )

        started = _parse_iso_utc(r["started_utc"])
        ended = _parse_iso_utc(r["ended_utc"])
        dbw = _parse_iso_utc(r["db_written_utc"])

        dur_us = _safe_int(r["duration_us"])
        dur_ms = _safe_int(r["duration_ms"])

        if has_duration_us:
            if dur_us is None:
                add_violation(
                    "missing_duration_us", image_id, module_name, "duration_us IS NULL"
                )
            elif dur_us <= 0:
                add_violation(
                    "nonpositive_duration_us",
                    image_id,
                    module_name,
                    f"duration_us={dur_us}",
                )
        else:
            if dur_ms is None:
                add_violation(
                    "missing_duration_ms", image_id, module_name, "duration_ms IS NULL"
                )

        if started is None or ended is None:
            add_violation(
                "bad_timestamp_parse",
                image_id,
                module_name,
                "started_utc/ended_utc not parseable",
            )
        else:
            if ended < started:
                add_violation(
                    "ended_before_started",
                    image_id,
                    module_name,
                    "ended_utc < started_utc",
                )
            if dur_us is not None:
                wall_us = int((ended - started).total_seconds() * 1_000_000)
                diff_us = abs(int(dur_us) - int(wall_us))
                if diff_us > 250_000:
                    add_violation(
                        "duration_wall_mismatch_over_250ms",
                        image_id,
                        module_name,
                        f"diff_us={diff_us}",
                    )

        if ended is not None and dbw is not None:
            if dbw < ended:
                add_violation(
                    "db_written_before_ended",
                    image_id,
                    module_name,
                    "db_written_utc < ended_utc",
                )
            else:
                lag_us = int((dbw - ended).total_seconds() * 1_000_000)
                if lag_us > 1_000_000:
                    add_violation(
                        "db_write_lag_over_1s",
                        image_id,
                        module_name,
                        f"lag_us={lag_us}",
                    )

        if image_id not in by_image:
            by_image[image_id] = {"imagetyp": imagetyp, "modules": []}
        by_image[image_id]["modules"].append(module_name)

    for image_id, info in by_image.items():
        imagetyp = str(info["imagetyp"] or "")
        mods = list(info["modules"])
        expected = expected_modules_for(imagetyp)

        mods_set = set(mods)
        exp_set = set(expected)

        missing = [m for m in expected if m not in mods_set]
        extra = [m for m in sorted(mods_set) if m not in exp_set]

        if imagetyp == "FLAT":
            if len(mods) != len(flat_modules):
                add_violation(
                    "flat_wrong_module_count",
                    image_id,
                    "(image)",
                    f"n_runs={len(mods)} expected={len(flat_modules)}",
                )
        else:
            if len(mods) != len(light_modules):
                add_violation(
                    "light_wrong_module_count",
                    image_id,
                    "(image)",
                    f"n_runs={len(mods)} expected={len(light_modules)}",
                )

        if missing:
            add_violation(
                "missing_expected_modules", image_id, "(image)", ",".join(missing)
            )
        if extra:
            add_violation(
                "unexpected_modules_present", image_id, "(image)", ",".join(extra)
            )

    counts: Dict[str, int] = {}
    for v in violations:
        counts[v["kind"]] = counts.get(v["kind"], 0) + 1

    summary["module_runs_checks"]["enabled"] = True
    summary["module_runs_checks"]["n_rows_checked"] = int(len(rows))
    summary["module_runs_checks"]["n_images_checked"] = int(len(by_image))
    summary["module_runs_checks"]["violation_counts"] = counts
    summary["module_runs_checks"]["total_violations"] = int(len(violations))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "module_runs_checks"])
        w.writerow(["n_rows_checked", len(rows)])
        w.writerow(["n_images_checked", len(by_image)])
        w.writerow(["has_duration_us", int(has_duration_us)])
        w.writerow(["total_violations", len(violations)])
        w.writerow([])

        w.writerow(["section", "violation_counts"])
        w.writerow(["kind", "count"])
        for kind, cnt in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            w.writerow([kind, cnt])
        w.writerow([])

        w.writerow(["section", "violations"])
        w.writerow(["kind", "image_id", "module_name", "detail"])
        for v in violations:
            w.writerow([v["kind"], v["image_id"], v["module_name"], v["detail"]])


# -----------------------------
# Main report generation
# -----------------------------


def run(
    db_path: Path,
    out_csv: Optional[Path],
    out_json: Optional[Path],
    out_perf_csv: Optional[Path],
    out_runs_csv: Optional[Path],
) -> int:
    conn = _connect(db_path)

    stamp = _now_stamp()
    out_csv = out_csv or (Path("data") / f"sanity_report_{stamp}.csv")
    out_json = out_json or (Path("data") / f"sanity_summary_{stamp}.json")
    out_perf_csv = out_perf_csv or (Path("data") / f"sanity_performance_{stamp}.csv")
    out_runs_csv = out_runs_csv or (
        Path("data") / f"sanity_module_runs_checks_{stamp}.csv"
    )

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
            "module_runs_checks_csv": str(out_runs_csv),
        },
    }

    for t in [
        "images",
        "fits_header_core",
        "sky_basic_metrics",
        "sky_background2d_metrics",
        "signal_structure_metrics",
        "saturation_metrics",
        "roi_signal_metrics",
        "nebula_mask_metrics",
        "masked_signal_metrics",
        "star_headroom_metrics",
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

    _check_perf_views(conn, summary)
    _export_perf_csv_from_views(conn, out_perf_csv, summary)
    _write_module_runs_checks_csv(conn, out_runs_csv, summary)

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
        summary["imagetyp_counts"][imagetyp] = (
            summary["imagetyp_counts"].get(imagetyp, 0) + 1
        )

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

        is_flat = imagetyp == "FLAT"
        is_lightish = not is_flat  # pipeline treats everything non-FLAT as LIGHT route

        def set_flag(attr: str, applicable: bool, module_name: str) -> None:
            if not applicable:
                setattr(row, attr, "")
                return
            setattr(
                row, attr, "1" if mod_ok.get((image_id, module_name), 0) == 1 else "0"
            )

        # flags
        set_flag("flat_group_ok", is_flat, "flat_group_worker")
        set_flag("flat_basic_ok", is_flat, "flat_basic_worker")

        set_flag("sky_basic_ok", is_lightish, "sky_basic_worker")
        set_flag("sky_bkg2d_ok", is_lightish, "sky_background2d_worker")

        set_flag("signal_structure_ok", is_lightish, "signal_structure_worker")
        set_flag("saturation_ok", is_lightish, "saturation_worker")
        set_flag("roi_signal_ok", is_lightish, "roi_signal_worker")

        set_flag("nebula_mask_ok", is_lightish, "nebula_mask_worker")
        set_flag("masked_signal_ok", is_lightish, "masked_signal_worker")
        set_flag("star_headroom_ok", is_lightish, "star_headroom_worker")

        set_flag("exposure_ok", is_lightish, "exposure_advice_worker")

        set_flag("psf0_ok", is_lightish, "psf_detect_worker")
        set_flag("psf1_ok", is_lightish, "psf_basic_worker")
        set_flag("psf_grid_ok", is_lightish, "psf_grid_worker")
        set_flag("psf2_ok", is_lightish, "psf_model_worker")

        # checks
        _check_fits_header(row, anoms)

        if is_lightish:
            _check_sky_basic(conn, row, anoms)
            _check_sky_bkg2d(conn, row, anoms)

            _check_signal_structure(conn, row, anoms)
            _check_saturation(conn, row, anoms)
            _check_roi_signal(conn, row, anoms)

            _check_nebula_mask(conn, row, anoms)
            _check_masked_signal(conn, row, anoms)

            _check_psf0(conn, row, anoms)
            _check_psf1(conn, row, anoms)
            _check_star_headroom(conn, row, anoms)
            _check_psf_grid(conn, row, anoms)
            _check_psf2(conn, row, anoms)
        else:
            if _table_exists(conn, "flat_metrics"):
                fr = conn.execute(
                    "SELECT 1 FROM flat_metrics WHERE image_id=?", (image_id,)
                ).fetchone()
                if not fr and row.flat_basic_ok == "1":
                    _append_anom(anoms, "flat_metrics_missing_row")

        # durations (ms)
        row.fits_duration_ms = _module_duration_ms(conn, image_id, "fits_header_worker")
        row.sky_basic_duration_ms = _module_duration_ms(
            conn, image_id, "sky_basic_worker"
        )
        row.sky_bkg2d_duration_ms = _module_duration_ms(
            conn, image_id, "sky_background2d_worker"
        )

        row.signal_structure_duration_ms = _module_duration_ms(
            conn, image_id, "signal_structure_worker"
        )
        row.saturation_duration_ms = _module_duration_ms(
            conn, image_id, "saturation_worker"
        )
        row.roi_signal_duration_ms = _module_duration_ms(
            conn, image_id, "roi_signal_worker"
        )
        row.nebula_mask_duration_ms = _module_duration_ms(
            conn, image_id, "nebula_mask_worker"
        )
        row.masked_signal_duration_ms = _module_duration_ms(
            conn, image_id, "masked_signal_worker"
        )
        row.star_headroom_duration_ms = _module_duration_ms(
            conn, image_id, "star_headroom_worker"
        )

        row.exposure_duration_ms = _module_duration_ms(
            conn, image_id, "exposure_advice_worker"
        )
        row.flat_group_duration_ms = _module_duration_ms(
            conn, image_id, "flat_group_worker"
        )
        row.flat_basic_duration_ms = _module_duration_ms(
            conn, image_id, "flat_basic_worker"
        )
        row.psf0_duration_ms = _module_duration_ms(conn, image_id, "psf_detect_worker")
        row.psf1_duration_ms = _module_duration_ms(conn, image_id, "psf_basic_worker")
        row.psf_grid_duration_ms = _module_duration_ms(
            conn, image_id, "psf_grid_worker"
        )
        row.psf2_duration_ms = _module_duration_ms(conn, image_id, "psf_model_worker")

        # durations (us)
        row.fits_duration_us = _module_duration_us(conn, image_id, "fits_header_worker")
        row.sky_basic_duration_us = _module_duration_us(
            conn, image_id, "sky_basic_worker"
        )
        row.sky_bkg2d_duration_us = _module_duration_us(
            conn, image_id, "sky_background2d_worker"
        )

        row.signal_structure_duration_us = _module_duration_us(
            conn, image_id, "signal_structure_worker"
        )
        row.saturation_duration_us = _module_duration_us(
            conn, image_id, "saturation_worker"
        )
        row.roi_signal_duration_us = _module_duration_us(
            conn, image_id, "roi_signal_worker"
        )
        row.nebula_mask_duration_us = _module_duration_us(
            conn, image_id, "nebula_mask_worker"
        )
        row.masked_signal_duration_us = _module_duration_us(
            conn, image_id, "masked_signal_worker"
        )
        row.star_headroom_duration_us = _module_duration_us(
            conn, image_id, "star_headroom_worker"
        )

        row.exposure_duration_us = _module_duration_us(
            conn, image_id, "exposure_advice_worker"
        )
        row.flat_group_duration_us = _module_duration_us(
            conn, image_id, "flat_group_worker"
        )
        row.flat_basic_duration_us = _module_duration_us(
            conn, image_id, "flat_basic_worker"
        )
        row.psf0_duration_us = _module_duration_us(conn, image_id, "psf_detect_worker")
        row.psf1_duration_us = _module_duration_us(conn, image_id, "psf_basic_worker")
        row.psf_grid_duration_us = _module_duration_us(
            conn, image_id, "psf_grid_worker"
        )
        row.psf2_duration_us = _module_duration_us(conn, image_id, "psf_model_worker")

        if (
            row.psf2_duration_ms is not None
            and row.psf2_n_modeled
            and row.psf2_n_modeled > 0
        ):
            row.psf2_ms_per_star = float(row.psf2_duration_ms) / float(
                row.psf2_n_modeled
            )

        row.anomalies = ";".join(anoms)
        for a in anoms:
            summary["anomaly_counts"][a] = summary["anomaly_counts"].get(a, 0) + 1

        rows_out.append(row)

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "image_id",
                "file_name",
                "status",
                "imagetyp",
                "naxis1",
                "naxis2",
                "fits_ok",
                "sky_basic_ok",
                "sky_background2d_ok",
                "signal_structure_ok",
                "saturation_ok",
                "roi_signal_ok",
                "nebula_mask_ok",
                "masked_signal_ok",
                "star_headroom_ok",
                "exposure_ok",
                "psf0_ok",
                "psf1_ok",
                "psf_grid_ok",
                "psf2_ok",
                "flat_group_ok",
                "flat_basic_ok",
                "sky_roi_median_adu",
                "sky_roi_madstd_adu",
                "sat_frac",
                "sat_max_pixel_adu",
                "sat_saturation_adu",
                "roi_signal_adu_s",
                "struct_p99m_p50_adu_s",
                "nebula_mask_frac",
                "masked_signal_adu_s",
                "masked_snr",
                "n_stars_used",
                "star_peak_p90_adu",
                "saturation_adu_used",
                "headroom_p90",
                "headroom_p99",
                "sat_star_frac",
                "psf0_n_peaks_good",
                "psf1_n_measured",
                "psf_grid_n_input_stars",
                "psf_grid_n_cells_with_data",
                "psf_grid_center_fwhm_px",
                "psf_grid_corner_fwhm_px_median",
                "psf_grid_center_to_corner_ratio",
                "psf2_n_modeled",
                "psf2_gauss_fwhm_med",
                "psf2_gauss_fwhm_p90",
                "psf2_gauss_ecc_med",
                # durations (ms)
                "fits_duration_ms",
                "sky_basic_duration_ms",
                "sky_bkg2d_duration_ms",
                "signal_structure_duration_ms",
                "saturation_duration_ms",
                "roi_signal_duration_ms",
                "nebula_mask_duration_ms",
                "masked_signal_duration_ms",
                "star_headroom_duration_ms",
                "exposure_duration_ms",
                "flat_group_duration_ms",
                "flat_basic_duration_ms",
                "psf0_duration_ms",
                "psf1_duration_ms",
                "psf_grid_duration_ms",
                "psf2_duration_ms",
                # durations (us)
                "fits_duration_us",
                "sky_basic_duration_us",
                "sky_bkg2d_duration_us",
                "signal_structure_duration_us",
                "saturation_duration_us",
                "roi_signal_duration_us",
                "nebula_mask_duration_us",
                "masked_signal_duration_us",
                "star_headroom_duration_us",
                "exposure_duration_us",
                "flat_group_duration_us",
                "flat_basic_duration_us",
                "psf0_duration_us",
                "psf1_duration_us",
                "psf_grid_duration_us",
                "psf2_duration_us",
                "psf2_ms_per_star",
                "anomalies",
            ]
        )
        for r in rows_out:
            w.writerow(
                [
                    r.image_id,
                    r.file_name,
                    r.status,
                    r.imagetyp,
                    r.naxis1,
                    r.naxis2,
                    r.fits_ok,
                    r.sky_basic_ok,
                    r.sky_bkg2d_ok,
                    r.signal_structure_ok,
                    r.saturation_ok,
                    r.roi_signal_ok,
                    r.nebula_mask_ok,
                    r.masked_signal_ok,
                    r.star_headroom_ok,
                    r.exposure_ok,
                    r.psf0_ok,
                    r.psf1_ok,
                    r.psf_grid_ok,
                    r.psf2_ok,
                    r.flat_group_ok,
                    r.flat_basic_ok,
                    r.sky_roi_median_adu,
                    r.sky_roi_madstd_adu,
                    r.sat_frac,
                    r.sat_max_pixel_adu,
                    r.sat_saturation_adu,
                    r.roi_signal_adu_s,
                    r.struct_p99m_p50_adu_s,
                    r.nebula_mask_frac,
                    r.masked_signal_adu_s,
                    r.masked_snr,
                    r.n_stars_used,
                    r.star_peak_p90_adu,
                    r.saturation_adu_used,
                    r.headroom_p90,
                    r.headroom_p99,
                    r.sat_star_frac,
                    r.psf0_n_peaks_good,
                    r.psf1_n_measured,
                    r.psf_grid_n_input_stars,
                    r.psf_grid_n_cells_with_data,
                    r.psf_grid_center_fwhm_px,
                    r.psf_grid_corner_fwhm_px_median,
                    r.psf_grid_center_to_corner_ratio,
                    r.psf2_n_modeled,
                    r.psf2_gauss_fwhm_med,
                    r.psf2_gauss_fwhm_p90,
                    r.psf2_gauss_ecc_med,
                    r.fits_duration_ms,
                    r.sky_basic_duration_ms,
                    r.sky_bkg2d_duration_ms,
                    r.signal_structure_duration_ms,
                    r.saturation_duration_ms,
                    r.roi_signal_duration_ms,
                    r.nebula_mask_duration_ms,
                    r.masked_signal_duration_ms,
                    r.star_headroom_duration_ms,
                    r.exposure_duration_ms,
                    r.flat_group_duration_ms,
                    r.flat_basic_duration_ms,
                    r.psf0_duration_ms,
                    r.psf1_duration_ms,
                    r.psf_grid_duration_ms,
                    r.psf2_duration_ms,
                    r.fits_duration_us,
                    r.sky_basic_duration_us,
                    r.sky_bkg2d_duration_us,
                    r.signal_structure_duration_us,
                    r.saturation_duration_us,
                    r.roi_signal_duration_us,
                    r.nebula_mask_duration_us,
                    r.masked_signal_duration_us,
                    r.star_headroom_duration_us,
                    r.exposure_duration_us,
                    r.flat_group_duration_us,
                    r.flat_basic_duration_us,
                    r.psf0_duration_us,
                    r.psf1_duration_us,
                    r.psf_grid_duration_us,
                    r.psf2_duration_us,
                    r.psf2_ms_per_star,
                    r.anomalies,
                ]
            )

    # Write JSON summary
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    n_anom_total = sum(int(v) for v in summary["anomaly_counts"].values())
    print(f"[sanity_check] images={summary['n_images']} total_anomalies={n_anom_total}")
    print(f"[sanity_check] report={out_csv}")
    print(f"[sanity_check] summary={out_json}")
    print(f"[sanity_check] performance={out_perf_csv}")
    print(f"[sanity_check] module_runs_checks={out_runs_csv}")

    mrc = summary.get("module_runs_checks", {})
    if (
        isinstance(mrc, dict)
        and mrc.get("enabled")
        and isinstance(mrc.get("violation_counts"), dict)
    ):
        vc = mrc["violation_counts"]
        total_v = int(mrc.get("total_violations", 0) or 0)
        if total_v > 0:
            topk = sorted(vc.items(), key=lambda kv: kv[1], reverse=True)[:10]
            print(f"[sanity_check] module_runs_violations_total={total_v}")
            for k, v in topk:
                print(f"  module_runs::{k}: {v}")

    if summary["anomaly_counts"]:
        top = sorted(
            summary["anomaly_counts"].items(), key=lambda kv: kv[1], reverse=True
        )[:10]
        for k, v in top:
            print(f"  {k}: {v}")

    conn.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Centaur sanity checker (DB -> CSV + JSON + performance CSV)."
    )
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )

    ap.add_argument(
        "--out", type=str, default="", help="Output CSV path (blank = auto timestamp)"
    )
    ap.add_argument(
        "--summary",
        type=str,
        default="",
        help="Output JSON summary path (blank = auto timestamp)",
    )
    ap.add_argument(
        "--perf",
        type=str,
        default="",
        help="Output performance CSV path (blank = auto timestamp)",
    )
    ap.add_argument(
        "--runs",
        type=str,
        default="",
        help="Output module_runs checks CSV path (blank = auto timestamp)",
    )

    args = ap.parse_args()

    out_csv = Path(args.out) if args.out.strip() else None
    out_json = Path(args.summary) if args.summary.strip() else None
    out_perf = Path(args.perf) if args.perf.strip() else None
    out_runs = Path(args.runs) if args.runs.strip() else None

    return run(Path(args.db), out_csv, out_json, out_perf, out_runs)


if __name__ == "__main__":
    raise SystemExit(main())
