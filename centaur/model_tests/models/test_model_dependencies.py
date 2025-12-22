from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


TEST_NAME = "test_model_dependencies"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    r = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return r is not None


def _get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    if not _table_exists(conn, table):
        return []
    return [
        str(r["name"]) for r in conn.execute(f"PRAGMA table_info({table});").fetchall()
    ]


def _has_columns(
    conn: sqlite3.Connection, table: str, cols: Sequence[str]
) -> Tuple[bool, List[str]]:
    existing = set(_get_columns(conn, table))
    missing = [c for c in cols if c not in existing]
    return (len(missing) == 0), missing


def _in_list(image_ids: Sequence[int]) -> str:
    # Internal ids only; safe enough for a test harness.
    return ",".join(str(int(i)) for i in image_ids)


def _light_image_ids(conn: sqlite3.Connection) -> List[int]:
    if not _table_exists(conn, "fits_header_core"):
        return []
    rows = conn.execute(
        """
        SELECT image_id
        FROM fits_header_core
        WHERE UPPER(TRIM(COALESCE(imagetyp,''))) IN ('LIGHT','')
        ORDER BY image_id
        """
    ).fetchall()
    return [int(r["image_id"]) for r in rows]


def _flat_image_ids(conn: sqlite3.Connection) -> List[int]:
    if not _table_exists(conn, "fits_header_core"):
        return []
    rows = conn.execute(
        """
        SELECT image_id
        FROM fits_header_core
        WHERE UPPER(TRIM(COALESCE(imagetyp,'')))='FLAT'
        ORDER BY image_id
        """
    ).fetchall()
    return [int(r["image_id"]) for r in rows]


def _all_image_ids(conn: sqlite3.Connection) -> List[int]:
    if not _table_exists(conn, "images"):
        return []
    rows = conn.execute("SELECT image_id FROM images ORDER BY image_id").fetchall()
    return [int(r["image_id"]) for r in rows]


def _scope_ids(conn: sqlite3.Connection, scope: str) -> List[int]:
    s = scope.lower().strip()
    if s == "light":
        return _light_image_ids(conn)
    if s == "flat":
        return _flat_image_ids(conn)
    return _all_image_ids(conn)


# -----------------------------------------------------------------------------
# Count checks
# -----------------------------------------------------------------------------


def _count_missing_rows(
    conn: sqlite3.Connection, table: str, image_ids: Sequence[int]
) -> int:
    if not image_ids:
        return 0
    if not _table_exists(conn, table):
        return len(image_ids)

    ids = _in_list(image_ids)
    row = conn.execute(
        f"""
        WITH ids AS (SELECT image_id FROM images WHERE image_id IN ({ids}))
        SELECT COUNT(*) AS n_missing
        FROM ids
        LEFT JOIN {table} t ON t.image_id = ids.image_id
        WHERE t.image_id IS NULL
        """
    ).fetchone()
    return int(row["n_missing"]) if row else 0


def _count_nulls(
    conn: sqlite3.Connection, table: str, col: str, image_ids: Sequence[int]
) -> int:
    if not image_ids:
        return 0
    if not _table_exists(conn, table):
        return len(image_ids)

    ok, _missing_cols = _has_columns(conn, table, [col])
    if not ok:
        # Treat missing column as "all null" (hard fail).
        return len(image_ids)

    ids = _in_list(image_ids)
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS n_null
        FROM {table}
        WHERE image_id IN ({ids})
          AND {col} IS NULL
        """
    ).fetchone()
    return int(row["n_null"]) if row else 0


def _count_mismatches_passthrough(
    conn: sqlite3.Connection,
    image_ids: Sequence[int],
    src_table: str,
    src_col: str,
    dst_table: str,
    dst_col: str,
    *,
    cast_int: bool = False,
) -> int:
    """
    NULL-safe mismatch: requires both rows to exist; then (src IS dst) must be true.
    If cast_int=True, compare CAST(... AS INTEGER).
    """
    if not image_ids:
        return 0
    if not (_table_exists(conn, src_table) and _table_exists(conn, dst_table)):
        return 0

    ok1, _ = _has_columns(conn, src_table, [src_col])
    ok2, _ = _has_columns(conn, dst_table, [dst_col])
    if not (ok1 and ok2):
        return 0

    ids = _in_list(image_ids)
    if cast_int:
        expr_src = f"CAST(s.{src_col} AS INTEGER)"
        expr_dst = f"CAST(d.{dst_col} AS INTEGER)"
    else:
        expr_src = f"s.{src_col}"
        expr_dst = f"d.{dst_col}"

    row = conn.execute(
        f"""
        WITH ids AS (SELECT image_id FROM images WHERE image_id IN ({ids}))
        SELECT COUNT(*) AS n_bad
        FROM ids
        JOIN {src_table} s ON s.image_id = ids.image_id
        JOIN {dst_table} d ON d.image_id = ids.image_id
        WHERE NOT ({expr_src} IS {expr_dst})
        """
    ).fetchone()
    return int(row["n_bad"]) if row else 0


def _count_order_violations(
    conn: sqlite3.Connection,
    image_ids: Sequence[int],
    upstream_module: str,
    downstream_module: str,
) -> int:
    """
    Violation if downstream exists but upstream missing, or downstream ended before upstream ended.
    """
    if not image_ids:
        return 0
    if not _table_exists(conn, "module_runs"):
        return len(image_ids)

    ids = _in_list(image_ids)
    row = conn.execute(
        f"""
        WITH ids AS (SELECT image_id FROM images WHERE image_id IN ({ids})),
        up AS (
          SELECT image_id, ended_utc, status
          FROM module_runs
          WHERE module_name = ?
        ),
        dn AS (
          SELECT image_id, ended_utc, status
          FROM module_runs
          WHERE module_name = ?
        )
        SELECT COUNT(*) AS n_bad
        FROM ids
        LEFT JOIN dn ON dn.image_id = ids.image_id
        LEFT JOIN up ON up.image_id = ids.image_id
        WHERE dn.image_id IS NOT NULL
          AND (
            up.image_id IS NULL
            OR up.ended_utc IS NULL
            OR dn.ended_utc IS NULL
            OR dn.ended_utc < up.ended_utc
          )
        """,
        (upstream_module, downstream_module),
    ).fetchone()
    return int(row["n_bad"]) if row else 0


def _count_module_not_ok(
    conn: sqlite3.Connection, image_ids: Sequence[int], module_name: str
) -> int:
    """
    Missing module_run OR status != OK.
    """
    if not image_ids:
        return 0
    if not _table_exists(conn, "module_runs"):
        return len(image_ids)

    ids = _in_list(image_ids)
    row = conn.execute(
        f"""
        WITH ids AS (SELECT image_id FROM images WHERE image_id IN ({ids}))
        SELECT COUNT(*) AS n_bad
        FROM ids
        LEFT JOIN module_runs mr
               ON mr.image_id = ids.image_id
              AND mr.module_name = ?
        WHERE mr.image_id IS NULL
           OR UPPER(COALESCE(mr.status,'')) <> 'OK'
        """,
        (module_name,),
    ).fetchone()
    return int(row["n_bad"]) if row else 0


def _count_where(
    conn: sqlite3.Connection, table: str, where_sql: str, image_ids: Sequence[int]
) -> int:
    if not image_ids:
        return 0
    if not _table_exists(conn, table):
        return 0
    ids = _in_list(image_ids)
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS n
        FROM {table}
        WHERE image_id IN ({ids})
          AND ({where_sql})
        """
    ).fetchone()
    return int(row["n"]) if row else 0


def _count_gain_missing_in_header_full(
    conn: sqlite3.Connection, image_ids: Sequence[int]
) -> int:
    """
    Warning-only: for each image_id, does fits_header_full.header_json contain a GAIN keyword?
    """
    if not image_ids:
        return 0
    if not _table_exists(conn, "fits_header_full"):
        return 0

    ids = _in_list(image_ids)
    row = conn.execute(
        f"""
        WITH ids AS (SELECT image_id FROM images WHERE image_id IN ({ids}))
        SELECT COUNT(*) AS n_missing_gain
        FROM ids
        JOIN fits_header_full ff ON ff.image_id = ids.image_id
        WHERE NOT EXISTS (
          SELECT 1
          FROM json_each(ff.header_json) je
          WHERE UPPER(TRIM(json_extract(je.value,'$.keyword'))) = 'GAIN'
        )
        """
    ).fetchone()
    return int(row["n_missing_gain"]) if row else 0


def _count_sql_violations(
    conn: sqlite3.Connection,
    image_ids: Sequence[int],
    sql: str,
    params: Sequence[Any] = (),
) -> int:
    """
    Run a COUNT(*) query that should return a column named n_bad.
    Supports {in_list} placeholder.
    If query doesn't return n_bad, returns 0 (and the spec should be fixed).
    """
    if not image_ids:
        return 0

    q = sql.format(in_list=_in_list(image_ids))
    row = conn.execute(q, tuple(params)).fetchone()
    if row is None:
        return 0
    # Robustly accept first column if n_bad isn't present
    if "n_bad" in row.keys():
        return int(row["n_bad"])
    # fallback: first column
    try:
        return int(row[0])
    except Exception:
        return 0


# -----------------------------------------------------------------------------
# Reporting structures
# -----------------------------------------------------------------------------


@dataclass
class CheckIssue:
    key: str
    count: int
    detail: str


@dataclass
class TestReport:
    test: str
    status: str
    n_images_scope: int
    scope: str
    issues: List[Dict[str, Any]]
    warnings: List[str]


# -----------------------------------------------------------------------------
# Dependency specs ("DSL")
# -----------------------------------------------------------------------------
#
# Supported keys:
# - name: str
# - scope: "light" | "flat" | "all"
# - requires_rows: [table,...]
# - optional_rows: [table,...]
# - requires_nonnull: {table:[col,...]}
# - optional_passthrough: [(src_table, src_col, dst_table, dst_col, cast_int_bool), ...]
# - requires_order: [(upstream_module, downstream_module), ...]
# - requires_module_ok: [module_name,...]
# - warn_where: [(table, where_sql, label), ...]
# - warn_gain_keyword: bool
# - sql_checks: [{"key":..., "detail":..., "sql":..., "params":(...)}]
#
DEPENDENCY_SPECS: List[Dict[str, Any]] = [
    # =====================================================================
    # NEW: Root-level LIGHT checks (do not affect FLAT logic)
    # =====================================================================
    # -------------------------
    # SKY BASIC depends on FITS header (LIGHT)
    # -------------------------
    {
        "name": "sky_basic_depends_on_fits_header",
        "scope": "light",
        "requires_rows": ["fits_header_core", "sky_basic_metrics"],
        "requires_nonnull": {
            "fits_header_core": ["exptime"],
            "sky_basic_metrics": [
                "exptime_s",
                "roi_fraction",
                "ff_p50_adu",
                "ff_p99_adu",
                "ff_madstd_adu",
                "ff_madstd_adu_s",
                "roi_p50_adu",
                "roi_p99_adu",
                "roi_madstd_adu",
                "roi_madstd_adu_s",
                "nan_fraction",
                "inf_fraction",
                "clipped_fraction_ff",
                "clipped_fraction_roi",
            ],
        },
        "optional_passthrough": [
            ("fits_header_core", "exptime", "sky_basic_metrics", "exptime_s", False),
        ],
        "requires_order": [("fits_header_worker", "sky_basic_worker")],
        "requires_module_ok": ["fits_header_worker", "sky_basic_worker"],
        "sql_checks": [
            {
                "key": "sky_basic:roi_fraction_range",
                "detail": "sky_basic_metrics.roi_fraction must be in (0,1]",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM sky_basic_metrics
                    WHERE image_id IN ({in_list})
                      AND (roi_fraction IS NULL OR roi_fraction <= 0.0 OR roi_fraction > 1.0)
                """,
            },
            {
                "key": "sky_basic:fractions_in_0_1",
                "detail": "nan/inf/clipped fractions must be in [0,1]",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM sky_basic_metrics
                    WHERE image_id IN ({in_list})
                      AND (
                           nan_fraction IS NULL OR nan_fraction < 0.0 OR nan_fraction > 1.0
                        OR inf_fraction IS NULL OR inf_fraction < 0.0 OR inf_fraction > 1.0
                        OR clipped_fraction_ff IS NULL OR clipped_fraction_ff < 0.0 OR clipped_fraction_ff > 1.0
                        OR clipped_fraction_roi IS NULL OR clipped_fraction_roi < 0.0 OR clipped_fraction_roi > 1.0
                      )
                """,
            },
            {
                "key": "sky_basic:per_second_rates",
                "detail": "ff_madstd_adu_s and roi_madstd_adu_s should equal *_madstd_adu / exptime_s (within tolerance)",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM sky_basic_metrics
                    WHERE image_id IN ({in_list})
                      AND exptime_s IS NOT NULL AND exptime_s > 0
                      AND (
                           (ff_madstd_adu IS NOT NULL AND ff_madstd_adu_s IS NOT NULL
                            AND ABS(ff_madstd_adu_s - (ff_madstd_adu / exptime_s)) > 1e-6)
                        OR (roi_madstd_adu IS NOT NULL AND roi_madstd_adu_s IS NOT NULL
                            AND ABS(roi_madstd_adu_s - (roi_madstd_adu / exptime_s)) > 1e-6)
                      )
                """,
            },
        ],
    },
    # -------------------------
    # SKY BACKGROUND 2D depends on FITS header (LIGHT)
    # -------------------------
    {
        "name": "sky_background2d_depends_on_fits_header",
        "scope": "light",
        "requires_rows": ["fits_header_core", "sky_background2d_metrics"],
        "requires_nonnull": {
            "fits_header_core": ["exptime"],
            "sky_background2d_metrics": [
                "exptime_s",
                "tile_size_px",
                "grid_nx",
                "grid_ny",
                "clipped_fraction_mean",
                "bkg2d_median_adu",
                "bkg2d_min_adu",
                "bkg2d_max_adu",
                "bkg2d_range_adu",
                "plane_slope_mag_adu_per_tile",
                "grad_p95_adu_per_tile",
                "bkg2d_median_adu_s",
                "plane_slope_mag_adu_per_tile_s",
            ],
        },
        "optional_passthrough": [
            (
                "fits_header_core",
                "exptime",
                "sky_background2d_metrics",
                "exptime_s",
                False,
            ),
        ],
        "requires_order": [("fits_header_worker", "sky_background2d_worker")],
        "requires_module_ok": ["fits_header_worker", "sky_background2d_worker"],
        "sql_checks": [
            {
                "key": "sky_bkg2d:grid_positive",
                "detail": "tile_size_px, grid_nx, grid_ny must be > 0",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM sky_background2d_metrics
                    WHERE image_id IN ({in_list})
                      AND (
                           tile_size_px IS NULL OR tile_size_px <= 0
                        OR grid_nx IS NULL OR grid_nx <= 0
                        OR grid_ny IS NULL OR grid_ny <= 0
                      )
                """,
            },
            {
                "key": "sky_bkg2d:clipped_fraction_range",
                "detail": "clipped_fraction_mean must be in [0,1]",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM sky_background2d_metrics
                    WHERE image_id IN ({in_list})
                      AND (clipped_fraction_mean IS NULL OR clipped_fraction_mean < 0.0 OR clipped_fraction_mean > 1.0)
                """,
            },
            {
                "key": "sky_bkg2d:range_matches_min_max",
                "detail": "bkg2d_range_adu should equal (bkg2d_max_adu - bkg2d_min_adu) within tolerance",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM sky_background2d_metrics
                    WHERE image_id IN ({in_list})
                      AND bkg2d_min_adu IS NOT NULL
                      AND bkg2d_max_adu IS NOT NULL
                      AND bkg2d_range_adu IS NOT NULL
                      AND ABS(bkg2d_range_adu - (bkg2d_max_adu - bkg2d_min_adu)) > 1e-6
                """,
            },
            {
                "key": "sky_bkg2d:per_second_rates",
                "detail": "bkg2d_median_adu_s and plane_slope_mag_adu_per_tile_s should match / exptime_s (tolerance)",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM sky_background2d_metrics
                    WHERE image_id IN ({in_list})
                      AND exptime_s IS NOT NULL AND exptime_s > 0
                      AND (
                           (bkg2d_median_adu IS NOT NULL AND bkg2d_median_adu_s IS NOT NULL
                            AND ABS(bkg2d_median_adu_s - (bkg2d_median_adu / exptime_s)) > 1e-6)
                        OR (plane_slope_mag_adu_per_tile IS NOT NULL AND plane_slope_mag_adu_per_tile_s IS NOT NULL
                            AND ABS(plane_slope_mag_adu_per_tile_s - (plane_slope_mag_adu_per_tile / exptime_s)) > 1e-6)
                      )
                """,
            },
        ],
    },
    # -------------------------
    # SATURATION depends on FITS header (LIGHT)
    # -------------------------
    {
        "name": "saturation_depends_on_fits_header",
        "scope": "light",
        "requires_rows": ["fits_header_core", "saturation_metrics"],
        "requires_nonnull": {
            "fits_header_core": ["exptime"],
            "saturation_metrics": [
                "exptime_s",
                "saturation_adu",
                "max_pixel_adu",
                "saturated_pixel_count",
                "saturated_pixel_fraction",
                "nan_fraction",
                "inf_fraction",
                "usable",
            ],
        },
        "optional_passthrough": [
            ("fits_header_core", "exptime", "saturation_metrics", "exptime_s", False),
        ],
        "requires_order": [("fits_header_worker", "saturation_worker")],
        "requires_module_ok": ["fits_header_worker", "saturation_worker"],
        "sql_checks": [
            {
                "key": "saturation:fraction_range",
                "detail": "saturated_pixel_fraction must be in [0,1]",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM saturation_metrics
                    WHERE image_id IN ({in_list})
                      AND (saturated_pixel_fraction IS NULL OR saturated_pixel_fraction < 0.0 OR saturated_pixel_fraction > 1.0)
                """,
            },
            {
                "key": "saturation:counts_nonnegative",
                "detail": "saturated_pixel_count must be >= 0",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM saturation_metrics
                    WHERE image_id IN ({in_list})
                      AND (saturated_pixel_count IS NULL OR saturated_pixel_count < 0)
                """,
            },
            {
                "key": "saturation:fractions_in_0_1",
                "detail": "nan_fraction and inf_fraction must be in [0,1]",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM saturation_metrics
                    WHERE image_id IN ({in_list})
                      AND (
                           nan_fraction IS NULL OR nan_fraction < 0.0 OR nan_fraction > 1.0
                        OR inf_fraction IS NULL OR inf_fraction < 0.0 OR inf_fraction > 1.0
                      )
                """,
            },
            {
                "key": "saturation:ceiling_vs_max",
                "detail": "saturation_adu should be >= max_pixel_adu when both are present",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM saturation_metrics
                    WHERE image_id IN ({in_list})
                      AND saturation_adu IS NOT NULL
                      AND max_pixel_adu IS NOT NULL
                      AND saturation_adu < max_pixel_adu
                """,
            },
        ],
    },
    # -------------------------
    # PSF DETECT depends on SKY BASIC + header ordering (LIGHT)
    # -------------------------
    {
        "name": "psf_detect_depends_on_sky_basic_and_ordering",
        "scope": "light",
        "requires_rows": ["sky_basic_metrics", "psf_detect_metrics"],
        "requires_order": [
            ("fits_header_worker", "psf_detect_worker"),
            ("sky_basic_worker", "psf_detect_worker"),
        ],
        "requires_module_ok": ["psf_detect_worker", "sky_basic_worker"],
        "requires_nonnull": {
            "psf_detect_metrics": [
                "roi_fraction",
                "threshold_sigma",
                "min_separation_px",
                "max_stars",
                "bg_median_adu",
                "bg_madstd_adu",
                "threshold_adu",
                "n_candidates_total",
                "n_candidates_used",
                "nan_fraction",
                "inf_fraction",
                "usable",
                "n_peaks_total",
                "n_peaks_good",
                "peak_window",
                "good_extra_sigma",
                "good_threshold_adu",
            ],
        },
        "sql_checks": [
            {
                "key": "psf_detect:roi_fraction_range",
                "detail": "psf_detect_metrics.roi_fraction must be in (0,1]",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM psf_detect_metrics
                    WHERE image_id IN ({in_list})
                      AND (roi_fraction IS NULL OR roi_fraction <= 0.0 OR roi_fraction > 1.0)
                """,
            },
            {
                "key": "psf_detect:count_sanity",
                "detail": "candidate/peak counts must be nonnegative; used<=total; good<=total",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM psf_detect_metrics
                    WHERE image_id IN ({in_list})
                      AND (
                           n_candidates_total IS NULL OR n_candidates_total < 0
                        OR n_candidates_used IS NULL OR n_candidates_used < 0
                        OR n_candidates_used > n_candidates_total
                        OR n_peaks_total IS NULL OR n_peaks_total < 0
                        OR n_peaks_good IS NULL OR n_peaks_good < 0
                        OR n_peaks_good > n_peaks_total
                      )
                """,
            },
            {
                "key": "psf_detect:threshold_sane",
                "detail": "threshold_adu >= bg_median_adu and good_threshold_adu >= threshold_adu (when present)",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM psf_detect_metrics
                    WHERE image_id IN ({in_list})
                      AND (
                           (bg_median_adu IS NOT NULL AND threshold_adu IS NOT NULL AND threshold_adu < bg_median_adu)
                        OR (threshold_adu IS NOT NULL AND good_threshold_adu IS NOT NULL AND good_threshold_adu < threshold_adu)
                      )
                """,
            },
        ],
    },
    # -------------------------
    # NEW: Frame quality (LIGHT)
    # -------------------------
    {
        "name": "frame_quality_depends_on_light_metrics_and_ordering",
        "scope": "light",
        "requires_rows": ["frame_quality_metrics"],
        "optional_rows": [
            "psf_basic_metrics",
            "saturation_metrics",
            "sky_background2d_metrics",
        ],
        "requires_nonnull": {
            "frame_quality_metrics": [
                "db_written_utc",
                "quality_score",
                "decision",
                "reason_mask",
                "primary_reason",
                "psf_score",
                "bg_score",
                "clip_score",
                "usable",
                "reason",
            ],
        },
        "requires_order": [
            ("fits_header_worker", "frame_quality_worker"),
            ("sky_basic_worker", "frame_quality_worker"),
            ("sky_background2d_worker", "frame_quality_worker"),
            ("saturation_worker", "frame_quality_worker"),
            ("psf_basic_worker", "frame_quality_worker"),
            ("exposure_advice_worker", "frame_quality_worker"),
            ("signal_structure_worker", "frame_quality_worker"),
            ("nebula_mask_worker", "frame_quality_worker"),
            ("masked_signal_worker", "frame_quality_worker"),
            ("star_headroom_worker", "frame_quality_worker"),
        ],
        "requires_module_ok": ["frame_quality_worker"],
        "sql_checks": [
            {
                "key": "frame_quality:score_range",
                "detail": "frame_quality_metrics.quality_score must be in [0,100]",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM frame_quality_metrics
                    WHERE image_id IN ({in_list})
                      AND (quality_score IS NULL OR quality_score < 0 OR quality_score > 100)
                """,
            },
            {
                "key": "frame_quality:subscore_ranges",
                "detail": "psf_score/bg_score/clip_score must be in [0,100]",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM frame_quality_metrics
                    WHERE image_id IN ({in_list})
                      AND (
                           psf_score IS NULL OR psf_score < 0 OR psf_score > 100
                        OR bg_score  IS NULL OR bg_score  < 0 OR bg_score  > 100
                        OR clip_score IS NULL OR clip_score < 0 OR clip_score > 100
                      )
                """,
            },
            {
                "key": "frame_quality:decision_valid",
                "detail": "decision must be KEEP/WARN/REJECT",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM frame_quality_metrics
                    WHERE image_id IN ({in_list})
                      AND UPPER(TRIM(COALESCE(decision,''))) NOT IN ('KEEP','WARN','REJECT')
                """,
            },
            {
                "key": "frame_quality:reason_mask_nonneg",
                "detail": "reason_mask must be >= 0",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM frame_quality_metrics
                    WHERE image_id IN ({in_list})
                      AND (reason_mask IS NULL OR CAST(reason_mask AS INTEGER) < 0)
                """,
            },
            {
                "key": "frame_quality:primary_reason_present",
                "detail": "primary_reason must be non-empty",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM frame_quality_metrics
                    WHERE image_id IN ({in_list})
                      AND (primary_reason IS NULL OR TRIM(primary_reason) = '')
                """,
            },
            {
                "key": "frame_quality:usable_bool",
                "detail": "usable must be 0 or 1",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM frame_quality_metrics
                    WHERE image_id IN ({in_list})
                      AND (usable IS NULL OR CAST(usable AS INTEGER) NOT IN (0,1))
                """,
            },
            {
                "key": "frame_quality:reject_implies_unusable",
                "detail": "if decision='REJECT' then usable must be 0",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM frame_quality_metrics
                    WHERE image_id IN ({in_list})
                      AND UPPER(TRIM(COALESCE(decision,'')))='REJECT'
                      AND CAST(COALESCE(usable, 0) AS INTEGER) <> 0
                """,
            },
            {
                "key": "frame_quality:db_written_utc_present",
                "detail": "db_written_utc must be non-null/non-empty",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM frame_quality_metrics
                    WHERE image_id IN ({in_list})
                      AND (db_written_utc IS NULL OR TRIM(db_written_utc) = '')
                """,
            },
        ],
    },
    # =====================================================================
    # YOUR EXISTING SPECS (UNCHANGED) START HERE
    # =====================================================================
    # -------------------------
    # PSF chain (LIGHT)
    # -------------------------
    {
        "name": "psf_basic_depends_on_psf_detect",
        "scope": "light",
        "requires_rows": ["psf_detect_metrics", "psf_basic_metrics"],
        "requires_order": [("psf_detect_worker", "psf_basic_worker")],
        "requires_module_ok": ["psf_detect_worker", "psf_basic_worker"],
    },
    {
        "name": "psf_grid_depends_on_psf_basic",
        "scope": "light",
        "requires_rows": ["psf_basic_metrics", "psf_grid_metrics"],
        "requires_order": [("psf_basic_worker", "psf_grid_worker")],
        "requires_module_ok": ["psf_basic_worker", "psf_grid_worker"],
    },
    {
        "name": "psf_model_depends_on_psf_grid",
        "scope": "light",
        "requires_rows": ["psf_grid_metrics", "psf_model_metrics"],
        "requires_order": [("psf_grid_worker", "psf_model_worker")],
        "requires_module_ok": ["psf_grid_worker", "psf_model_worker"],
    },
    # -------------------------
    # ROI signal (LIGHT)
    # -------------------------
    {
        "name": "roi_signal_depends_on_fits_header",
        "scope": "light",
        "requires_rows": ["fits_header_core", "roi_signal_metrics"],
        "requires_nonnull": {
            "fits_header_core": ["exptime"],
            "roi_signal_metrics": [
                "exptime_s",
                "obj_median_adu",
                "bg_median_adu",
                "usable",
            ],
        },
        "optional_passthrough": [
            ("fits_header_core", "exptime", "roi_signal_metrics", "exptime_s", False),
        ],
        "requires_order": [("fits_header_worker", "roi_signal_worker")],
        "requires_module_ok": ["fits_header_worker", "roi_signal_worker"],
        "sql_checks": [
            {
                "key": "roi_signal:roi_fraction_range",
                "detail": "roi_signal_metrics.roi_fraction must be in (0,1]",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM roi_signal_metrics
                    WHERE image_id IN ({in_list})
                      AND (roi_fraction IS NULL OR roi_fraction <= 0.0 OR roi_fraction > 1.0)
                """,
            },
            {
                "key": "roi_signal:obj_minus_bg_rate_formula",
                "detail": "obj_minus_bg_adu_s should equal obj_minus_bg_adu / exptime_s (within tolerance)",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM roi_signal_metrics
                    WHERE image_id IN ({in_list})
                      AND exptime_s IS NOT NULL AND exptime_s > 0
                      AND obj_minus_bg_adu IS NOT NULL
                      AND obj_minus_bg_adu_s IS NOT NULL
                      AND ABS(obj_minus_bg_adu_s - (obj_minus_bg_adu / exptime_s)) > 1e-6
                """,
            },
        ],
    },
    # -------------------------
    # Signal structure (LIGHT)
    # -------------------------
    {
        "name": "signal_structure_inputs_and_passthrough",
        "scope": "light",
        "requires_rows": ["sky_basic_metrics", "signal_structure_metrics"],
        "requires_nonnull": {
            "sky_basic_metrics": [
                "exptime_s",
                "ff_madstd_adu_s",
                "ff_p50_adu",
                "ff_p99_adu",
            ],
            "signal_structure_metrics": [
                "exptime_s",
                "ff_madstd_adu_s",
                "eff_score",
                "time_weight",
            ],
        },
        "optional_rows": [
            "sky_background2d_metrics",
            "saturation_metrics",
            "psf_basic_metrics",
        ],
        "optional_passthrough": [
            (
                "sky_basic_metrics",
                "exptime_s",
                "signal_structure_metrics",
                "exptime_s",
                False,
            ),
            (
                "sky_basic_metrics",
                "ff_madstd_adu_s",
                "signal_structure_metrics",
                "ff_madstd_adu_s",
                False,
            ),
            (
                "sky_background2d_metrics",
                "plane_slope_mag_adu_per_tile_s",
                "signal_structure_metrics",
                "plane_slope_mag_adu_per_tile_s",
                False,
            ),
            (
                "saturation_metrics",
                "saturated_pixel_fraction",
                "signal_structure_metrics",
                "saturated_pixel_fraction",
                False,
            ),
            # NOTE: psf_* are NOT enforced as passthroughs. They are optional enrichments.
        ],
        "requires_order": [
            ("sky_basic_worker", "signal_structure_worker"),
            ("sky_background2d_worker", "signal_structure_worker"),
            ("saturation_worker", "signal_structure_worker"),
            # NOTE: do NOT require psf_basic_worker to precede signal_structure_worker.
        ],
        "requires_module_ok": ["sky_basic_worker", "signal_structure_worker"],
        "sql_checks": [
            {
                "key": "signal_structure:psf_usable_bool",
                "detail": "signal_structure_metrics.psf_usable must be 0/1 when present",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM signal_structure_metrics
                    WHERE image_id IN ({in_list})
                      AND psf_usable IS NOT NULL
                      AND psf_usable NOT IN (0,1)
                """,
            },
            {
                "key": "signal_structure:psf_fwhm_positive",
                "detail": "signal_structure_metrics.psf_fwhm_px_median must be > 0 when present",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM signal_structure_metrics
                    WHERE image_id IN ({in_list})
                      AND psf_fwhm_px_median IS NOT NULL
                      AND psf_fwhm_px_median <= 0
                """,
            },
            {
                "key": "signal_structure:psf_ecc_range",
                "detail": "signal_structure_metrics.psf_ecc_median must be in [0,1] when present",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM signal_structure_metrics
                    WHERE image_id IN ({in_list})
                      AND psf_ecc_median IS NOT NULL
                      AND (psf_ecc_median < 0.0 OR psf_ecc_median > 1.0)
                """,
            },
        ],
        "warn_where": [
            (
                "signal_structure_metrics",
                """
                (
                  (psf_fwhm_px_median IS NOT NULL)
                  + (psf_ecc_median IS NOT NULL)
                  + (psf_usable IS NOT NULL)
                ) IN (1,2)
                """,
                "psf_fields_partially_populated",
            ),
        ],
    },
    # -------------------------
    # Nebula mask (LIGHT) invariants
    # -------------------------
    {
        "name": "nebula_mask_invariants",
        "scope": "light",
        "requires_rows": ["nebula_mask_metrics"],
        "requires_order": [("fits_header_worker", "nebula_mask_worker")],
        "requires_module_ok": ["fits_header_worker", "nebula_mask_worker"],
        "requires_nonnull": {
            "nebula_mask_metrics": [
                "exptime_s",
                "bg_median_adu",
                "bg_madstd_adu",
                "threshold_adu",
                "mask_pixel_count",
                "mask_coverage_frac",
                "n_components",
                "largest_component_frac",
                "largest_component_bbox_json",
                "usable",
            ],
        },
        "sql_checks": [
            {
                "key": "nebula_mask:coverage_range",
                "detail": "mask_coverage_frac must be in [0,1]",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM nebula_mask_metrics
                    WHERE image_id IN ({in_list})
                      AND (mask_coverage_frac < 0.0 OR mask_coverage_frac > 1.0)
                """,
            },
            {
                "key": "nebula_mask:components_nonnegative",
                "detail": "n_components must be >= 0",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM nebula_mask_metrics
                    WHERE image_id IN ({in_list})
                      AND (n_components IS NULL OR n_components < 0)
                """,
            },
            {
                "key": "nebula_mask:largest_component_frac_range",
                "detail": "largest_component_frac must be in [0,1]",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM nebula_mask_metrics
                    WHERE image_id IN ({in_list})
                      AND (largest_component_frac < 0.0 OR largest_component_frac > 1.0)
                """,
            },
        ],
    },
    # -------------------------
    # Masked signal (LIGHT)
    # -------------------------
    {
        "name": "masked_signal_depends_on_nebula_mask_and_header",
        "scope": "light",
        "requires_rows": [
            "fits_header_core",
            "nebula_mask_metrics",
            "masked_signal_metrics",
        ],
        "requires_nonnull": {
            "fits_header_core": ["exptime"],
            "nebula_mask_metrics": ["threshold_adu"],
            "masked_signal_metrics": [
                "exptime_s",
                "nebula_pixel_count",
                "bg_pixel_count",
                "usable",
            ],
        },
        "optional_passthrough": [
            (
                "fits_header_core",
                "exptime",
                "masked_signal_metrics",
                "exptime_s",
                False,
            ),
        ],
        "requires_order": [
            ("fits_header_worker", "masked_signal_worker"),
            ("nebula_mask_worker", "masked_signal_worker"),
        ],
        "requires_module_ok": [
            "fits_header_worker",
            "nebula_mask_worker",
            "masked_signal_worker",
        ],
        "sql_checks": [
            {
                "key": "masked_signal:nebula_minus_bg_rate_formula",
                "detail": "nebula_minus_bg_adu_s should equal nebula_minus_bg_adu / exptime_s (within tolerance)",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM masked_signal_metrics
                    WHERE image_id IN ({in_list})
                      AND exptime_s IS NOT NULL AND exptime_s > 0
                      AND nebula_minus_bg_adu IS NOT NULL
                      AND nebula_minus_bg_adu_s IS NOT NULL
                      AND ABS(nebula_minus_bg_adu_s - (nebula_minus_bg_adu / exptime_s)) > 1e-6
                """,
            },
            {
                "key": "masked_signal:sane_fractions_and_counts",
                "detail": "nebula_frac in [0,1] and pixel counts nonnegative",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM masked_signal_metrics
                    WHERE image_id IN ({in_list})
                      AND (
                           nebula_frac IS NULL OR nebula_frac < 0.0 OR nebula_frac > 1.0
                        OR nebula_pixel_count IS NULL OR nebula_pixel_count < 0
                        OR bg_pixel_count IS NULL OR bg_pixel_count < 0
                      )
                """,
            },
        ],
    },
    # -------------------------
    # Star headroom (LIGHT)
    # -------------------------
    {
        "name": "star_headroom_depends_and_ranges",
        "scope": "light",
        "requires_rows": ["star_headroom_metrics", "saturation_metrics"],
        "optional_rows": ["psf_basic_metrics"],
        "requires_order": [
            ("fits_header_worker", "star_headroom_worker"),
            ("saturation_worker", "star_headroom_worker"),
            ("psf_basic_worker", "star_headroom_worker"),
        ],
        "requires_module_ok": [
            "fits_header_worker",
            "saturation_worker",
            "star_headroom_worker",
        ],
        "requires_nonnull": {"star_headroom_metrics": ["usable", "saturation_source"]},
        "sql_checks": [
            {
                "key": "star_headroom:headroom_range",
                "detail": "headroom_p50/p90/p99 must be in [0,1] when present",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM star_headroom_metrics
                    WHERE image_id IN ({in_list})
                      AND (
                           (headroom_p50 IS NOT NULL AND (headroom_p50 < 0.0 OR headroom_p50 > 1.0))
                        OR (headroom_p90 IS NOT NULL AND (headroom_p90 < 0.0 OR headroom_p90 > 1.0))
                        OR (headroom_p99 IS NOT NULL AND (headroom_p99 < 0.0 OR headroom_p99 > 1.0))
                      )
                """,
            },
            {
                "key": "star_headroom:saturation_adu_used_positive",
                "detail": "saturation_adu_used must be > 0 when usable=1",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM star_headroom_metrics
                    WHERE image_id IN ({in_list})
                      AND usable = 1
                      AND (saturation_adu_used IS NULL OR saturation_adu_used <= 0)
                """,
            },
        ],
    },
    # -------------------------
    # Exposure advice (LIGHT)
    # -------------------------
    {
        "name": "exposure_advice_depends_on_header_sky_basic_and_bkg2d",
        "scope": "light",
        "requires_rows": [
            "fits_header_core",
            "fits_header_full",
            "sky_basic_metrics",
            "sky_background2d_metrics",
            "exposure_advice",
        ],
        "requires_nonnull": {
            "fits_header_core": ["instrume", "exptime"],
            "sky_basic_metrics": ["roi_median_adu_s"],
            "sky_background2d_metrics": ["grad_p95_adu_per_tile"],
            "exposure_advice": ["decision_reason"],
        },
        "requires_order": [
            ("fits_header_worker", "exposure_advice_worker"),
            ("sky_basic_worker", "exposure_advice_worker"),
            ("sky_background2d_worker", "exposure_advice_worker"),
        ],
        "requires_module_ok": [
            "fits_header_worker",
            "sky_basic_worker",
            "sky_background2d_worker",
            "exposure_advice_worker",
        ],
        "warn_where": [
            (
                "exposure_advice",
                "decision_reason IS NOT NULL AND decision_reason <> 'ok'",
                "non_ok_decision_reason",
            ),
        ],
        "warn_gain_keyword": True,
        "sql_checks": [
            {
                "key": "exposure_advice:ok_requires_recs",
                "detail": "when decision_reason='ok', recommended_min_s and recommended_max_s must be non-null",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM exposure_advice
                    WHERE image_id IN ({in_list})
                      AND LOWER(TRIM(COALESCE(decision_reason,'')))='ok'
                      AND (recommended_min_s IS NULL OR recommended_max_s IS NULL)
                """,
            },
            {
                "key": "exposure_advice:rec_order",
                "detail": "recommended_max_s must be >= recommended_min_s when both present",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM exposure_advice
                    WHERE image_id IN ({in_list})
                      AND recommended_min_s IS NOT NULL
                      AND recommended_max_s IS NOT NULL
                      AND recommended_max_s < recommended_min_s
                """,
            },
        ],
    },
    # -------------------------
    # All LIGHT metrics after fits header (LIGHT scope)
    # -------------------------
    {
        "name": "all_light_metrics_after_fits_header",
        "scope": "light",
        "requires_module_ok": [
            "fits_header_worker",
            "sky_basic_worker",
            "sky_background2d_worker",
            "saturation_worker",
            "roi_signal_worker",
            "psf_detect_worker",
            "psf_basic_worker",
            "psf_grid_worker",
            "psf_model_worker",
            "signal_structure_worker",
            "nebula_mask_worker",
            "masked_signal_worker",
            "star_headroom_worker",
            "exposure_advice_worker",
            "frame_quality_worker",
        ],
        "requires_order": [
            ("fits_header_worker", "sky_basic_worker"),
            ("fits_header_worker", "sky_background2d_worker"),
            ("fits_header_worker", "saturation_worker"),
            ("fits_header_worker", "roi_signal_worker"),
            ("fits_header_worker", "psf_detect_worker"),
            ("fits_header_worker", "psf_basic_worker"),
            ("fits_header_worker", "psf_grid_worker"),
            ("fits_header_worker", "psf_model_worker"),
            ("fits_header_worker", "signal_structure_worker"),
            ("fits_header_worker", "nebula_mask_worker"),
            ("fits_header_worker", "masked_signal_worker"),
            ("fits_header_worker", "star_headroom_worker"),
            ("fits_header_worker", "exposure_advice_worker"),
            ("fits_header_worker", "frame_quality_worker"),
        ],
    },
    # -------------------------
    # FLAT pipeline (FLAT scope)
    # (everything below here is unchanged from your current file)
    # -------------------------
    {
        "name": "flat_pipeline_outputs_exist_and_link_integrity",
        "scope": "flat",
        "requires_module_ok": [
            "fits_header_worker",
            "flat_group_worker",
            "flat_basic_worker",
        ],
        "requires_order": [
            ("fits_header_worker", "flat_group_worker"),
            ("flat_group_worker", "flat_basic_worker"),
        ],
        "sql_checks": [
            {
                "key": "flat:tables_exist",
                "detail": "flat_* tables must exist when flats exist",
                "sql": """
                    SELECT
                      CASE
                        WHEN EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='flat_profiles')
                         AND EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='flat_capture_sets')
                         AND EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='flat_metrics')
                         AND EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='flat_frame_links')
                        THEN 0 ELSE 1
                      END AS n_bad
                """,
            },
            {
                "key": "flat:every_flat_has_link_row",
                "detail": "each FLAT image_id must have a row in flat_frame_links",
                "sql": """
                    WITH ids AS (
                      SELECT image_id
                      FROM fits_header_core
                      WHERE UPPER(TRIM(COALESCE(imagetyp,'')))='FLAT'
                        AND image_id IN ({in_list})
                    )
                    SELECT COUNT(*) AS n_bad
                    FROM ids
                    LEFT JOIN flat_frame_links l ON l.image_id = ids.image_id
                    WHERE l.image_id IS NULL
                """,
            },
            {
                "key": "flat:link_profile_fk_valid",
                "detail": "flat_frame_links.flat_profile_id must reference flat_profiles.flat_profile_id",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_frame_links l
                    LEFT JOIN flat_profiles p
                      ON p.flat_profile_id = l.flat_profile_id
                    WHERE l.image_id IN ({in_list})
                      AND p.flat_profile_id IS NULL
                """,
            },
            {
                "key": "flat:link_capture_set_fk_valid",
                "detail": "flat_frame_links.flat_capture_set_id must reference flat_capture_sets.flat_capture_set_id",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_frame_links l
                    LEFT JOIN flat_capture_sets s
                      ON s.flat_capture_set_id = l.flat_capture_set_id
                    WHERE l.image_id IN ({in_list})
                      AND s.flat_capture_set_id IS NULL
                """,
            },
        ],
    },
    {
        "name": "all_flat_metrics_after_fits_header",
        "scope": "flat",
        "requires_module_ok": [
            "fits_header_worker",
            "flat_group_worker",
            "flat_basic_worker",
        ],
        "requires_order": [
            ("fits_header_worker", "flat_group_worker"),
            ("fits_header_worker", "flat_basic_worker"),
            ("flat_group_worker", "flat_basic_worker"),
        ],
    },
    {
        "name": "flat_per_image_outputs_exist",
        "scope": "flat",
        "requires_rows": [
            "flat_metrics",
            "flat_frame_links",
        ],
        "requires_order": [
            ("fits_header_worker", "flat_group_worker"),
            ("flat_group_worker", "flat_basic_worker"),
        ],
        "requires_module_ok": [
            "fits_header_worker",
            "flat_group_worker",
            "flat_basic_worker",
        ],
    },
    {
        "name": "flat_links_and_metrics_integrity",
        "scope": "flat",
        "requires_module_ok": [
            "flat_group_worker",
            "flat_basic_worker",
        ],
        "sql_checks": [
            {
                "key": "flat:every_flat_has_link_row",
                "detail": "each FLAT image_id must have a row in flat_frame_links",
                "sql": """
                    WITH ids AS (
                      SELECT image_id
                      FROM fits_header_core
                      WHERE UPPER(TRIM(COALESCE(imagetyp,'')))='FLAT'
                        AND image_id IN ({in_list})
                    )
                    SELECT COUNT(*) AS n_bad
                    FROM ids
                    LEFT JOIN flat_frame_links l ON l.image_id = ids.image_id
                    WHERE l.image_id IS NULL
                """,
            },
            {
                "key": "flat:every_flat_has_metrics_row",
                "detail": "each FLAT image_id must have a row in flat_metrics",
                "sql": """
                    WITH ids AS (
                      SELECT image_id
                      FROM fits_header_core
                      WHERE UPPER(TRIM(COALESCE(imagetyp,'')))='FLAT'
                        AND image_id IN ({in_list})
                    )
                    SELECT COUNT(*) AS n_bad
                    FROM ids
                    LEFT JOIN flat_metrics m ON m.image_id = ids.image_id
                    WHERE m.image_id IS NULL
                """,
            },
            {
                "key": "flat:link_profile_fk_valid",
                "detail": "flat_frame_links.flat_profile_id must reference flat_profiles.flat_profile_id",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_frame_links l
                    LEFT JOIN flat_profiles p
                      ON p.flat_profile_id = l.flat_profile_id
                    WHERE l.image_id IN ({in_list})
                      AND p.flat_profile_id IS NULL
                """,
            },
            {
                "key": "flat:link_capture_set_fk_valid",
                "detail": "flat_frame_links.flat_capture_set_id must reference flat_capture_sets.flat_capture_set_id",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_frame_links l
                    LEFT JOIN flat_capture_sets s
                      ON s.flat_capture_set_id = l.flat_capture_set_id
                    WHERE l.image_id IN ({in_list})
                      AND s.flat_capture_set_id IS NULL
                """,
            },
        ],
    },
    {
        "name": "flat_metrics_invariants",
        "scope": "flat",
        "requires_rows": ["flat_metrics"],
        "requires_nonnull": {
            "flat_metrics": [
                "db_written_utc",
                "usable",
                "mean_adu",
                "median_adu",
                "std_adu",
                "madstd_adu",
                "min_adu",
                "max_adu",
                "clipped_low_frac",
                "clipped_high_frac",
                "corner_vignette_frac",
                "gradient_p95",
            ],
        },
        "sql_checks": [
            {
                "key": "flat_metrics:finite_ranges",
                "detail": "std_adu/madstd_adu must be >= 0 and max_adu must be >= min_adu",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_metrics
                    WHERE image_id IN ({in_list})
                      AND (
                           std_adu IS NULL OR std_adu < 0
                        OR madstd_adu IS NULL OR madstd_adu < 0
                        OR min_adu IS NULL OR max_adu IS NULL
                        OR max_adu < min_adu
                      )
                """,
            },
            {
                "key": "flat_metrics:clip_fracs_range",
                "detail": "clipped_low/high fractions must be in [0,1]",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_metrics
                    WHERE image_id IN ({in_list})
                      AND (
                           clipped_low_frac IS NULL OR clipped_low_frac < 0.0 OR clipped_low_frac > 1.0
                        OR clipped_high_frac IS NULL OR clipped_high_frac < 0.0 OR clipped_high_frac > 1.0
                      )
                """,
            },
            {
                "key": "flat_metrics:corner_vignette_frac_range",
                "detail": "corner_vignette_frac must be in [0,1]",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_metrics
                    WHERE image_id IN ({in_list})
                      AND (
                           corner_vignette_frac IS NULL
                        OR corner_vignette_frac < 0.0
                        OR corner_vignette_frac > 1.0
                      )
                """,
            },
            {
                "key": "flat_metrics:usable_is_0_or_1",
                "detail": "usable should be 0 or 1",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_metrics
                    WHERE image_id IN ({in_list})
                      AND (
                           usable IS NULL
                        OR CAST(usable AS INTEGER) NOT IN (0, 1)
                      )
                """,
            },
        ],
    },
    {
        "name": "flat_frame_links_invariants",
        "scope": "flat",
        "requires_rows": ["flat_frame_links"],
        "requires_nonnull": {
            "flat_frame_links": [
                "flat_profile_id",
                "flat_capture_set_id",
                "created_utc",
            ],
        },
        "sql_checks": [
            {
                "key": "flat_frame_links:ids_positive",
                "detail": "flat_profile_id and flat_capture_set_id should be positive integers",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_frame_links
                    WHERE image_id IN ({in_list})
                      AND (
                           flat_profile_id IS NULL OR CAST(flat_profile_id AS INTEGER) <= 0
                        OR flat_capture_set_id IS NULL OR CAST(flat_capture_set_id AS INTEGER) <= 0
                      )
                """,
            },
        ],
    },
    {
        "name": "flat_profiles_referenced_rows_invariants",
        "scope": "flat",
        "sql_checks": [
            {
                "key": "flat_profiles:referenced_profile_rows_exist",
                "detail": "each flat_frame_links.flat_profile_id must exist in flat_profiles.flat_profile_id",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_frame_links l
                    LEFT JOIN flat_profiles p
                      ON p.flat_profile_id = l.flat_profile_id
                    WHERE l.image_id IN ({in_list})
                      AND p.flat_profile_id IS NULL
                """,
            },
            {
                "key": "flat_profiles:camera_nonempty_for_referenced",
                "detail": "referenced flat_profiles.camera must be present and non-empty",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_frame_links l
                    JOIN flat_profiles p
                      ON p.flat_profile_id = l.flat_profile_id
                    WHERE l.image_id IN ({in_list})
                      AND (p.camera IS NULL OR TRIM(p.camera) = '')
                """,
            },
            {
                "key": "flat_profiles:created_utc_present_for_referenced",
                "detail": "referenced flat_profiles.created_utc must be non-null/non-empty",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_frame_links l
                    JOIN flat_profiles p
                      ON p.flat_profile_id = l.flat_profile_id
                    WHERE l.image_id IN ({in_list})
                      AND (p.created_utc IS NULL OR TRIM(p.created_utc) = '')
                """,
            },
            {
                "key": "flat_profiles:naxis_positive_for_referenced",
                "detail": "referenced flat_profiles.naxis1/naxis2 should be > 0 when present",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_frame_links l
                    JOIN flat_profiles p
                      ON p.flat_profile_id = l.flat_profile_id
                    WHERE l.image_id IN ({in_list})
                      AND (
                           (p.naxis1 IS NOT NULL AND CAST(p.naxis1 AS INTEGER) <= 0)
                        OR (p.naxis2 IS NOT NULL AND CAST(p.naxis2 AS INTEGER) <= 0)
                      )
                """,
            },
        ],
    },
    {
        "name": "flat_capture_sets_referenced_rows_invariants",
        "scope": "flat",
        "sql_checks": [
            {
                "key": "flat_capture_sets:referenced_rows_exist",
                "detail": "each flat_frame_links.flat_capture_set_id must exist in flat_capture_sets.flat_capture_set_id",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_frame_links l
                    LEFT JOIN flat_capture_sets s
                      ON s.flat_capture_set_id = l.flat_capture_set_id
                    WHERE l.image_id IN ({in_list})
                      AND s.flat_capture_set_id IS NULL
                """,
            },
            {
                "key": "flat_capture_sets:created_utc_present_for_referenced",
                "detail": "referenced flat_capture_sets.created_utc must be non-null/non-empty",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM flat_frame_links l
                    JOIN flat_capture_sets s
                      ON s.flat_capture_set_id = l.flat_capture_set_id
                    WHERE l.image_id IN ({in_list})
                      AND (s.created_utc IS NULL OR TRIM(s.created_utc) = '')
                """,
            },
        ],
    },
    {
        "name": "flat:metrics_and_links_cover_same_images",
        "scope": "flat",
        "requires_rows": ["flat_metrics", "flat_frame_links"],
        "sql_checks": [
            {
                "key": "flat:metrics_links_1to1",
                "detail": "each FLAT image should have exactly one row in flat_metrics and flat_frame_links",
                "sql": """
                    WITH ids AS (
                      SELECT image_id
                      FROM fits_header_core
                      WHERE UPPER(TRIM(COALESCE(imagetyp,'')))='FLAT'
                        AND image_id IN ({in_list})
                    ),
                    m AS (
                      SELECT image_id, COUNT(*) AS n FROM flat_metrics WHERE image_id IN ({in_list}) GROUP BY 1
                    ),
                    l AS (
                      SELECT image_id, COUNT(*) AS n FROM flat_frame_links WHERE image_id IN ({in_list}) GROUP BY 1
                    )
                    SELECT COUNT(*) AS n_bad
                    FROM ids
                    LEFT JOIN m ON m.image_id = ids.image_id
                    LEFT JOIN l ON l.image_id = ids.image_id
                    WHERE COALESCE(m.n,0) <> 1 OR COALESCE(l.n,0) <> 1
                """,
            },
        ],
    },
    {
        "name": "schema_core_tables_have_expected_columns",
        "scope": "all",
        "sql_checks": [
            {
                "key": "schema:images_has_file_path",
                "detail": "images must have file_path column",
                "sql": """
                    SELECT
                      CASE
                        WHEN EXISTS (
                          SELECT 1 FROM pragma_table_info('images') WHERE name='file_path'
                        )
                        THEN 0 ELSE 1
                      END AS n_bad
                """,
            },
            {
                "key": "schema:images_has_image_id",
                "detail": "images must have image_id column",
                "sql": """
                    SELECT
                      CASE
                        WHEN EXISTS (
                          SELECT 1 FROM pragma_table_info('images') WHERE name='image_id'
                        )
                        THEN 0 ELSE 1
                      END AS n_bad
                """,
            },
            {
                "key": "schema:module_runs_has_expected_cols",
                "detail": "module_runs must have image_id, module_name, status, ended_utc columns",
                "sql": """
                    SELECT
                      CASE
                        WHEN
                          EXISTS (SELECT 1 FROM pragma_table_info('module_runs') WHERE name='image_id')
                          AND EXISTS (SELECT 1 FROM pragma_table_info('module_runs') WHERE name='module_name')
                          AND EXISTS (SELECT 1 FROM pragma_table_info('module_runs') WHERE name='status')
                          AND EXISTS (SELECT 1 FROM pragma_table_info('module_runs') WHERE name='ended_utc')
                        THEN 0 ELSE 1
                      END AS n_bad
                """,
            },
            {
                "key": "schema:fits_header_full_has_header_json",
                "detail": "fits_header_full must have header_json column",
                "sql": """
                    SELECT
                      CASE
                        WHEN EXISTS (
                          SELECT 1 FROM pragma_table_info('fits_header_full') WHERE name='header_json'
                        )
                        THEN 0 ELSE 1
                      END AS n_bad
                """,
            },
            {
                "key": "schema:camera_constants_has_expected_cols",
                "detail": "camera_constants must have camera_name, gain_setting, gain_e_per_adu, read_noise_e columns",
                "sql": """
                    SELECT
                      CASE
                        WHEN
                          EXISTS (SELECT 1 FROM pragma_table_info('camera_constants') WHERE name='camera_name')
                          AND EXISTS (SELECT 1 FROM pragma_table_info('camera_constants') WHERE name='gain_setting')
                          AND EXISTS (SELECT 1 FROM pragma_table_info('camera_constants') WHERE name='gain_e_per_adu')
                          AND EXISTS (SELECT 1 FROM pragma_table_info('camera_constants') WHERE name='read_noise_e')
                        THEN 0 ELSE 1
                      END AS n_bad
                """,
            },
        ],
    },
    {
        "name": "exposure_advice_ok_requires_camera_constants_row",
        "scope": "light",
        "sql_checks": [
            {
                "key": "exposure_advice:ok_has_matching_camera_constants",
                "detail": "when exposure_advice.decision_reason='ok', camera_constants row must exist for (lower(instrume), gain_setting from header_json)",
                "sql": r"""
                    WITH ids AS (
                      SELECT image_id
                      FROM fits_header_core
                      WHERE image_id IN ({in_list})
                        AND UPPER(TRIM(COALESCE(imagetyp,''))) IN ('LIGHT','')
                    ),
                    gain AS (
                      SELECT
                        ff.image_id,
                        CAST(json_extract(je.value, '$.value') AS INTEGER) AS gain_setting
                      FROM fits_header_full ff
                      JOIN json_each(ff.header_json) AS je
                      WHERE ff.image_id IN (SELECT image_id FROM ids)
                        AND UPPER(json_extract(je.value, '$.keyword')) = 'GAIN'
                    ),
                    need AS (
                      SELECT
                        ea.image_id,
                        LOWER(COALESCE(fhc.instrume,'')) AS camera_name,
                        g.gain_setting AS gain_setting
                      FROM exposure_advice ea
                      JOIN fits_header_core fhc ON fhc.image_id = ea.image_id
                      LEFT JOIN gain g ON g.image_id = ea.image_id
                      WHERE ea.image_id IN (SELECT image_id FROM ids)
                        AND LOWER(TRIM(COALESCE(ea.decision_reason,''))) = 'ok'
                    )
                    SELECT COUNT(*) AS n_bad
                    FROM need n
                    WHERE NOT EXISTS (
                      SELECT 1
                      FROM camera_constants c
                      WHERE c.camera_name = n.camera_name
                        AND (
                              (c.gain_setting IS NULL AND n.gain_setting IS NULL)
                           OR (c.gain_setting = n.gain_setting)
                        )
                    )
                """,
            }
        ],
    },
    {
        "name": "module_runs_ok_must_have_ended_utc",
        "scope": "all",
        "sql_checks": [
            {
                "key": "module_runs:ok_requires_ended_utc",
                "detail": "for status='OK', module_runs.ended_utc must be non-null",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM module_runs
                    WHERE image_id IN ({in_list})
                      AND UPPER(COALESCE(status,''))='OK'
                      AND ended_utc IS NULL
                """,
            }
        ],
    },
    {
        "name": "images_watch_root_fk_integrity",
        "scope": "all",
        "requires_rows": ["images"],
        "sql_checks": [
            {
                "key": "images_watch_roots:tables_exist",
                "detail": "images and watch_roots tables must exist",
                "sql": """
                    SELECT
                      CASE
                        WHEN EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='images')
                         AND EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='watch_roots')
                        THEN 0 ELSE 1
                      END AS n_bad
                """,
            },
            {
                "key": "images_watch_roots:fk_valid",
                "detail": "images.watch_root_id must reference watch_roots.watch_root_id when non-null",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM images i
                    LEFT JOIN watch_roots w
                      ON w.watch_root_id = i.watch_root_id
                    WHERE i.image_id IN ({in_list})
                      AND i.watch_root_id IS NOT NULL
                      AND w.watch_root_id IS NULL
                """,
            },
        ],
    },
    {
        "name": "image_setups_row_per_image_and_fk_to_optical_setups",
        "scope": "all",
        "requires_rows": ["images", "image_setups"],
        "requires_nonnull": {
            "image_setups": ["setup_id", "method", "db_written_utc"],
        },
        "sql_checks": [
            {
                "key": "image_setups:tables_exist",
                "detail": "image_setups and optical_setups tables must exist",
                "sql": """
                    SELECT
                      CASE
                        WHEN EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='image_setups')
                         AND EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='optical_setups')
                        THEN 0 ELSE 1
                      END AS n_bad
                """,
            },
            {
                "key": "image_setups:row_per_image",
                "detail": "each images.image_id must have a row in image_setups",
                "sql": """
                    WITH ids AS (
                      SELECT image_id
                      FROM images
                      WHERE image_id IN ({in_list})
                    )
                    SELECT COUNT(*) AS n_bad
                    FROM ids
                    LEFT JOIN image_setups s ON s.image_id = ids.image_id
                    WHERE s.image_id IS NULL
                """,
            },
            {
                "key": "image_setups:setup_id_fk_valid",
                "detail": "image_setups.setup_id must reference optical_setups.setup_id",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM image_setups s
                    LEFT JOIN optical_setups o
                      ON o.setup_id = s.setup_id
                    WHERE s.image_id IN ({in_list})
                      AND s.setup_id IS NOT NULL
                      AND o.setup_id IS NULL
                """,
            },
            {
                "key": "image_setups:no_orphans",
                "detail": "image_setups.image_id must exist in images.image_id (no orphan setup rows)",
                "sql": """
                    SELECT COUNT(*) AS n_bad
                    FROM image_setups s
                    LEFT JOIN images i ON i.image_id = s.image_id
                    WHERE i.image_id IS NULL
                """,
            },
        ],
    },
]

# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


def run_dependency_spec(conn: sqlite3.Connection, spec: Dict[str, Any]) -> TestReport:
    scope = str(spec.get("scope", "all"))
    image_ids = _scope_ids(conn, scope)

    warnings: List[str] = []
    issues: List[CheckIssue] = []

    if not image_ids:
        warnings.append(f"no_images_in_scope:{scope}")
        return TestReport(
            test=f"{TEST_NAME}:{spec.get('name','unnamed')}",
            status="PASS",
            n_images_scope=0,
            scope=scope,
            issues=[],
            warnings=warnings,
        )

    # 1) Required row existence
    for table in spec.get("requires_rows", []) or []:
        t = str(table)
        missing = _count_missing_rows(conn, t, image_ids)
        if missing > 0:
            issues.append(
                CheckIssue(f"missing_rows:{t}", missing, f"{t} missing rows in scope")
            )

    # 2) Optional row existence (warn if partially missing)
    for table in spec.get("optional_rows", []) or []:
        t = str(table)
        if not _table_exists(conn, t):
            continue
        missing = _count_missing_rows(conn, t, image_ids)
        if 0 < missing < len(image_ids):
            warnings.append(
                f"optional_rows_partial_missing:{t}:{missing}/{len(image_ids)}"
            )

    # 3) Required non-null columns
    req_nonnull: Dict[str, List[str]] = spec.get("requires_nonnull", {}) or {}
    for table, cols in req_nonnull.items():
        t = str(table)
        if not _table_exists(conn, t):
            issues.append(
                CheckIssue(
                    f"missing_table:{t}", len(image_ids), "required table missing"
                )
            )
            continue

        ok, missing_cols = _has_columns(conn, t, cols)
        if not ok:
            issues.append(
                CheckIssue(
                    f"missing_cols:{t}",
                    len(missing_cols),
                    f"missing columns: {missing_cols}",
                )
            )
            continue

        for c in cols:
            n_null = _count_nulls(conn, t, str(c), image_ids)
            if n_null > 0:
                issues.append(
                    CheckIssue(
                        f"null_required:{t}.{c}", n_null, "required column has NULLs"
                    )
                )

    # 4) Optional passthrough checks (only if src exists in scope)
    for pt in spec.get("optional_passthrough", []) or []:
        src_table, src_col, dst_table, dst_col, cast_int = pt
        src_table = str(src_table)
        dst_table = str(dst_table)
        src_col = str(src_col)
        dst_col = str(dst_col)

        if not (_table_exists(conn, src_table) and _table_exists(conn, dst_table)):
            continue

        # If src missing for all, skip (truly optional)
        src_missing = _count_missing_rows(conn, src_table, image_ids)
        if src_missing == len(image_ids):
            continue

        n_bad = _count_mismatches_passthrough(
            conn,
            image_ids,
            src_table,
            src_col,
            dst_table,
            dst_col,
            cast_int=bool(cast_int),
        )
        if n_bad > 0:
            issues.append(
                CheckIssue(
                    f"passthrough_mismatch:{src_table}.{src_col}->{dst_table}.{dst_col}",
                    n_bad,
                    "NULL-safe mismatch between upstream and downstream values",
                )
            )

    # 5) Ordering checks
    for upstream, downstream in spec.get("requires_order", []) or []:
        n_bad = _count_order_violations(conn, image_ids, str(upstream), str(downstream))
        if n_bad > 0:
            issues.append(
                CheckIssue(
                    f"order_violation:{upstream}->{downstream}",
                    n_bad,
                    "downstream ended before upstream (or upstream missing)",
                )
            )

    # 6) Arbitrary SQL checks
    for chk in spec.get("sql_checks", []) or []:
        key = str(chk.get("key", "sql_check"))
        detail = str(chk.get("detail", "sql invariant failed"))
        sql = str(chk.get("sql", "")).strip()
        params = chk.get("params", ()) or ()
        if not sql:
            continue
        try:
            n_bad = _count_sql_violations(conn, image_ids, sql, params)
        except sqlite3.OperationalError as e:
            warnings.append(f"sql_check_skipped:{key}:{e}")
            continue
        if n_bad > 0:
            issues.append(CheckIssue(key, n_bad, detail))

    # 7) Module OK checks
    for module_name in spec.get("requires_module_ok", []) or []:
        n_bad = _count_module_not_ok(conn, image_ids, str(module_name))
        if n_bad > 0:
            issues.append(
                CheckIssue(
                    f"module_not_ok:{module_name}",
                    n_bad,
                    "module_run missing or status != OK",
                )
            )

    # 8) Warnings
    for w in spec.get("warn_where", []) or []:
        table, where_sql, label = str(w[0]), str(w[1]), str(w[2])
        n = _count_where(conn, table, where_sql, image_ids)
        if n > 0:
            warnings.append(f"warn:{label}:{table}:{n}/{len(image_ids)}")

    if bool(spec.get("warn_gain_keyword", False)) and scope.lower().strip() == "light":
        n_missing_gain = _count_gain_missing_in_header_full(conn, image_ids)
        if n_missing_gain > 0:
            warnings.append(
                f"warn:fits_header_full_missing_GAIN:{n_missing_gain}/{len(image_ids)}"
            )

    status = "PASS" if not issues else "FAIL"
    return TestReport(
        test=f"{TEST_NAME}:{spec.get('name','unnamed')}",
        status=status,
        n_images_scope=len(image_ids),
        scope=scope,
        issues=[asdict(x) for x in issues],
        warnings=warnings,
    )


def _print_console_summary(summary: Dict[str, Any]) -> None:
    print(f"[{TEST_NAME}] status={summary.get('status')}")
    failed = [r for r in summary.get("reports", []) if r.get("status") != "PASS"]
    if not failed:
        return

    print()
    print("=" * 80)
    print("FAILED SPECS")
    print("=" * 80)
    for r in failed:
        print()
        print(f"SPEC: {r.get('test')}")
        print(f"scope: {r.get('scope')} n_images: {r.get('n_images_scope')}")
        for i in r.get("issues", []):
            print(f" - ISSUE {i.get('key')}  count={i.get('count')}")
            print(f"   {i.get('detail')}")
        for w in r.get("warnings", []):
            print(f" - WARNING {w}")
    print("=" * 80)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to sqlite db")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[{TEST_NAME}] ERROR: db not found: {db_path}")
        return 2

    out_dir = (
        Path("data/model_tests/model_dependencies") / f"test_results_{utc_stamp()}"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "test_model_dependencies.json"

    with _connect(db_path) as conn:
        reports: List[TestReport] = []
        any_fail = False
        for spec in DEPENDENCY_SPECS:
            rep = run_dependency_spec(conn, spec)
            reports.append(rep)
            if rep.status != "PASS":
                any_fail = True

    summary: Dict[str, Any] = {
        "test": TEST_NAME,
        "status": "FAIL" if any_fail else "PASS",
        "n_specs": len(reports),
        "reports": [asdict(r) for r in reports],
    }

    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[{TEST_NAME}] wrote_json={out_path}")
    _print_console_summary(summary)

    return 2 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
