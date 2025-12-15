from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent

MODULE_NAME = "psf_grid_worker"


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _median(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return _safe_float(float(np.median(np.asarray(vals, dtype=np.float64))))


def _iqr(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return None
    q75 = float(np.percentile(a, 75))
    q25 = float(np.percentile(a, 25))
    return _safe_float(q75 - q25)


def _get_image_id(db: Database, file_path: Path) -> Optional[int]:
    row = db.execute(
        "SELECT image_id FROM images WHERE file_path = ?",
        (str(file_path),),
    ).fetchone()
    return int(row["image_id"]) if row else None


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


# Duck-typed ctx for safety: pipeline passes an object with .psf1 dict
@dataclass
class _DummyCtx:
    psf1: Optional[Dict[str, Any]] = None


def process_file_event(cfg: AppConfig, logger: Logger, event: FileReadyEvent, ctx: Optional[Any] = None) -> Optional[bool]:
    t0 = time.monotonic()
    started_utc = utc_now()

    grid_rows = 3
    grid_cols = 3
    min_stars_per_cell = int(getattr(cfg, "psf_grid_min_stars_per_cell", 10))

    try:
        ctx = ctx if ctx is not None else _DummyCtx()
        psf1 = getattr(ctx, "psf1", None)

        with Database().transaction() as db:
            image_id = _get_image_id(db, event.file_path)
            if image_id is None:
                raise RuntimeError("image_id_not_found")

        # Gate: need PSF-1 in-memory payload
        if not isinstance(psf1, dict):
            fields = {"usable": 0, "reason": "psf1_context_missing"}
            row = _build_empty_row(
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                min_stars_per_cell=min_stars_per_cell,
                n_input_stars=0,
                reason="psf1_context_missing",
            )
            _write_row_and_log(cfg, logger, event, image_id, row, fields, started_utc, t0)
            return None

        star_xy: List[Tuple[int, int]] = list(psf1.get("star_xy", []))
        fwhm_px: List[float] = list(psf1.get("fwhm_px", []))
        ecc: List[float] = list(psf1.get("ecc", []))
        shape = psf1.get("image_shape", None)

        if shape is None or not isinstance(shape, tuple) or len(shape) != 2:
            raise RuntimeError("psf1_missing_image_shape")
        h, w = int(shape[0]), int(shape[1])

        if not (len(star_xy) == len(fwhm_px) == len(ecc)):
            raise RuntimeError("psf1_length_mismatch")

        # Filter accepted stars
        acc_xy: List[Tuple[int, int]] = []
        acc_f: List[float] = []
        acc_e: List[float] = []

        for (x, y), f, e in zip(star_xy, fwhm_px, ecc):
            if not (0 <= int(x) < w and 0 <= int(y) < h):
                continue
            f = float(f)
            e = float(e)
            if not (np.isfinite(f) and f > 0):
                continue
            if not (np.isfinite(e) and 0.0 <= e <= 1.0):
                continue
            acc_xy.append((int(x), int(y)))
            acc_f.append(f)
            acc_e.append(e)

        n_input_stars = int(len(acc_xy))

        if n_input_stars < 25:
            row = _build_empty_row(
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                min_stars_per_cell=min_stars_per_cell,
                n_input_stars=n_input_stars,
                reason="too_few_input_stars",
            )
            fields = {"n_input_stars": n_input_stars, "usable": 0, "reason": "too_few_input_stars"}
            _write_row_and_log(cfg, logger, event, image_id, row, fields, started_utc, t0)
            return None

        # Bin into 3x3
        cell_w = w / float(grid_cols)
        cell_h = h / float(grid_rows)

        cell_f: List[List[List[float]]] = [[[] for _ in range(grid_cols)] for _ in range(grid_rows)]
        cell_e: List[List[List[float]]] = [[[] for _ in range(grid_cols)] for _ in range(grid_rows)]

        for (x, y), f, e in zip(acc_xy, acc_f, acc_e):
            c = int(x / cell_w) if cell_w > 0 else 0
            r = int(y / cell_h) if cell_h > 0 else 0
            c = max(0, min(grid_cols - 1, c))
            r = max(0, min(grid_rows - 1, r))
            cell_f[r][c].append(float(f))
            cell_e[r][c].append(float(e))

        # Cell medians (min-stars rule)
        cell_n = [[len(cell_f[r][c]) for c in range(grid_cols)] for r in range(grid_rows)]
        cell_f_med: List[List[Optional[float]]] = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]
        cell_e_med: List[List[Optional[float]]] = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]

        n_cells_with_data = 0
        for r in range(grid_rows):
            for c in range(grid_cols):
                if cell_n[r][c] >= min_stars_per_cell:
                    cell_f_med[r][c] = _median(cell_f[r][c])
                    cell_e_med[r][c] = _median(cell_e[r][c])
                    if cell_f_med[r][c] is not None:
                        n_cells_with_data += 1

        # Existing rollups
        center = cell_f_med[1][1]
        corners = [cell_f_med[0][0], cell_f_med[0][2], cell_f_med[2][0], cell_f_med[2][2]]
        corners_ok = [v for v in corners if v is not None]
        corner_med = _median([float(v) for v in corners_ok]) if len(corners_ok) >= 2 else None

        def _side_med(vals: List[Optional[float]]) -> Optional[float]:
            ok = [v for v in vals if v is not None]
            return _median([float(v) for v in ok]) if ok else None

        left_med = _side_med([cell_f_med[0][0], cell_f_med[1][0], cell_f_med[2][0]])
        right_med = _side_med([cell_f_med[0][2], cell_f_med[1][2], cell_f_med[2][2]])
        top_med = _side_med([cell_f_med[0][0], cell_f_med[0][1], cell_f_med[0][2]])
        bot_med = _side_med([cell_f_med[2][0], cell_f_med[2][1], cell_f_med[2][2]])

        center_to_corner = (center / corner_med) if (center is not None and corner_med is not None and corner_med > 0) else None
        left_right = (left_med / right_med) if (left_med is not None and right_med is not None and right_med > 0) else None
        top_bottom = (top_med / bot_med) if (top_med is not None and bot_med is not None and bot_med > 0) else None

        # NEW: FWHM rollups across cells (using available cell medians only)
        fwhm_cells: List[float] = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                v = cell_f_med[r][c]
                if v is None:
                    continue
                fwhm_cells.append(float(v))

        fwhm_cell_median_overall = _median(fwhm_cells)
        fwhm_cell_iqr_overall = _iqr(fwhm_cells)
        fwhm_cell_min = _safe_float(min(fwhm_cells)) if fwhm_cells else None
        fwhm_cell_max = _safe_float(max(fwhm_cells)) if fwhm_cells else None
        fwhm_cell_range = (fwhm_cell_max - fwhm_cell_min) if (fwhm_cell_max is not None and fwhm_cell_min is not None) else None

        fwhm_center_minus_corner = (center - corner_med) if (center is not None and corner_med is not None) else None
        fwhm_left_right_delta = (right_med - left_med) if (right_med is not None and left_med is not None) else None
        fwhm_top_bottom_delta = (bot_med - top_med) if (bot_med is not None and top_med is not None) else None

        # NEW: ECC rollups across cells
        ecc_cells: List[float] = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                v = cell_e_med[r][c]
                if v is None:
                    continue
                ecc_cells.append(float(v))

        ecc_cell_median_overall = _median(ecc_cells)
        ecc_cell_max = _safe_float(max(ecc_cells)) if ecc_cells else None

        ecc_center = cell_e_med[1][1]
        ecc_corners = [cell_e_med[0][0], cell_e_med[0][2], cell_e_med[2][0], cell_e_med[2][2]]
        ecc_corners_ok = [v for v in ecc_corners if v is not None]
        ecc_corners_median = _median([float(v) for v in ecc_corners_ok]) if len(ecc_corners_ok) >= 2 else None
        ecc_center_minus_corners = (ecc_center - ecc_corners_median) if (ecc_center is not None and ecc_corners_median is not None) else None

        # Usable decision (unchanged)
        if n_cells_with_data < 3:
            usable = 0
            reason = "too_few_cells_with_data"
        else:
            usable = 1
            reason = "ok"

        # Build row dict matching schema
        row = {
            "image_id": image_id,

            "expected_fields": 0,
            "read_fields": 0,
            "written_fields": 0,
            "parse_warnings": None,
            "db_written_utc": utc_now(),

            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "grid_min_stars_per_cell": min_stars_per_cell,

            "n_input_stars": n_input_stars,
            "n_cells_with_data": int(n_cells_with_data),

            # existing rollups
            "center_fwhm_px": center,
            "corner_fwhm_px_median": corner_med,
            "center_to_corner_fwhm_ratio": center_to_corner,
            "left_right_fwhm_ratio": left_right,
            "top_bottom_fwhm_ratio": top_bottom,

            # new rollups
            "fwhm_cell_median_overall": fwhm_cell_median_overall,
            "fwhm_cell_iqr_overall": fwhm_cell_iqr_overall,
            "fwhm_cell_min": fwhm_cell_min,
            "fwhm_cell_max": fwhm_cell_max,
            "fwhm_cell_range": fwhm_cell_range,
            "fwhm_center_minus_corner": fwhm_center_minus_corner,
            "fwhm_left_right_delta": fwhm_left_right_delta,
            "fwhm_top_bottom_delta": fwhm_top_bottom_delta,

            "ecc_cell_median_overall": ecc_cell_median_overall,
            "ecc_cell_max": ecc_cell_max,
            "ecc_center": ecc_center,
            "ecc_corners_median": ecc_corners_median,
            "ecc_center_minus_corners": ecc_center_minus_corners,

            # cells
            "cell_r0c0_n": int(cell_n[0][0]), "cell_r0c0_fwhm_px_median": cell_f_med[0][0], "cell_r0c0_ecc_median": cell_e_med[0][0],
            "cell_r0c1_n": int(cell_n[0][1]), "cell_r0c1_fwhm_px_median": cell_f_med[0][1], "cell_r0c1_ecc_median": cell_e_med[0][1],
            "cell_r0c2_n": int(cell_n[0][2]), "cell_r0c2_fwhm_px_median": cell_f_med[0][2], "cell_r0c2_ecc_median": cell_e_med[0][2],

            "cell_r1c0_n": int(cell_n[1][0]), "cell_r1c0_fwhm_px_median": cell_f_med[1][0], "cell_r1c0_ecc_median": cell_e_med[1][0],
            "cell_r1c1_n": int(cell_n[1][1]), "cell_r1c1_fwhm_px_median": cell_f_med[1][1], "cell_r1c1_ecc_median": cell_e_med[1][1],
            "cell_r1c2_n": int(cell_n[1][2]), "cell_r1c2_fwhm_px_median": cell_f_med[1][2], "cell_r1c2_ecc_median": cell_e_med[1][2],

            "cell_r2c0_n": int(cell_n[2][0]), "cell_r2c0_fwhm_px_median": cell_f_med[2][0], "cell_r2c0_ecc_median": cell_e_med[2][0],
            "cell_r2c1_n": int(cell_n[2][1]), "cell_r2c1_fwhm_px_median": cell_f_med[2][1], "cell_r2c1_ecc_median": cell_e_med[2][1],
            "cell_r2c2_n": int(cell_n[2][2]), "cell_r2c2_fwhm_px_median": cell_f_med[2][2], "cell_r2c2_ecc_median": cell_e_med[2][2],

            "usable": int(usable),
            "reason": str(reason),
        }

        expected_fields = len(row) - 1  # excluding image_id
        row["expected_fields"] = expected_fields
        row["read_fields"] = expected_fields

        written_fields = 0
        for k, v in row.items():
            if k == "image_id":
                continue
            if v is None:
                continue
            written_fields += 1
        row["written_fields"] = written_fields

        fields_for_logging: Dict[str, Any] = {
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "grid_min_stars_per_cell": min_stars_per_cell,
            "n_input_stars": n_input_stars,
            "n_cells_with_data": int(n_cells_with_data),
            "center_fwhm_px": center,
            "corner_fwhm_px_median": corner_med,
            "center_to_corner_fwhm_ratio": center_to_corner,
            "left_right_fwhm_ratio": left_right,
            "top_bottom_fwhm_ratio": top_bottom,
            "fwhm_cell_median_overall": fwhm_cell_median_overall,
            "fwhm_cell_iqr_overall": fwhm_cell_iqr_overall,
            "fwhm_cell_range": fwhm_cell_range,
            "fwhm_left_right_delta": fwhm_left_right_delta,
            "fwhm_top_bottom_delta": fwhm_top_bottom_delta,
            "ecc_cell_median_overall": ecc_cell_median_overall,
            "ecc_cell_max": ecc_cell_max,
            "ecc_center": ecc_center,
            "ecc_corners_median": ecc_corners_median,
            "usable": int(usable),
            "reason": str(reason),
        }

        with Database().transaction() as db:
            db.execute(
                """
                INSERT OR REPLACE INTO psf_grid_metrics (
                  image_id,
                  expected_fields, read_fields, written_fields, parse_warnings, db_written_utc,
                  grid_rows, grid_cols, grid_min_stars_per_cell,
                  n_input_stars, n_cells_with_data,

                  center_fwhm_px, corner_fwhm_px_median, center_to_corner_fwhm_ratio,
                  left_right_fwhm_ratio, top_bottom_fwhm_ratio,

                  fwhm_cell_median_overall, fwhm_cell_iqr_overall,
                  fwhm_cell_min, fwhm_cell_max, fwhm_cell_range,
                  fwhm_center_minus_corner, fwhm_left_right_delta, fwhm_top_bottom_delta,

                  ecc_cell_median_overall, ecc_cell_max, ecc_center,
                  ecc_corners_median, ecc_center_minus_corners,

                  cell_r0c0_n, cell_r0c0_fwhm_px_median, cell_r0c0_ecc_median,
                  cell_r0c1_n, cell_r0c1_fwhm_px_median, cell_r0c1_ecc_median,
                  cell_r0c2_n, cell_r0c2_fwhm_px_median, cell_r0c2_ecc_median,

                  cell_r1c0_n, cell_r1c0_fwhm_px_median, cell_r1c0_ecc_median,
                  cell_r1c1_n, cell_r1c1_fwhm_px_median, cell_r1c1_ecc_median,
                  cell_r1c2_n, cell_r1c2_fwhm_px_median, cell_r1c2_ecc_median,

                  cell_r2c0_n, cell_r2c0_fwhm_px_median, cell_r2c0_ecc_median,
                  cell_r2c1_n, cell_r2c1_fwhm_px_median, cell_r2c1_ecc_median,
                  cell_r2c2_n, cell_r2c2_fwhm_px_median, cell_r2c2_ecc_median,

                  usable, reason
                ) VALUES (
                  ?,
                  ?, ?, ?, ?, ?,
                  ?, ?, ?,
                  ?, ?,

                  ?, ?, ?,
                  ?, ?,

                  ?, ?,
                  ?, ?, ?,
                  ?, ?, ?,

                  ?, ?, ?,
                  ?, ?,

                  ?, ?, ?,
                  ?, ?, ?,
                  ?, ?, ?,

                  ?, ?, ?,
                  ?, ?, ?,
                  ?, ?, ?,

                  ?, ?, ?,
                  ?, ?, ?,
                  ?, ?, ?,

                  ?, ?
                )
                """,
                (
                    row["image_id"],
                    row["expected_fields"], row["read_fields"], row["written_fields"], row["parse_warnings"], row["db_written_utc"],
                    row["grid_rows"], row["grid_cols"], row["grid_min_stars_per_cell"],
                    row["n_input_stars"], row["n_cells_with_data"],

                    row["center_fwhm_px"], row["corner_fwhm_px_median"], row["center_to_corner_fwhm_ratio"],
                    row["left_right_fwhm_ratio"], row["top_bottom_fwhm_ratio"],

                    row["fwhm_cell_median_overall"], row["fwhm_cell_iqr_overall"],
                    row["fwhm_cell_min"], row["fwhm_cell_max"], row["fwhm_cell_range"],
                    row["fwhm_center_minus_corner"], row["fwhm_left_right_delta"], row["fwhm_top_bottom_delta"],

                    row["ecc_cell_median_overall"], row["ecc_cell_max"], row["ecc_center"],
                    row["ecc_corners_median"], row["ecc_center_minus_corners"],

                    row["cell_r0c0_n"], row["cell_r0c0_fwhm_px_median"], row["cell_r0c0_ecc_median"],
                    row["cell_r0c1_n"], row["cell_r0c1_fwhm_px_median"], row["cell_r0c1_ecc_median"],
                    row["cell_r0c2_n"], row["cell_r0c2_fwhm_px_median"], row["cell_r0c2_ecc_median"],

                    row["cell_r1c0_n"], row["cell_r1c0_fwhm_px_median"], row["cell_r1c0_ecc_median"],
                    row["cell_r1c1_n"], row["cell_r1c1_fwhm_px_median"], row["cell_r1c1_ecc_median"],
                    row["cell_r1c2_n"], row["cell_r1c2_fwhm_px_median"], row["cell_r1c2_ecc_median"],

                    row["cell_r2c0_n"], row["cell_r2c0_fwhm_px_median"], row["cell_r2c0_ecc_median"],
                    row["cell_r2c1_n"], row["cell_r2c1_fwhm_px_median"], row["cell_r2c1_ecc_median"],
                    row["cell_r2c2_n"], row["cell_r2c2_fwhm_px_median"], row["cell_r2c2_ecc_median"],

                    row["usable"], row["reason"],
                ),
            )

            _insert_module_run(
                db,
                image_id,
                expected_read=row["expected_fields"],
                read=row["read_fields"],
                written=row["written_fields"],
                status="ok",
                message=None,
                started_utc=started_utc,
                ended_utc=utc_now(),
                duration_ms=int((time.monotonic() - t0) * 1000),
            )

        logger.log_module_result(
            module=MODULE_NAME,
            file=str(event.file_path),
            expected_read=row["expected_fields"],
            read=row["read_fields"],
            expected_written=row["expected_fields"],
            written=row["written_fields"],
            status="OK",
            duration_s=time.monotonic() - t0,
            verbose_fields=fields_for_logging if cfg.logging.is_verbose(MODULE_NAME) else None,
        )

        return None

    except Exception as e:
        logger.log_failure(
            module=MODULE_NAME,
            file=str(event.file_path),
            action=cfg.on_metric_failure,
            reason=f"{type(e).__name__}:{e}",
            duration_s=time.monotonic() - t0,
        )
        return False


def _build_empty_row(*, grid_rows: int, grid_cols: int, min_stars_per_cell: int, n_input_stars: int, reason: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "expected_fields": 0,
        "read_fields": 0,
        "written_fields": 0,
        "parse_warnings": None,
        "db_written_utc": utc_now(),
        "grid_rows": int(grid_rows),
        "grid_cols": int(grid_cols),
        "grid_min_stars_per_cell": int(min_stars_per_cell),
        "n_input_stars": int(n_input_stars),
        "n_cells_with_data": 0,

        "center_fwhm_px": None,
        "corner_fwhm_px_median": None,
        "center_to_corner_fwhm_ratio": None,
        "left_right_fwhm_ratio": None,
        "top_bottom_fwhm_ratio": None,

        "fwhm_cell_median_overall": None,
        "fwhm_cell_iqr_overall": None,
        "fwhm_cell_min": None,
        "fwhm_cell_max": None,
        "fwhm_cell_range": None,
        "fwhm_center_minus_corner": None,
        "fwhm_left_right_delta": None,
        "fwhm_top_bottom_delta": None,

        "ecc_cell_median_overall": None,
        "ecc_cell_max": None,
        "ecc_center": None,
        "ecc_corners_median": None,
        "ecc_center_minus_corners": None,

        "usable": 0,
        "reason": str(reason),
    }

    for r in range(3):
        for c in range(3):
            row[f"cell_r{r}c{c}_n"] = 0
            row[f"cell_r{r}c{c}_fwhm_px_median"] = None
            row[f"cell_r{r}c{c}_ecc_median"] = None

    expected_fields = len(row)
    row["expected_fields"] = expected_fields
    row["read_fields"] = expected_fields
    row["written_fields"] = sum(1 for v in row.values() if v is not None)
    return row


def _write_row_and_log(
    cfg: AppConfig,
    logger: Logger,
    event: FileReadyEvent,
    image_id: int,
    row: Dict[str, Any],
    fields_for_logging: Dict[str, Any],
    started_utc: str,
    t0: float,
) -> None:
    with Database().transaction() as db:
        db.execute(
            """
            INSERT OR REPLACE INTO psf_grid_metrics (
              image_id,
              expected_fields, read_fields, written_fields, parse_warnings, db_written_utc,
              grid_rows, grid_cols, grid_min_stars_per_cell,
              n_input_stars, n_cells_with_data,

              center_fwhm_px, corner_fwhm_px_median, center_to_corner_fwhm_ratio,
              left_right_fwhm_ratio, top_bottom_fwhm_ratio,

              fwhm_cell_median_overall, fwhm_cell_iqr_overall,
              fwhm_cell_min, fwhm_cell_max, fwhm_cell_range,
              fwhm_center_minus_corner, fwhm_left_right_delta, fwhm_top_bottom_delta,

              ecc_cell_median_overall, ecc_cell_max, ecc_center,
              ecc_corners_median, ecc_center_minus_corners,

              cell_r0c0_n, cell_r0c0_fwhm_px_median, cell_r0c0_ecc_median,
              cell_r0c1_n, cell_r0c1_fwhm_px_median, cell_r0c1_ecc_median,
              cell_r0c2_n, cell_r0c2_fwhm_px_median, cell_r0c2_ecc_median,

              cell_r1c0_n, cell_r1c0_fwhm_px_median, cell_r1c0_ecc_median,
              cell_r1c1_n, cell_r1c1_fwhm_px_median, cell_r1c1_ecc_median,
              cell_r1c2_n, cell_r1c2_fwhm_px_median, cell_r1c2_ecc_median,

              cell_r2c0_n, cell_r2c0_fwhm_px_median, cell_r2c0_ecc_median,
              cell_r2c1_n, cell_r2c1_fwhm_px_median, cell_r2c1_ecc_median,
              cell_r2c2_n, cell_r2c2_fwhm_px_median, cell_r2c2_ecc_median,

              usable, reason
            ) VALUES (
              ?,
              ?, ?, ?, ?, ?,
              ?, ?, ?,
              ?, ?,

              ?, ?, ?,
              ?, ?,

              ?, ?,
              ?, ?, ?,
              ?, ?, ?,

              ?, ?, ?,
              ?, ?,

              ?, ?, ?,
              ?, ?, ?,
              ?, ?, ?,

              ?, ?, ?,
              ?, ?, ?,
              ?, ?, ?,

              ?, ?, ?,
              ?, ?, ?,
              ?, ?, ?,

              ?, ?
            )
            """,
            (
                image_id,
                row["expected_fields"], row["read_fields"], row["written_fields"], row["parse_warnings"], row["db_written_utc"],
                row["grid_rows"], row["grid_cols"], row["grid_min_stars_per_cell"],
                row["n_input_stars"], row["n_cells_with_data"],

                row["center_fwhm_px"], row["corner_fwhm_px_median"], row["center_to_corner_fwhm_ratio"],
                row["left_right_fwhm_ratio"], row["top_bottom_fwhm_ratio"],

                row["fwhm_cell_median_overall"], row["fwhm_cell_iqr_overall"],
                row["fwhm_cell_min"], row["fwhm_cell_max"], row["fwhm_cell_range"],
                row["fwhm_center_minus_corner"], row["fwhm_left_right_delta"], row["fwhm_top_bottom_delta"],

                row["ecc_cell_median_overall"], row["ecc_cell_max"], row["ecc_center"],
                row["ecc_corners_median"], row["ecc_center_minus_corners"],

                row["cell_r0c0_n"], row["cell_r0c0_fwhm_px_median"], row["cell_r0c0_ecc_median"],
                row["cell_r0c1_n"], row["cell_r0c1_fwhm_px_median"], row["cell_r0c1_ecc_median"],
                row["cell_r0c2_n"], row["cell_r0c2_fwhm_px_median"], row["cell_r0c2_ecc_median"],

                row["cell_r1c0_n"], row["cell_r1c0_fwhm_px_median"], row["cell_r1c0_ecc_median"],
                row["cell_r1c1_n"], row["cell_r1c1_fwhm_px_median"], row["cell_r1c1_ecc_median"],
                row["cell_r1c2_n"], row["cell_r1c2_fwhm_px_median"], row["cell_r1c2_ecc_median"],

                row["cell_r2c0_n"], row["cell_r2c0_fwhm_px_median"], row["cell_r2c0_ecc_median"],
                row["cell_r2c1_n"], row["cell_r2c1_fwhm_px_median"], row["cell_r2c1_ecc_median"],
                row["cell_r2c2_n"], row["cell_r2c2_fwhm_px_median"], row["cell_r2c2_ecc_median"],

                row["usable"], row["reason"],
            ),
        )

        _insert_module_run(
            db,
            image_id,
            expected_read=row["expected_fields"],
            read=row["read_fields"],
            written=row["written_fields"],
            status="ok",
            message="skipped_psf_grid",
            started_utc=started_utc,
            ended_utc=utc_now(),
            duration_ms=int((time.monotonic() - t0) * 1000),
        )

    logger.log_module_result(
        module=MODULE_NAME,
        file=str(event.file_path),
        expected_read=row["expected_fields"],
        read=row["read_fields"],
        expected_written=row["expected_fields"],
        written=row["written_fields"],
        status="OK",
        duration_s=time.monotonic() - t0,
        verbose_fields=fields_for_logging if cfg.logging.is_verbose(MODULE_NAME) else None,
    )
