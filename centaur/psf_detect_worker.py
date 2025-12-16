from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits  # type: ignore

from centaur.config import AppConfig
from centaur.database import Database
from centaur.logging import Logger
from centaur.watcher import FileReadyEvent

# NEW: structured return so pipeline writes module_runs centrally (with duration_us)
from centaur.pipeline import ModuleRunRecord

MODULE_NAME = "psf_detect_worker"


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _nan_inf_fractions(arr: np.ndarray) -> Tuple[float, float]:
    total = arr.size if arr.size else 1
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    return nan_count / total, inf_count / total


def _flatten_image_data(data: np.ndarray) -> np.ndarray:
    if data is None:
        return np.array([], dtype=np.float64)

    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return np.array([], dtype=np.float64)

    return arr.astype(np.float64, copy=False)


def _central_roi(arr2d: np.ndarray, roi_fraction: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    roi_fraction = float(roi_fraction)
    roi_fraction = max(0.1, min(1.0, roi_fraction))

    h, w = arr2d.shape[:2]
    rh = int(h * roi_fraction)
    rw = int(w * roi_fraction)
    y0 = (h - rh) // 2
    x0 = (w - rw) // 2
    return arr2d[y0 : y0 + rh, x0 : x0 + rw], (x0, y0)


def _robust_bg_median_madstd(values_1d: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    finite = values_1d[np.isfinite(values_1d)]
    if finite.size == 0:
        return None, None

    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    madstd = 1.4826 * mad
    return _safe_float(med), _safe_float(madstd)


@dataclass(frozen=True)
class Peak:
    x: int
    y: int
    value: float
    saturated: bool


@dataclass(frozen=True)
class Candidate:
    x: int
    y: int
    value: float
    saturated: bool
    edge_rejected: bool
    rejected: bool
    reject_reason: str


def _local_maxima_3x3(img2d: np.ndarray, threshold_adu: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (xs, ys) arrays of 3x3 local maxima above threshold_adu.
    Scales with pixels once; does NOT enumerate all above-threshold pixels.
    """
    if img2d.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    finite = np.isfinite(img2d)
    base = np.where(finite, img2d, -np.inf)

    m = base >= float(threshold_adu)

    pad = np.pad(base, 1, mode="constant", constant_values=-np.inf)
    center = pad[1:-1, 1:-1]

    n0 = pad[0:-2, 0:-2]
    n1 = pad[0:-2, 1:-1]
    n2 = pad[0:-2, 2:]
    n3 = pad[1:-1, 0:-2]
    n4 = pad[1:-1, 2:]
    n5 = pad[2:, 0:-2]
    n6 = pad[2:, 1:-1]
    n7 = pad[2:, 2:]

    is_peak = m
    is_peak &= (center >= n0)
    is_peak &= (center >= n1)
    is_peak &= (center >= n2)
    is_peak &= (center >= n3)
    is_peak &= (center >= n4)
    is_peak &= (center >= n5)
    is_peak &= (center >= n6)
    is_peak &= (center >= n7)

    ys, xs = np.nonzero(is_peak)
    return xs.astype(np.int32, copy=False), ys.astype(np.int32, copy=False)


def _build_peaks(img2d: np.ndarray, *, threshold_adu: float, saturation_adu: Optional[float]) -> List[Peak]:
    xs, ys = _local_maxima_3x3(img2d, threshold_adu)
    if xs.size == 0:
        return []

    vals = img2d[ys, xs].astype(np.float64, copy=False)

    if saturation_adu is not None and np.isfinite(saturation_adu):
        sat = (vals >= float(saturation_adu))
    else:
        sat = None

    peaks: List[Peak] = []
    if sat is None:
        for x, y, v in zip(xs.tolist(), ys.tolist(), vals.tolist()):
            peaks.append(Peak(x=int(x), y=int(y), value=float(v), saturated=False))
    else:
        sat_list = sat.tolist()
        for x, y, v, s in zip(xs.tolist(), ys.tolist(), vals.tolist(), sat_list):
            peaks.append(Peak(x=int(x), y=int(y), value=float(v), saturated=bool(s)))

    # Brightest first -> deterministic selection
    peaks.sort(key=lambda p: p.value, reverse=True)
    return peaks


def _grid_key(x: int, y: int, cell: int) -> Tuple[int, int]:
    return (x // cell, y // cell)


def _grid_accept(
    accepted: Dict[Tuple[int, int], Tuple[int, int]],
    x: int,
    y: int,
    *,
    cell: int,
    min_sep2: float,
) -> bool:
    """
    Check only neighbor grid cells (<= 9) for too-close accepted stars.
    accepted maps cell->(x,y) of accepted star.
    """
    gx, gy = _grid_key(x, y, cell)

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            k = (gx + dx, gy + dy)
            if k not in accepted:
                continue
            ax, ay = accepted[k]
            ddx = float(x - ax)
            ddy = float(y - ay)
            if (ddx * ddx + ddy * ddy) < min_sep2:
                return False

    accepted[(gx, gy)] = (x, y)
    return True


def _select_good_peaks_grid(
    peaks: List[Peak],
    *,
    w: int,
    h: int,
    min_separation_px: int,
    edge_margin_px: int,
    good_threshold_adu: float,
) -> Tuple[List[Candidate], int, int]:
    """
    Fast, scalable “stars that matter” selection:
    - applies brightness, saturation, edge rules
    - replaces O(N^2) proximity checks with grid-based O(N)
    """
    candidates: List[Candidate] = []
    n_edge_rejected = 0
    n_saturated = 0

    cell = max(1, int(min_separation_px))
    min_sep2 = float(min_separation_px * min_separation_px)
    accepted_cells: Dict[Tuple[int, int], Tuple[int, int]] = {}

    for p in peaks:
        edge_rejected = (
            p.x < edge_margin_px
            or p.y < edge_margin_px
            or p.x >= (w - edge_margin_px)
            or p.y >= (h - edge_margin_px)
        )

        rejected = False
        reason = "ok"

        if p.value < good_threshold_adu:
            rejected = True
            reason = "too_faint_for_psf"
        elif p.saturated:
            rejected = True
            reason = "saturated"
        elif edge_rejected:
            rejected = True
            reason = "edge"
        else:
            ok = _grid_accept(
                accepted_cells,
                p.x,
                p.y,
                cell=cell,
                min_sep2=min_sep2,
            )
            if not ok:
                rejected = True
                reason = "too_close_to_brighter"

        if edge_rejected:
            n_edge_rejected += 1
        if p.saturated:
            n_saturated += 1

        candidates.append(
            Candidate(
                x=p.x,
                y=p.y,
                value=p.value,
                saturated=p.saturated,
                edge_rejected=edge_rejected,
                rejected=rejected,
                reject_reason=reason,
            )
        )

    return candidates, n_edge_rejected, n_saturated


def _get_image_id(db: Database, file_path: Path) -> Optional[int]:
    row = db.execute(
        "SELECT image_id FROM images WHERE file_path = ?",
        (str(file_path),),
    ).fetchone()
    return int(row["image_id"]) if row else None


def _maybe_dump_candidates_csv(cfg: AppConfig, image_id: int, file_path: Path, candidates: List[Candidate]) -> None:
    if not getattr(cfg, "psf_debug_dump_candidates_csv", False):
        return

    out_dir = Path("data") / "debug" / "psf" / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{image_id}_{file_path.stem}.candidates.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "peak_adu", "saturated", "edge_rejected", "rejected", "reject_reason"])
        for c in candidates:
            w.writerow([c.x, c.y, f"{c.value:.3f}", int(c.saturated), int(c.edge_rejected), int(c.rejected), c.reject_reason])


def process_file_event(cfg: AppConfig, logger: Logger, event: FileReadyEvent) -> Any:
    t0 = time.monotonic()

    use_roi = getattr(cfg, "psf_use_roi", True)
    roi_fraction = float(getattr(cfg, "psf_roi_fraction", 0.5)) if use_roi else 1.0
    threshold_sigma = float(getattr(cfg, "psf_threshold_sigma", 8.0))
    min_separation_px = int(getattr(cfg, "psf_min_separation_px", 8))
    max_stars = int(getattr(cfg, "psf_max_stars", 0))  # 0 = no cap
    edge_margin_px = int(getattr(cfg, "psf_edge_margin_px", 16))

    good_extra_sigma = float(getattr(cfg, "psf_good_extra_sigma", 2.0))
    peak_window = 3  # currently fixed (3x3)

    expected_fields = 0
    image_id: Optional[int] = None

    try:
        with fits.open(str(event.file_path), memmap=False) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data

        img2d = _flatten_image_data(data)
        if img2d.size == 0:
            raise ValueError("unsupported_fits_data_shape")

        nan_fraction, inf_fraction = _nan_inf_fractions(img2d)

        if use_roi and roi_fraction < 1.0:
            roi2d, (x0, y0) = _central_roi(img2d, roi_fraction=roi_fraction)
        else:
            roi2d = img2d
            x0, y0 = 0, 0

        roi_vals = roi2d.reshape(-1)
        bg_median, bg_madstd = _robust_bg_median_madstd(roi_vals)

        if bg_median is None or bg_madstd is None:
            threshold_adu = None
            good_threshold_adu = None
        else:
            if bg_madstd <= 1e-9:
                threshold_adu = bg_median + 50.0
                good_threshold_adu = bg_median + 100.0
            else:
                threshold_adu = bg_median + threshold_sigma * bg_madstd
                good_threshold_adu = bg_median + (threshold_sigma + good_extra_sigma) * bg_madstd

        saturation_adu = _safe_float(hdr.get("SATURATE", None))

        candidates: List[Candidate] = []
        n_peaks_total = 0
        n_peaks_good = 0
        n_edge_rejected = 0
        n_saturated_candidates = 0

        if threshold_adu is not None and good_threshold_adu is not None:
            peaks = _build_peaks(roi2d, threshold_adu=float(threshold_adu), saturation_adu=saturation_adu)

            if max_stars > 0 and len(peaks) > max_stars:
                peaks = peaks[:max_stars]

            n_peaks_total = int(len(peaks))

            h, w = roi2d.shape[:2]
            cand_roi, n_edge_rejected, n_saturated_candidates = _select_good_peaks_grid(
                peaks,
                w=w,
                h=h,
                min_separation_px=min_separation_px,
                edge_margin_px=edge_margin_px,
                good_threshold_adu=float(good_threshold_adu),
            )

            for c in cand_roi:
                candidates.append(
                    Candidate(
                        x=c.x + x0,
                        y=c.y + y0,
                        value=c.value,
                        saturated=c.saturated,
                        edge_rejected=c.edge_rejected,
                        rejected=c.rejected,
                        reject_reason=c.reject_reason,
                    )
                )

            n_peaks_good = int(sum(1 for c in candidates if not c.rejected))

        n_total = int(n_peaks_total)
        n_used = int(n_peaks_good)

        if threshold_adu is None:
            usable = 0
            reason = "no_background_estimate"
        elif n_used < 25:
            usable = 0
            reason = "too_few_good_stars"
        else:
            usable = 1
            reason = "ok"

        fields: Dict[str, Any] = {
            "roi_fraction": roi_fraction,
            "threshold_sigma": threshold_sigma,
            "good_extra_sigma": good_extra_sigma,
            "peak_window": peak_window,
            "min_separation_px": min_separation_px,
            "max_stars": max_stars,
            "bg_median_adu": bg_median,
            "bg_madstd_adu": bg_madstd,
            "threshold_adu": _safe_float(threshold_adu),
            "good_threshold_adu": _safe_float(good_threshold_adu),
            "n_peaks_total": n_peaks_total,
            "n_peaks_good": n_peaks_good,
            "n_candidates_total": n_total,
            "n_candidates_used": n_used,
            "n_saturated_candidates": n_saturated_candidates,
            "n_edge_rejected": n_edge_rejected,
            "nan_fraction": nan_fraction,
            "inf_fraction": inf_fraction,
            "usable": usable,
            "reason": reason,
        }

        expected_fields = len(fields)
        read_fields = expected_fields
        written_fields = sum(1 for v in fields.values() if v is not None)

        with Database().transaction() as db:
            image_id = _get_image_id(db, event.file_path)
            if image_id is None:
                raise RuntimeError("image_id_not_found_for_file_path")

            db.execute(
                """
                INSERT OR REPLACE INTO psf_detect_metrics (
                  image_id,
                  expected_fields, read_fields, written_fields,
                  parse_warnings, db_written_utc,
                  roi_fraction, threshold_sigma, min_separation_px, max_stars,
                  bg_median_adu, bg_madstd_adu, threshold_adu,
                  n_candidates_total, n_candidates_used, n_saturated_candidates, n_edge_rejected,
                  nan_fraction, inf_fraction,
                  usable, reason,
                  n_peaks_total, n_peaks_good, peak_window, good_extra_sigma, good_threshold_adu
                ) VALUES (
                  ?,
                  ?, ?, ?,
                  ?, ?,
                  ?, ?, ?, ?,
                  ?, ?, ?,
                  ?, ?, ?, ?,
                  ?, ?,
                  ?, ?,
                  ?, ?, ?, ?, ?
                )
                """,
                (
                    image_id,
                    expected_fields, read_fields, written_fields,
                    None, utc_now(),
                    roi_fraction, threshold_sigma, min_separation_px, max_stars,
                    fields["bg_median_adu"], fields["bg_madstd_adu"], fields["threshold_adu"],
                    n_total, n_used, n_saturated_candidates, n_edge_rejected,
                    nan_fraction, inf_fraction,
                    usable, reason,
                    n_peaks_total, n_peaks_good, peak_window, good_extra_sigma, fields["good_threshold_adu"],
                ),
            )

            _maybe_dump_candidates_csv(cfg, image_id, event.file_path, candidates)

        duration_s = time.monotonic() - t0
        logger.log_module_result(
            module=MODULE_NAME,
            file=str(event.file_path),
            expected_read=expected_fields,
            read=read_fields,
            expected_written=expected_fields,
            written=written_fields,
            status="OK",
            duration_s=duration_s,
            verbose_fields=fields,
        )

        return ModuleRunRecord(
            image_id=int(image_id),
            expected_read=expected_fields,
            read=read_fields,
            expected_written=expected_fields,
            written=written_fields,
            status="OK",
            message=None,
        )

    except Exception as e:
        duration_s = time.monotonic() - t0
        logger.log_failure(
            module=MODULE_NAME,
            file=str(event.file_path),
            action=cfg.on_metric_failure,
            reason=f"{type(e).__name__}:{e}",
            duration_s=duration_s,
        )

        if image_id is not None:
            return ModuleRunRecord(
                image_id=int(image_id),
                expected_read=int(expected_fields) if expected_fields else 0,
                read=0,
                expected_written=int(expected_fields) if expected_fields else 0,
                written=0,
                status="FAILED",
                message=f"{type(e).__name__}:{e}",
            )

        return False
