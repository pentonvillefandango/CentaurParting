from __future__ import annotations

import csv
import json
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

# NEW: return record so pipeline writes module_runs centrally (with duration_us)
from centaur.pipeline import ModuleRunRecord

MODULE_NAME = "psf_basic_worker"


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


def _local_maxima_3x3(img2d: np.ndarray, threshold_adu: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast 3x3 local maxima finder above threshold. Returns (xs, ys).
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


def _build_peaks(img2d: np.ndarray, threshold_adu: float, max_peaks: int) -> List[Peak]:
    xs, ys = _local_maxima_3x3(img2d, threshold_adu)
    if xs.size == 0:
        return []

    vals = img2d[ys, xs].astype(np.float64, copy=False)

    peaks: List[Peak] = []
    for x, y, v in zip(xs.tolist(), ys.tolist(), vals.tolist()):
        peaks.append(Peak(x=int(x), y=int(y), value=float(v)))

    peaks.sort(key=lambda p: p.value, reverse=True)

    if max_peaks > 0 and len(peaks) > max_peaks:
        return peaks[:max_peaks]
    return peaks


def _grid_key(x: int, y: int, cell: int) -> Tuple[int, int]:
    return (x // cell, y // cell)


def _grid_accept(accepted: Dict[Tuple[int, int], Tuple[int, int]], x: int, y: int, cell: int, min_sep2: float) -> bool:
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


def _select_good_peaks(
    peaks: List[Peak],
    *,
    w: int,
    h: int,
    min_separation_px: int,
    edge_margin_px: int,
    good_threshold_adu: float,
) -> Tuple[List[Peak], int]:
    """
    Brightness + edge + grid separation. Returns (good_peaks, edge_rejected_count).
    """
    good: List[Peak] = []
    edge_rej = 0

    cell = max(1, int(min_separation_px))
    min_sep2 = float(min_separation_px * min_separation_px)
    accepted_cells: Dict[Tuple[int, int], Tuple[int, int]] = {}

    for p in peaks:
        if p.value < good_threshold_adu:
            continue

        edge = (
            p.x < edge_margin_px
            or p.y < edge_margin_px
            or p.x >= (w - edge_margin_px)
            or p.y >= (h - edge_margin_px)
        )
        if edge:
            edge_rej += 1
            continue

        if _grid_accept(accepted_cells, p.x, p.y, cell=cell, min_sep2=min_sep2):
            good.append(p)

    return good, edge_rej


def _cutout(img: np.ndarray, x: int, y: int, r: int) -> Optional[np.ndarray]:
    h, w = img.shape
    if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
        return None
    return img[y - r : y + r + 1, x - r : x + r + 1]


def _measure_star(cut: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    Returns (hfr_px, fwhm_px_proxy, ecc, theta_rad) or None on failure.
    All computed from moments / half-flux radius, background-subtracted.
    """
    bg = float(np.median(cut))
    data = cut.astype(np.float64, copy=False) - bg
    data[data < 0] = 0.0

    total = float(data.sum())
    if total <= 0 or not np.isfinite(total):
        return None

    yy, xx = np.indices(data.shape)
    cy = float((yy * data).sum() / total)
    cx = float((xx * data).sum() / total)

    dx = xx - cx
    dy = yy - cy

    # Second central moments
    mxx = float((dx * dx * data).sum() / total)
    myy = float((dy * dy * data).sum() / total)
    mxy = float((dx * dy * data).sum() / total)

    if not np.isfinite(mxx) or not np.isfinite(myy) or mxx <= 0 or myy <= 0:
        return None

    # Eigenvalues of covariance-like matrix give axis sigmas
    tr = mxx + myy
    det = mxx * myy - mxy * mxy
    if det <= 0:
        return None

    disc = tr * tr - 4.0 * det
    if disc < 0:
        disc = 0.0
    s = float(np.sqrt(disc))
    lam1 = 0.5 * (tr + s)
    lam2 = 0.5 * (tr - s)
    if lam1 <= 0 or lam2 <= 0:
        return None

    sigma_major = float(np.sqrt(max(lam1, lam2)))
    sigma_minor = float(np.sqrt(min(lam1, lam2)))

    ecc = 1.0 - (sigma_minor / sigma_major)
    theta = 0.5 * float(np.arctan2(2.0 * mxy, (mxx - myy)))

    sigma_eff = float(np.sqrt(0.5 * (sigma_major * sigma_major + sigma_minor * sigma_minor)))
    fwhm = 2.355 * sigma_eff

    r = np.sqrt(dx * dx + dy * dy)
    order = np.argsort(r.flat)
    cumsum = np.cumsum(data.flat[order])
    idx = int(np.searchsorted(cumsum, total * 0.5))
    if idx < 0 or idx >= order.size:
        return None
    hfr = float(r.flat[order][idx])

    if not (np.isfinite(hfr) and np.isfinite(fwhm) and np.isfinite(ecc) and np.isfinite(theta)):
        return None

    return hfr, fwhm, ecc, theta


def _pct(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    return _safe_float(np.percentile(np.asarray(vals, dtype=np.float64), p))


def _maybe_dump_psf1_csv(cfg: AppConfig, image_id: int, file_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not getattr(cfg, "psf1_debug_dump_measurements_csv", False):
        return

    out_dir = Path("data") / "debug" / "psf" / "measurements"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{image_id}_{file_path.stem}.psf1.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "hfr_px", "fwhm_px", "ecc", "theta_rad"])
        for r in rows:
            w.writerow([r["x"], r["y"], f"{r['hfr_px']:.6f}", f"{r['fwhm_px']:.6f}", f"{r['ecc']:.6f}", f"{r['theta_rad']:.6f}"])


def _get_image_id(db: Database, file_path: Path) -> Optional[int]:
    row = db.execute(
        "SELECT image_id FROM images WHERE file_path = ?",
        (str(file_path),),
    ).fetchone()
    return int(row["image_id"]) if row else None


def process_file_event(cfg: AppConfig, logger: Logger, event: FileReadyEvent, ctx: Optional[Any] = None) -> Any:
    t0 = time.monotonic()

    use_roi = getattr(cfg, "psf_use_roi", True)
    roi_fraction = float(getattr(cfg, "psf_roi_fraction", 0.5)) if use_roi else 1.0
    threshold_sigma = float(getattr(cfg, "psf_threshold_sigma", 8.0))
    good_extra_sigma = float(getattr(cfg, "psf_good_extra_sigma", 2.0))
    min_separation_px = int(getattr(cfg, "psf_min_separation_px", 8))
    edge_margin_px = int(getattr(cfg, "psf_edge_margin_px", 16))

    cutout_radius_px = int(getattr(cfg, "psf_cutout_radius_px", 8))
    max_stars_measured = int(getattr(cfg, "psf1_max_stars_measured", 5000))  # 0 = unlimited
    max_peaks_considered = int(getattr(cfg, "psf_max_stars", 0))  # reuse PSF-0 cap if set (0=unlimited)

    expected_fields = 0
    image_id: Optional[int] = None

    try:
        with fits.open(str(event.file_path), memmap=False) as hdul:
            data = hdul[0].data

        img2d = _flatten_image_data(data)
        if img2d.size == 0:
            raise ValueError("unsupported_fits_data_shape")

        if use_roi and roi_fraction < 1.0:
            roi2d, (x0, y0) = _central_roi(img2d, roi_fraction=roi_fraction)
        else:
            roi2d = img2d
            x0, y0 = 0, 0

        roi_vals = roi2d.reshape(-1)
        bg_median, bg_madstd = _robust_bg_median_madstd(roi_vals)

        if bg_median is None or bg_madstd is None:
            raise ValueError("no_background_estimate")

        if bg_madstd <= 1e-9:
            threshold_adu = float(bg_median + 50.0)
            good_threshold_adu = float(bg_median + 100.0)
        else:
            threshold_adu = float(bg_median + threshold_sigma * bg_madstd)
            good_threshold_adu = float(bg_median + (threshold_sigma + good_extra_sigma) * bg_madstd)

        peaks = _build_peaks(roi2d, threshold_adu=threshold_adu, max_peaks=max_peaks_considered)
        n_peaks_total = int(len(peaks))

        h, w = roi2d.shape[:2]
        good_peaks, _edge_rej = _select_good_peaks(
            peaks,
            w=w,
            h=h,
            min_separation_px=min_separation_px,
            edge_margin_px=edge_margin_px,
            good_threshold_adu=good_threshold_adu,
        )
        n_peaks_good = int(len(good_peaks))

        hfr_vals: List[float] = []
        fwhm_vals: List[float] = []
        ecc_vals: List[float] = []
        theta_vals: List[float] = []

        per_star_rows: List[Dict[str, Any]] = []

        n_rej_cutout = 0
        n_rej_measure = 0

        measured_xy: List[Tuple[int, int]] = []

        to_measure = good_peaks
        if max_stars_measured > 0 and len(to_measure) > max_stars_measured:
            to_measure = to_measure[:max_stars_measured]

        for p in to_measure:
            cut = _cutout(roi2d, p.x, p.y, cutout_radius_px)
            if cut is None:
                n_rej_cutout += 1
                continue

            m = _measure_star(cut)
            if m is None:
                n_rej_measure += 1
                continue

            hfr, fwhm, ecc, theta = m
            hfr_vals.append(hfr)
            fwhm_vals.append(fwhm)
            ecc_vals.append(ecc)
            theta_vals.append(theta)

            fx = int(p.x + x0)
            fy = int(p.y + y0)
            measured_xy.append((fx, fy))

            if getattr(cfg, "psf1_debug_dump_measurements_csv", False):
                per_star_rows.append(
                    {
                        "x": fx,
                        "y": fy,
                        "hfr_px": float(hfr),
                        "fwhm_px": float(fwhm),
                        "ecc": float(ecc),
                        "theta_rad": float(theta),
                    }
                )

        n_measured = int(len(hfr_vals))

        if ctx is not None:
            try:
                ctx.psf1 = {
                    "star_xy": list(measured_xy),
                    "fwhm_px": list(fwhm_vals),
                    "ecc": list(ecc_vals),
                    "image_shape": (int(img2d.shape[0]), int(img2d.shape[1])),
                }
            except Exception:
                pass

        star_xy_json = json.dumps([{"x": x, "y": y} for (x, y) in measured_xy]) if measured_xy else "[]"

        theta_med = _pct(theta_vals, 50)
        theta_p90abs = None
        if theta_med is not None and theta_vals:
            diffs = [abs(float(t) - float(theta_med)) for t in theta_vals]
            theta_p90abs = _pct(diffs, 90)

        if n_measured < 25:
            usable = 0
            reason = "too_few_psf_stars_measured"
        else:
            usable = 1
            reason = "ok"

        fields: Dict[str, Any] = {
            "roi_fraction": roi_fraction,
            "threshold_sigma": threshold_sigma,
            "good_extra_sigma": good_extra_sigma,
            "min_separation_px": min_separation_px,
            "edge_margin_px": edge_margin_px,
            "cutout_radius_px": cutout_radius_px,
            "max_stars_measured": max_stars_measured,
            "n_peaks_total": n_peaks_total,
            "n_peaks_good": n_peaks_good,
            "n_measured": n_measured,
            "n_rejected_cutout": n_rej_cutout,
            "n_rejected_measure": n_rej_measure,
            "hfr_px_median": _pct(hfr_vals, 50),
            "hfr_px_p10": _pct(hfr_vals, 10),
            "hfr_px_p90": _pct(hfr_vals, 90),
            "fwhm_px_median": _pct(fwhm_vals, 50),
            "fwhm_px_p10": _pct(fwhm_vals, 10),
            "fwhm_px_p90": _pct(fwhm_vals, 90),
            "ecc_median": _pct(ecc_vals, 50),
            "ecc_p90": _pct(ecc_vals, 90),
            "theta_rad_median": theta_med,
            "theta_rad_p90abs": theta_p90abs,
            "star_xy_json": star_xy_json,
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
                INSERT OR REPLACE INTO psf_basic_metrics (
                  image_id,
                  expected_fields, read_fields, written_fields,
                  parse_warnings, db_written_utc,

                  roi_fraction, threshold_sigma, good_extra_sigma,
                  min_separation_px, edge_margin_px, cutout_radius_px, max_stars_measured,

                  n_peaks_total, n_peaks_good, n_measured, n_rejected_cutout, n_rejected_measure,

                  hfr_px_median, hfr_px_p10, hfr_px_p90,
                  fwhm_px_median, fwhm_px_p10, fwhm_px_p90,
                  ecc_median, ecc_p90,
                  theta_rad_median, theta_rad_p90abs,

                  star_xy_json,

                  usable, reason
                ) VALUES (
                  ?,
                  ?, ?, ?,
                  ?, ?,

                  ?, ?, ?,
                  ?, ?, ?, ?,

                  ?, ?, ?, ?, ?,

                  ?, ?, ?,
                  ?, ?, ?,
                  ?, ?,
                  ?, ?,

                  ?,

                  ?, ?
                )
                """,
                (
                    image_id,
                    expected_fields, read_fields, written_fields,
                    None, utc_now(),

                    fields["roi_fraction"], fields["threshold_sigma"], fields["good_extra_sigma"],
                    fields["min_separation_px"], fields["edge_margin_px"], fields["cutout_radius_px"], fields["max_stars_measured"],

                    fields["n_peaks_total"], fields["n_peaks_good"], fields["n_measured"], fields["n_rejected_cutout"], fields["n_rejected_measure"],

                    fields["hfr_px_median"], fields["hfr_px_p10"], fields["hfr_px_p90"],
                    fields["fwhm_px_median"], fields["fwhm_px_p10"], fields["fwhm_px_p90"],
                    fields["ecc_median"], fields["ecc_p90"],
                    fields["theta_rad_median"], fields["theta_rad_p90abs"],

                    fields["star_xy_json"],

                    fields["usable"], fields["reason"],
                ),
            )

            _maybe_dump_psf1_csv(cfg, int(image_id), event.file_path, per_star_rows)

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
