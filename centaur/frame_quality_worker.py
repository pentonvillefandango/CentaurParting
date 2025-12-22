from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, Optional, Tuple

from centaur.config import AppConfig
from centaur.logging import Logger, utc_now
from centaur.pipeline import ModuleRunRecord
from centaur.watcher import FileReadyEvent

MODULE_NAME = "frame_quality_worker"

# Reason flags (bitmask)
RQ_MISSING_UPSTREAM = 1 << 0
RQ_PSF_UNUSABLE = 1 << 1
RQ_SAT_UNUSABLE = 1 << 2
RQ_NAN_OR_INF = 1 << 3
RQ_TOO_FEW_STARS = 1 << 4
RQ_BAD_FWHM = 1 << 5
RQ_BAD_ECC = 1 << 6
RQ_CLIPPING = 1 << 7
RQ_STRONG_GRADIENT = 1 << 8
RQ_HIGH_BKG_STRUCTURE = 1 << 9

# NEW flags
RQ_PSF_TAIL_ECC = 1 << 10
RQ_FWHM_P90 = 1 << 11
RQ_FWHM_SPREAD = 1 << 12
RQ_LOW_SIGNAL = 1 << 13
RQ_LOW_HEADROOM = 1 << 14

# We now read up to 6 upstream rows
EXPECTED_READ = 6
EXPECTED_WRITTEN = 1

# Audit counts are "best effort" but should be stable and >0.
# Keep it in sync with the INSERT columns count you consider "expected".
EXPECTED_FIELDS_AUDIT = 32


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _int_clamp(x: float, lo: int, hi: int) -> int:
    return int(round(_clamp(float(x), float(lo), float(hi))))


def _primary_reason_from_mask(mask: int) -> str:
    # Priority order: reject-y things first
    if mask & RQ_MISSING_UPSTREAM:
        return "MISSING_UPSTREAM"
    if mask & RQ_NAN_OR_INF:
        return "NAN_OR_INF"
    if mask & RQ_SAT_UNUSABLE:
        return "SAT_UNUSABLE"
    if mask & RQ_PSF_UNUSABLE:
        return "PSF_UNUSABLE"
    if mask & RQ_CLIPPING:
        return "CLIPPING"
    if mask & RQ_LOW_HEADROOM:
        return "LOW_HEADROOM"
    if mask & RQ_TOO_FEW_STARS:
        return "TOO_FEW_STARS"
    if mask & RQ_PSF_TAIL_ECC:
        return "PSF_TAIL_ECC"
    if mask & RQ_BAD_FWHM:
        return "BAD_FWHM"
    if mask & RQ_FWHM_P90:
        return "FWHM_P90_HIGH"
    if mask & RQ_FWHM_SPREAD:
        return "FWHM_SPREAD_HIGH"
    if mask & RQ_BAD_ECC:
        return "BAD_ECC"
    if mask & RQ_STRONG_GRADIENT:
        return "STRONG_GRADIENT"
    if mask & RQ_HIGH_BKG_STRUCTURE:
        return "HIGH_BKG_STRUCTURE"
    if mask & RQ_LOW_SIGNAL:
        return "LOW_SIGNAL"
    return "OK"


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v or v == float("inf") or v == float("-inf"):
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


def _score_psf(
    *,
    fwhm_px_median: Optional[float],
    fwhm_px_p10: Optional[float],
    fwhm_px_p90: Optional[float],
    ecc_median: Optional[float],
    ecc_p90: Optional[float],
    n_measured: Optional[int],
) -> int:
    # Base: median FWHM / ecc, plus stars count
    fwhm = float(fwhm_px_median) if fwhm_px_median is not None else 999.0
    ecc = float(ecc_median) if ecc_median is not None else 1.0
    stars = int(n_measured) if n_measured is not None else 0

    fwhm_pen = _clamp((fwhm - 4.0) * 10.0, 0.0, 60.0)
    ecc_pen = _clamp((ecc - 0.40) * 200.0, 0.0, 60.0)
    stars_pen = _clamp((120.0 - float(stars)) * 0.30, 0.0, 40.0)

    # Robustness extras (small penalties so they influence but don't dominate)
    ecc90 = float(ecc_p90) if ecc_p90 is not None else None
    f90 = float(fwhm_px_p90) if fwhm_px_p90 is not None else None
    f10 = float(fwhm_px_p10) if fwhm_px_p10 is not None else None

    ecc90_pen = 0.0
    if ecc90 is not None:
        ecc90_pen = _clamp((ecc90 - 0.55) * 120.0, 0.0, 25.0)

    f90_pen = 0.0
    if f90 is not None:
        f90_pen = _clamp((f90 - 8.0) * 6.0, 0.0, 25.0)

    spread_pen = 0.0
    if f90 is not None and f10 is not None:
        spread = max(0.0, float(f90 - f10))
        spread_pen = _clamp((spread - 2.0) * 8.0, 0.0, 25.0)

    return _int_clamp(
        100.0 - fwhm_pen - ecc_pen - stars_pen - ecc90_pen - f90_pen - spread_pen,
        0,
        100,
    )


def _score_clip(sat_frac: Optional[float]) -> int:
    sf = float(sat_frac) if sat_frac is not None else 0.0
    pen = _clamp(sf * 200000.0, 0.0, 100.0)  # 0.0005 -> -100
    return _int_clamp(100.0 - pen, 0, 100)


def _score_bg(
    bkg_rms_map: Optional[float],
    plane_slope_mag: Optional[float],
    grad_p95: Optional[float],
    corner_delta: Optional[float],
) -> int:
    rms = float(bkg_rms_map) if bkg_rms_map is not None else 0.0
    slope = float(plane_slope_mag) if plane_slope_mag is not None else 0.0
    g95 = float(grad_p95) if grad_p95 is not None else 0.0
    cd = float(corner_delta) if corner_delta is not None else 0.0

    slope_pen = _clamp(abs(slope) * 0.5, 0.0, 45.0)
    rms_pen = _clamp(rms * 0.2, 0.0, 45.0)
    g95_pen = _clamp(abs(g95) * 0.2, 0.0, 25.0)
    cd_pen = _clamp(abs(cd) * 0.02, 0.0, 25.0)

    return _int_clamp(100.0 - slope_pen - rms_pen - g95_pen - cd_pen, 0, 100)


def _score_signal(
    obj_minus_bg_adu_s: Optional[float],
    nebula_minus_bg_adu_s: Optional[float],
    *,
    warn_min_obj: float,
    warn_min_nebula: float,
) -> int:
    # Neutral baseline if we don't have signal metrics yet.
    # We don't want to punish setups where mask isn't working or not applicable.
    base = 70.0

    o = float(obj_minus_bg_adu_s) if obj_minus_bg_adu_s is not None else None
    n = float(nebula_minus_bg_adu_s) if nebula_minus_bg_adu_s is not None else None

    pen = 0.0
    if warn_min_obj > 0.0 and o is not None and o < warn_min_obj:
        pen += _clamp((warn_min_obj - o) * 20.0, 0.0, 40.0)

    if warn_min_nebula > 0.0 and n is not None and n < warn_min_nebula:
        pen += _clamp((warn_min_nebula - n) * 20.0, 0.0, 40.0)

    # If we have strong positive signals, we can nudge upward a bit
    boost = 0.0
    if o is not None and o > 0:
        boost += _clamp(o * 0.5, 0.0, 15.0)
    if n is not None and n > 0:
        boost += _clamp(n * 0.5, 0.0, 15.0)

    return _int_clamp(base + boost - pen, 0, 100)


def _score_headroom(headroom_p99: Optional[float]) -> int:
    # headroom near 1 is good; near 0 means clipping risk.
    h = float(headroom_p99) if headroom_p99 is not None else None
    if h is None:
        return 70  # neutral if missing (and confidence will handle it)
    # Map: h<=0.05 -> ~0, h>=0.40 -> ~100
    return _int_clamp((h - 0.05) / (0.40 - 0.05) * 100.0, 0, 100)


def _score_confidence(
    *,
    has_psf: bool,
    has_sat: bool,
    has_bkg: bool,
    has_roi_signal: bool,
    has_masked_signal: bool,
    has_headroom: bool,
) -> int:
    # Simple: each available upstream adds confidence.
    total = 6
    have = sum(
        [has_psf, has_sat, has_bkg, has_roi_signal, has_masked_signal, has_headroom]
    )
    return _int_clamp(100.0 * float(have) / float(total), 0, 100)


def _weighted_quality(
    *,
    psf_score: int,
    bg_score: int,
    clip_score: int,
    signal_score: int,
    headroom_score: int,
    confidence_score: int,
) -> int:
    # Per-image only; keep weights stable and intuitive.
    # Confidence gently scales down "excellent" scores if lots of inputs missing.
    q = (
        0.45 * float(psf_score)
        + 0.20 * float(bg_score)
        + 0.15 * float(clip_score)
        + 0.10 * float(signal_score)
        + 0.10 * float(headroom_score)
    )
    # Apply confidence: if confidence is low, cap the effective score
    conf = float(confidence_score) / 100.0
    q_eff = q * (0.70 + 0.30 * conf)  # never zero it out; just temper it
    return _int_clamp(q_eff, 0, 100)


def _lookup_image_id(conn: sqlite3.Connection, file_path: str) -> Optional[int]:
    row = conn.execute(
        "SELECT image_id FROM images WHERE file_path = ?",
        (file_path,),
    ).fetchone()
    return int(row[0]) if row else None


def process_file_event(
    cfg: AppConfig, logger: Logger, event: FileReadyEvent, ctx: Any = None
) -> Any:
    """
    Returns ModuleRunRecord on normal path.
    IMPORTANT: if we cannot resolve image_id, return False (so pipeline won't FK-fail on module_runs insert).
    """
    t0 = time.perf_counter()

    conn = sqlite3.connect(str(cfg.db_path))
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")

        resolved = _lookup_image_id(conn, str(event.file_path))
        if resolved is None:
            msg = "could_not_resolve_image_id (images.file_path lookup failed)"
            logger.log_failure(
                MODULE_NAME,
                str(event.file_path),
                action=cfg.on_metric_failure,
                reason=msg,
                duration_s=(time.perf_counter() - t0),
            )
            return False  # avoid FK failure in pipeline/module_runs

        image_id = int(resolved)

        # --- Read upstream ---
        psf = conn.execute(
            """
            SELECT
              usable,
              fwhm_px_median, fwhm_px_p10, fwhm_px_p90,
              ecc_median, ecc_p90,
              n_measured
            FROM psf_basic_metrics
            WHERE image_id = ?
            """,
            (image_id,),
        ).fetchone()

        sat = conn.execute(
            """
            SELECT usable, saturated_pixel_fraction, nan_fraction, inf_fraction
            FROM saturation_metrics
            WHERE image_id = ?
            """,
            (image_id,),
        ).fetchone()

        bkg = conn.execute(
            """
            SELECT
              bkg2d_rms_of_map_adu,
              plane_slope_mag_adu_per_tile,
              grad_p95_adu_per_tile,
              corner_delta_adu
            FROM sky_background2d_metrics
            WHERE image_id = ?
            """,
            (image_id,),
        ).fetchone()

        roi_sig = conn.execute(
            """
            SELECT usable, obj_minus_bg_adu_s
            FROM roi_signal_metrics
            WHERE image_id = ?
            """,
            (image_id,),
        ).fetchone()

        masked_sig = conn.execute(
            """
            SELECT usable, nebula_minus_bg_adu_s
            FROM masked_signal_metrics
            WHERE image_id = ?
            """,
            (image_id,),
        ).fetchone()

        head = conn.execute(
            """
            SELECT usable, headroom_p99
            FROM star_headroom_metrics
            WHERE image_id = ?
            """,
            (image_id,),
        ).fetchone()

        has_psf = psf is not None
        has_sat = sat is not None
        has_bkg = bkg is not None
        has_roi_signal = roi_sig is not None
        has_masked_signal = masked_sig is not None
        has_headroom = head is not None

        read = sum(
            [has_psf, has_sat, has_bkg, has_roi_signal, has_masked_signal, has_headroom]
        )

        # Parse basics
        reason_mask = 0
        parse_warnings: Optional[str] = None

        # For v1.1 we still REQUIRE psf+sat+bkg (core); the rest is optional.
        if (not has_psf) or (not has_sat) or (not has_bkg):
            reason_mask |= RQ_MISSING_UPSTREAM
            parse_warnings = "missing_core_upstream_rows(psf/sat/bkg)"

        psf_usable = int(psf["usable"]) if has_psf and psf["usable"] is not None else 0
        sat_usable = int(sat["usable"]) if has_sat and sat["usable"] is not None else 0
        bkg_usable = 1 if has_bkg else 0  # bkg table has no usable flag

        roi_usable = (
            int(roi_sig["usable"])
            if has_roi_signal and roi_sig["usable"] is not None
            else 0
        )
        masked_usable = (
            int(masked_sig["usable"])
            if has_masked_signal and masked_sig["usable"] is not None
            else 0
        )
        head_usable = (
            int(head["usable"]) if has_headroom and head["usable"] is not None else 0
        )

        if (not has_psf) or psf_usable == 0:
            reason_mask |= RQ_PSF_UNUSABLE
        if (not has_sat) or sat_usable == 0:
            reason_mask |= RQ_SAT_UNUSABLE

        nan_fraction = _safe_float(sat["nan_fraction"]) if has_sat else None
        inf_fraction = _safe_float(sat["inf_fraction"]) if has_sat else None
        if (nan_fraction is not None and nan_fraction > 0.0) or (
            inf_fraction is not None and inf_fraction > 0.0
        ):
            reason_mask |= RQ_NAN_OR_INF

        # Driver values
        fwhm_px_median = _safe_float(psf["fwhm_px_median"]) if has_psf else None
        fwhm_px_p10 = _safe_float(psf["fwhm_px_p10"]) if has_psf else None
        fwhm_px_p90 = _safe_float(psf["fwhm_px_p90"]) if has_psf else None
        ecc_median = _safe_float(psf["ecc_median"]) if has_psf else None
        ecc_p90 = _safe_float(psf["ecc_p90"]) if has_psf else None
        n_measured = _safe_int(psf["n_measured"]) if has_psf else None

        saturated_pixel_fraction = (
            _safe_float(sat["saturated_pixel_fraction"]) if has_sat else None
        )

        bkg2d_rms_of_map_adu = (
            _safe_float(bkg["bkg2d_rms_of_map_adu"]) if has_bkg else None
        )
        plane_slope_mag_adu_per_tile = (
            _safe_float(bkg["plane_slope_mag_adu_per_tile"]) if has_bkg else None
        )
        grad_p95_adu_per_tile = (
            _safe_float(bkg["grad_p95_adu_per_tile"]) if has_bkg else None
        )
        corner_delta_adu = _safe_float(bkg["corner_delta_adu"]) if has_bkg else None

        obj_minus_bg_adu_s = None
        if has_roi_signal and roi_usable == 1:
            obj_minus_bg_adu_s = _safe_float(roi_sig["obj_minus_bg_adu_s"])

        nebula_minus_bg_adu_s = None
        if has_masked_signal and masked_usable == 1:
            nebula_minus_bg_adu_s = _safe_float(masked_sig["nebula_minus_bg_adu_s"])

        headroom_p99 = None
        if has_headroom and head_usable == 1:
            headroom_p99 = _safe_float(head["headroom_p99"])

        # Config thresholds
        sf = (
            float(saturated_pixel_fraction)
            if saturated_pixel_fraction is not None
            else 0.0
        )
        nm = int(n_measured) if n_measured is not None else 0
        ecc = float(ecc_median) if ecc_median is not None else 1.0
        fwhm = float(fwhm_px_median) if fwhm_px_median is not None else 999.0

        sat_reject = float(cfg.quality_reject_saturated_pixel_fraction)
        sat_warn = float(cfg.quality_warn_saturated_pixel_fraction)

        min_stars_reject = int(cfg.quality_reject_min_stars_measured)
        min_stars_warn = int(cfg.quality_warn_min_stars_measured)

        ecc_reject = float(cfg.quality_reject_ecc_median)
        ecc_warn = float(cfg.quality_warn_ecc_median)

        ecc_p90_reject = float(getattr(cfg, "quality_reject_ecc_p90", 0.85))
        ecc_p90_warn = float(getattr(cfg, "quality_warn_ecc_p90", 0.70))

        fwhm_reject = float(cfg.quality_reject_fwhm_px_median)
        fwhm_warn = float(cfg.quality_warn_fwhm_px_median)

        fwhm_p90_warn = float(getattr(cfg, "quality_warn_fwhm_px_p90", 10.0))
        fwhm_spread_warn = float(getattr(cfg, "quality_warn_fwhm_spread_px", 4.0))

        warn_plane_slope = float(cfg.quality_warn_plane_slope_mag_adu_per_tile)
        warn_grad_p95 = float(cfg.quality_warn_grad_p95_adu_per_tile)
        warn_corner = float(cfg.quality_warn_corner_delta_adu)

        warn_signal_obj = float(
            getattr(cfg, "quality_warn_min_obj_minus_bg_adu_s", 0.0)
        )
        warn_signal_neb = float(
            getattr(cfg, "quality_warn_min_nebula_minus_bg_adu_s", 0.0)
        )

        warn_headroom_below = float(
            getattr(cfg, "quality_warn_headroom_p99_below", 0.10)
        )

        # Flags (WARN-level)
        if sf >= sat_warn:
            reason_mask |= RQ_CLIPPING
        if nm < min_stars_warn:
            reason_mask |= RQ_TOO_FEW_STARS
        if ecc >= ecc_warn:
            reason_mask |= RQ_BAD_ECC
        if fwhm >= fwhm_warn:
            reason_mask |= RQ_BAD_FWHM

        # Gradient / structure flags
        slope = (
            float(plane_slope_mag_adu_per_tile)
            if plane_slope_mag_adu_per_tile is not None
            else 0.0
        )
        g95 = float(grad_p95_adu_per_tile) if grad_p95_adu_per_tile is not None else 0.0
        cd = float(corner_delta_adu) if corner_delta_adu is not None else 0.0
        if (
            abs(slope) >= warn_plane_slope
            or abs(g95) >= warn_grad_p95
            or abs(cd) >= warn_corner
        ):
            reason_mask |= RQ_STRONG_GRADIENT

        rms_map = (
            float(bkg2d_rms_of_map_adu) if bkg2d_rms_of_map_adu is not None else 0.0
        )
        if rms_map >= float(cfg.quality_warn_bkg2d_rms_of_map_adu):
            reason_mask |= RQ_HIGH_BKG_STRUCTURE

        # PSF robustness flags
        ep90 = float(ecc_p90) if ecc_p90 is not None else None
        if ep90 is not None and ep90 >= ecc_p90_warn:
            reason_mask |= RQ_PSF_TAIL_ECC

        fp90 = float(fwhm_px_p90) if fwhm_px_p90 is not None else None
        if fp90 is not None and fp90 >= fwhm_p90_warn:
            reason_mask |= RQ_FWHM_P90

        if fwhm_px_p90 is not None and fwhm_px_p10 is not None:
            spread = max(0.0, float(fwhm_px_p90 - fwhm_px_p10))
            if spread >= fwhm_spread_warn:
                reason_mask |= RQ_FWHM_SPREAD

        # Signal flag (only if thresholds are enabled)
        if (
            warn_signal_obj > 0.0
            and obj_minus_bg_adu_s is not None
            and float(obj_minus_bg_adu_s) < warn_signal_obj
        ):
            reason_mask |= RQ_LOW_SIGNAL
        if (
            warn_signal_neb > 0.0
            and nebula_minus_bg_adu_s is not None
            and float(nebula_minus_bg_adu_s) < warn_signal_neb
        ):
            reason_mask |= RQ_LOW_SIGNAL

        # Headroom flag (only if present)
        if headroom_p99 is not None and float(headroom_p99) < warn_headroom_below:
            reason_mask |= RQ_LOW_HEADROOM

        # Scores
        psf_score = _score_psf(
            fwhm_px_median=fwhm_px_median,
            fwhm_px_p10=fwhm_px_p10,
            fwhm_px_p90=fwhm_px_p90,
            ecc_median=ecc_median,
            ecc_p90=ecc_p90,
            n_measured=n_measured,
        )
        clip_score = _score_clip(saturated_pixel_fraction)
        bg_score = _score_bg(
            bkg2d_rms_of_map_adu,
            plane_slope_mag_adu_per_tile,
            grad_p95_adu_per_tile,
            corner_delta_adu,
        )

        signal_score = _score_signal(
            obj_minus_bg_adu_s,
            nebula_minus_bg_adu_s,
            warn_min_obj=warn_signal_obj,
            warn_min_nebula=warn_signal_neb,
        )

        headroom_score = _score_headroom(headroom_p99)

        confidence_score = _score_confidence(
            has_psf=has_psf,
            has_sat=has_sat,
            has_bkg=has_bkg,
            has_roi_signal=has_roi_signal and roi_usable == 1,
            has_masked_signal=has_masked_signal and masked_usable == 1,
            has_headroom=has_headroom and head_usable == 1,
        )

        quality_score = _weighted_quality(
            psf_score=psf_score,
            bg_score=bg_score,
            clip_score=clip_score,
            signal_score=signal_score,
            headroom_score=headroom_score,
            confidence_score=confidence_score,
        )

        # Decision: keep your v1 rules, plus new reject conditions.
        hard_reject = False
        if reason_mask & RQ_MISSING_UPSTREAM:
            hard_reject = True
        if reason_mask & RQ_NAN_OR_INF:
            hard_reject = True
        if (not has_psf) or psf_usable == 0:
            hard_reject = True
        if (not has_sat) or sat_usable == 0:
            hard_reject = True
        if (not has_bkg) or bkg_usable == 0:
            hard_reject = True

        # existing hard thresholds
        if sf >= sat_reject:
            hard_reject = True
        if nm < min_stars_reject:
            hard_reject = True
        if ecc >= ecc_reject:
            hard_reject = True
        if fwhm >= fwhm_reject:
            hard_reject = True

        # NEW hard reject: ecc_p90
        if ep90 is not None and ep90 >= ecc_p90_reject:
            hard_reject = True

        if hard_reject:
            decision = "REJECT"
            quality_score = min(
                int(quality_score), int(cfg.quality_cap_score_on_reject)
            )
        else:
            warn = False
            if sf >= sat_warn:
                warn = True
            if nm < min_stars_warn:
                warn = True
            if ecc >= ecc_warn:
                warn = True
            if fwhm >= fwhm_warn:
                warn = True

            if reason_mask & (
                RQ_STRONG_GRADIENT
                | RQ_HIGH_BKG_STRUCTURE
                | RQ_PSF_TAIL_ECC
                | RQ_FWHM_P90
                | RQ_FWHM_SPREAD
                | RQ_LOW_SIGNAL
                | RQ_LOW_HEADROOM
            ):
                warn = True

            if quality_score < int(cfg.quality_warn_if_score_below):
                warn = True

            if warn:
                decision = "WARN"
                quality_score = min(
                    int(quality_score), int(cfg.quality_cap_score_on_warn)
                )
            else:
                decision = "KEEP"

        primary_reason = _primary_reason_from_mask(int(reason_mask))
        usable = 1 if decision == "KEEP" else 0
        reason = "ok" if decision == "KEEP" else primary_reason

        # Write
        conn.execute(
            """
            INSERT OR REPLACE INTO frame_quality_metrics (
              image_id,
              expected_fields, read_fields, written_fields, parse_warnings, db_written_utc,
              quality_score, decision, reason_mask, primary_reason,
              psf_score, bg_score, clip_score,
              signal_score, headroom_score, confidence_score,
              fwhm_px_median, fwhm_px_p10, fwhm_px_p90,
              ecc_median, ecc_p90, n_measured,
              saturated_pixel_fraction,
              bkg2d_rms_of_map_adu, plane_slope_mag_adu_per_tile, grad_p95_adu_per_tile, corner_delta_adu,
              obj_minus_bg_adu_s, nebula_minus_bg_adu_s, headroom_p99,
              usable, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(image_id),
                int(EXPECTED_FIELDS_AUDIT),
                int(read),
                int(EXPECTED_FIELDS_AUDIT),
                parse_warnings,
                utc_now(),
                int(quality_score),
                str(decision),
                int(reason_mask),
                str(primary_reason),
                int(psf_score),
                int(bg_score),
                int(clip_score),
                int(signal_score),
                int(headroom_score),
                int(confidence_score),
                fwhm_px_median,
                fwhm_px_p10,
                fwhm_px_p90,
                ecc_median,
                ecc_p90,
                n_measured,
                saturated_pixel_fraction,
                bkg2d_rms_of_map_adu,
                plane_slope_mag_adu_per_tile,
                grad_p95_adu_per_tile,
                corner_delta_adu,
                obj_minus_bg_adu_s,
                nebula_minus_bg_adu_s,
                headroom_p99,
                int(usable),
                str(reason),
            ),
        )
        conn.commit()

        elapsed_s = time.perf_counter() - t0

        # Verbose IO block (inputs + outputs)
        verbose_fields: Dict[str, Any] = {
            "__inputs__": {
                "image_id": image_id,
                "has_psf": has_psf,
                "has_sat": has_sat,
                "has_bkg": has_bkg,
                "has_roi_signal": has_roi_signal,
                "has_masked_signal": has_masked_signal,
                "has_headroom": has_headroom,
                "psf_usable": psf_usable,
                "sat_usable": sat_usable,
                "roi_usable": roi_usable,
                "masked_usable": masked_usable,
                "head_usable": head_usable,
                "fwhm_px_median": fwhm_px_median,
                "fwhm_px_p10": fwhm_px_p10,
                "fwhm_px_p90": fwhm_px_p90,
                "ecc_median": ecc_median,
                "ecc_p90": ecc_p90,
                "n_measured": n_measured,
                "saturated_pixel_fraction": saturated_pixel_fraction,
                "nan_fraction": nan_fraction,
                "inf_fraction": inf_fraction,
                "bkg2d_rms_of_map_adu": bkg2d_rms_of_map_adu,
                "plane_slope_mag_adu_per_tile": plane_slope_mag_adu_per_tile,
                "grad_p95_adu_per_tile": grad_p95_adu_per_tile,
                "corner_delta_adu": corner_delta_adu,
                "obj_minus_bg_adu_s": obj_minus_bg_adu_s,
                "nebula_minus_bg_adu_s": nebula_minus_bg_adu_s,
                "headroom_p99": headroom_p99,
            },
            "__outputs__": {
                "quality_score": quality_score,
                "decision": decision,
                "primary_reason": primary_reason,
                "reason_mask": reason_mask,
                "psf_score": psf_score,
                "bg_score": bg_score,
                "clip_score": clip_score,
                "signal_score": signal_score,
                "headroom_score": headroom_score,
                "confidence_score": confidence_score,
            },
        }

        logger.log_module_result(
            MODULE_NAME,
            str(event.file_path),
            expected_read=int(EXPECTED_READ),
            read=int(read),
            expected_written=int(EXPECTED_WRITTEN),
            written=1,
            status="OK",
            duration_s=elapsed_s,
            verbose_fields=verbose_fields,
        )

        return ModuleRunRecord(
            image_id=int(image_id),
            expected_read=int(EXPECTED_READ),
            read=int(read),
            expected_written=int(EXPECTED_WRITTEN),
            written=1,
            status="OK",
            message=f"{decision} score={quality_score} reason={primary_reason}",
        )

    except Exception as e:
        elapsed_s = time.perf_counter() - t0
        logger.log_failure(
            MODULE_NAME,
            str(event.file_path),
            action=cfg.on_metric_failure,
            reason=repr(e),
            duration_s=elapsed_s,
        )
        # If we got far enough to resolve image_id, return a proper ModuleRunRecord.
        # Otherwise, return False to avoid FK failure.
        try:
            # best effort resolve without crashing
            conn2 = sqlite3.connect(str(cfg.db_path))
            try:
                conn2.execute("PRAGMA foreign_keys = ON;")
                rid = _lookup_image_id(conn2, str(event.file_path))
            finally:
                conn2.close()
        except Exception:
            rid = None

        if rid is None:
            return False

        return ModuleRunRecord(
            image_id=int(rid),
            expected_read=int(EXPECTED_READ),
            read=0,
            expected_written=int(EXPECTED_WRITTEN),
            written=0,
            status="FAILED",
            message=repr(e),
        )
    finally:
        conn.close()
