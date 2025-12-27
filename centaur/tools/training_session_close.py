#!/usr/bin/env python3
# centaur/tools/training_session_close.py
from __future__ import annotations

import argparse
import sqlite3
from typing import Any, Dict, Tuple, List, Optional

from centaur.config import default_config
from centaur.training_session_engine import (
    solve_training_session,
    close_training_session,
)


def _round(x: Any, nd: int = 3) -> Any:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return x
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return float(round(x, nd))
        return x
    except Exception:
        return x


def _round_recursive(obj: Any, nd: int = 3) -> Any:
    if isinstance(obj, dict):
        return {str(k): _round_recursive(v, nd=nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_recursive(v, nd=nd) for v in obj]
    return _round(obj, nd=nd)


def _sort_exposure_keys(keys: List[str]) -> List[str]:
    def key_fn(k: str) -> float:
        try:
            return float(k)
        except Exception:
            return 1e18

    return sorted(keys, key=key_fn)


def _fmt_filters(filters: Any) -> str:
    if not filters:
        return "-"
    try:
        s = [str(x).upper().strip() for x in list(filters)]
    except Exception:
        return "-"
    short = []
    for f in s:
        if f.startswith("H"):
            short.append("H")
        elif f.startswith("O"):
            short.append("O")
        elif f.startswith("S"):
            short.append("S")
        else:
            short.append(f[:1])
    return ",".join(short)


def _fmt_float(x: Any, nd: int = 3, width: int = 0) -> str:
    if x is None:
        s = "—"
    else:
        try:
            s = f"{float(x):.{nd}f}"
        except Exception:
            s = "—"
    return s.rjust(width) if width > 0 else s


def _fmt_score(x: Any) -> str:
    return _fmt_float(x, nd=3)


def _fmt_ratio_line(ratio_vs_ha: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in ["SII", "HA", "OIII"]:
        if k in ratio_vs_ha:
            parts.append(f"{k} {_round(ratio_vs_ha[k], 2):.2f}x")
    if not parts:
        for k, v in ratio_vs_ha.items():
            parts.append(f"{k} {_round(v, 2):.2f}x")
    return " | ".join(parts)


def _build_report_text(payload: Dict[str, Any]) -> str:
    """
    Human-friendly multiline report for logs/CLI.

    IMPORTANT CONTRACT:
      - Ratio/science comes from engine result["ratios"] (single source of truth).
      - This file only adapts + formats.
    """
    target = str(payload.get("target_name") or "").strip()
    rec = payload.get("recommended") or {}
    candidates = payload.get("candidates") or []
    excluded = payload.get("excluded") or {}
    ratios = payload.get("ratios") or {}

    rx = rec.get("exptime_s")
    ratio_vs_ha = rec.get("ratio_vs_ha") or {}
    one_liner = rec.get("one_liner") or ""

    lines: List[str] = []
    lines.append(
        f"TRAINING RECOMMENDATION — {target}" if target else "TRAINING RECOMMENDATION"
    )
    if rx is not None:
        lines.append(f"RECOMMENDED: {int(round(float(rx)))}s")
    if ratio_vs_ha:
        lines.append("RATIO vs HA: " + _fmt_ratio_line(ratio_vs_ha))
    if one_liner:
        lines.append(f"SUMMARY: {one_liner}")

    # ----------------------------
    # Ratio comparison (Best / 2nd / Aggregate)
    # ----------------------------
    best = ratios.get("best") or {}
    second = ratios.get("second") or None
    agg = ratios.get("aggregate") or {}
    warnings = ratios.get("warnings") or {}
    used_src = str(ratios.get("source_used") or "").strip()

    if best or second or agg:
        lines.append("")
        lines.append("RATIO COMPARISON (Best / 2nd / Aggregate)")

        def fmt_cmp_row(
            label: str,
            exp: Optional[Any],
            ratio: Optional[Dict[str, Any]],
            E_mean: Optional[Any],
            E_min: Optional[Any],
            head_min: Optional[Any],
            extra: str = "",
        ) -> str:
            exp_s = f"{int(round(float(exp)))}s" if exp is not None else "—"
            ratio_s = _fmt_ratio_line(ratio or {}) if ratio else "—"
            row = (
                f"  {label:<9} {exp_s:>5} | {ratio_s}"
                f" | E_mean {_fmt_float(E_mean, nd=3, width=6)}"
                f" E_min {_fmt_float(E_min, nd=3, width=6)}"
                f" head_min {_fmt_float(head_min, nd=3, width=6)}"
            )
            if extra:
                row += f" | {extra}"
            return row

        # BEST
        lines.append(
            fmt_cmp_row(
                "BEST",
                best.get("exptime_s"),
                best.get("ratio_vs_ha"),
                best.get("E_mean"),
                best.get("E_min"),
                best.get("headroom_min"),
            )
        )

        # 2ND
        if isinstance(second, dict) and second:
            ds = second.get("delta_score")
            extra = f"Δscore {_fmt_float(ds, nd=3)}" if ds is not None else ""
            lines.append(
                fmt_cmp_row(
                    "2ND",
                    second.get("exptime_s"),
                    second.get("ratio_vs_ha"),
                    second.get("E_mean"),
                    second.get("E_min"),
                    second.get("headroom_min"),
                    extra=extra,
                )
            )
        else:
            lines.append("  2ND        — | —")

        # AGGREGATE
        if isinstance(agg, dict) and agg:
            exps = agg.get("exposures_used") or agg.get("exposures") or []
            exp_list: List[str] = []
            for e in exps:
                try:
                    exp_list.append(str(int(round(float(e)))))
                except Exception:
                    exp_list.append(str(e))
            exp_list_s = ",".join(exp_list) if exp_list else ""
            extra = f"exps {exp_list_s}" if exp_list_s else ""
            lines.append(
                fmt_cmp_row(
                    "AGGREGATE",
                    None,
                    agg.get("ratio_vs_ha"),
                    agg.get("E_mean"),
                    agg.get("E_min"),
                    agg.get("headroom_min"),
                    extra=extra,
                )
            )
            lines.append(
                "    ↳ Aggregate is averaged across all eligible exposures (stabilizes ratios when you only have 1 frame per filter per exposure)."
            )
        else:
            lines.append("  AGGREGATE  — | —")

        if used_src:
            lines.append(f"    ↳ Recommended ratio source: {used_src}")

        tw = warnings.get("transparency")
        if isinstance(tw, str) and tw.strip():
            lines.append(f"    ↳ Warning: {tw.strip()}")

    # ----------------------------
    # Candidates table
    # ----------------------------
    lines.append("")
    lines.append("CANDIDATES")
    lines.append("  exp   pick     score   E_mean  E_min   headroom_min  filters  note")

    for c in candidates:
        exp = c.get("exptime_s")
        exp_s = f"{int(exp):>3d}s" if isinstance(exp, int) else f"{str(exp):>4}"
        chosen = bool(c.get("chosen"))
        reason = str(c.get("reason") or "").strip()

        score = c.get("score")
        E_mean = c.get("E_mean")
        E_min = c.get("E_min")
        head = c.get("min_linear_headroom_p99")

        if chosen:
            pick = "✅ chosen "
        else:
            if c.get("excluded_reasons") or score is None:
                pick = "❌ excl. "
            else:
                pick = "❌ not   "

        filt = _fmt_filters(c.get("filters"))

        row = (
            f"  {exp_s}  {pick}"
            f" {_fmt_score(score):>6} "
            f" {_fmt_float(E_mean):>6} "
            f" {_fmt_float(E_min):>6} "
            f" {_fmt_float(head):>12}  "
            f"{filt:>7}  "
            f"{reason}"
        )
        lines.append(row)

        exr = c.get("excluded_reasons")
        if exr:
            try:
                exr_list = list(exr)
            except Exception:
                exr_list = [str(exr)]
            for r in exr_list:
                lines.append(f"           ↳ {r}")

    if excluded:
        lines.append("")
        lines.append("EXCLUDED (raw)")
        for k in _sort_exposure_keys(list(excluded.keys())):
            lines.append(f"  {k}s: {', '.join([str(x) for x in excluded[k]])}")

    return "\n".join(lines)


def solve_session_and_optionally_close(
    *,
    db_path: str,
    session_id: int,
    close_session: bool,
) -> Tuple[bool, Dict[str, Any]]:
    cfg = default_config()

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA foreign_keys = ON;")

        # Engine is the single source of truth for metrics/ratios/scoring.
        result = solve_training_session(
            con,
            int(session_id),
            required_filters_default=list(
                getattr(cfg, "training_required_filters_default", ["SII", "HA", "OIII"])
            ),
            min_linear_headroom_default=float(
                getattr(cfg, "training_min_linear_headroom_p99", 0.10)
            ),
            w_mean_default=float(getattr(cfg, "training_score_w_mean", 0.55)),
            w_min_default=float(getattr(cfg, "training_score_w_min", 0.45)),
            short_penalty_default=float(getattr(cfg, "training_short_penalty", 0.10)),
            long_penalty_default=float(getattr(cfg, "training_long_penalty", 0.08)),
            include_excluded_default=bool(
                getattr(cfg, "training_include_excluded", True)
            ),
            close_call_delta_default=float(
                getattr(cfg, "training_close_call_delta", 0.05)
            ),
            # NEW defaults (1-3) - only used if engine supports them (it does in your current file)
            sky_limited_target_ratio_default=float(
                getattr(cfg, "training_sky_limited_target_ratio", 10.0)
            ),
            sky_limited_penalty_weight_default=float(
                getattr(cfg, "training_sky_limited_penalty_weight", 0.08)
            ),
            transparency_variation_warn_frac_default=float(
                getattr(cfg, "training_transparency_variation_warn_frac", 0.35)
            ),
            prefer_aggregate_ratio_when_unstable_default=bool(
                getattr(cfg, "training_prefer_aggregate_ratio_when_unstable", True)
            ),
            aggregate_ratio_weight_mode_default=str(
                getattr(cfg, "training_aggregate_ratio_weight_mode", "nebula_over_sky")
            ),
        )

        if not result.get("ok"):
            return False, {
                "error": result.get("message") or "solve_failed",
                "details": result,
            }

        if close_session:
            with con:
                close_training_session(con, int(session_id))

        session = result.get("session") or {}
        best = result.get("best") or {}
        exposure_report = result.get("exposure_report") or {}
        excluded = result.get("excluded") or {}
        ratios = result.get("ratios") or {}

        recommended_exptime_s = best.get("recommended_exptime_s")
        ratio_vs_ha = best.get("ratio_vs_ha") or {}
        time_fractions = best.get("time_fractions") or {}

        one_liner = ""
        if recommended_exptime_s is not None:
            rx = float(recommended_exptime_s)
            ratio_str = " ".join(
                [f"{k}:{float(v):.2f}x" for k, v in ratio_vs_ha.items()]
            )
            one_liner = f"{rx:.0f}s | {ratio_str}".strip()

        # Candidates list from exposure_report (engine-generated, no duplication)
        candidates: List[Dict[str, Any]] = []
        for exp_key in _sort_exposure_keys(list(exposure_report.keys())):
            rep = exposure_report.get(exp_key) or {}
            candidates.append(
                {
                    "exptime_s": (
                        int(float(exp_key))
                        if str(exp_key).replace(".", "", 1).isdigit()
                        else exp_key
                    ),
                    "filters": rep.get("filters"),
                    "min_linear_headroom_p99": rep.get("min_linear_headroom_p99"),
                    "E_mean": rep.get("E_mean"),
                    "E_min": rep.get("E_min"),
                    "score": rep.get("score"),
                    "frames_used": rep.get("frames_used"),
                    "chosen": rep.get("chosen"),
                    "reason": rep.get("reason"),
                    "excluded_reasons": rep.get("excluded_reasons"),
                }
            )

        ratio_used_source = best.get("ratio_source") or ratios.get("source_used")

        summary: Dict[str, Any] = {
            "session_id": int(session_id),
            "target_name": session.get("target_name", ""),
            "recommended": {
                "exptime_s": recommended_exptime_s,
                "ratio_vs_ha": ratio_vs_ha,
                "time_fractions": time_fractions,
                "score": best.get("score"),
                "E_mean": best.get("E_mean"),
                "E_min": best.get("E_min"),
                "frames_used": best.get("frames_used"),
                "one_liner": one_liner,
                "ratio_used_source": ratio_used_source,
            },
            # Canonical: straight from engine
            "ratios": ratios,
            "candidates": candidates,
        }

        if excluded:
            summary["excluded"] = excluded

        summary = _round_recursive(summary, nd=3)
        summary["report_text"] = _build_report_text(summary)

        return True, summary

    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Solve a training session and optionally close it."
    )
    ap.add_argument("--db", required=True, help="Path to sqlite DB")
    ap.add_argument("--session-id", type=int, required=True, help="training_session_id")
    ap.add_argument("--close", action="store_true", help="Close session after solving")
    args = ap.parse_args()

    ok, payload = solve_session_and_optionally_close(
        db_path=str(args.db),
        session_id=int(args.session_id),
        close_session=bool(args.close),
    )

    if ok:
        print(
            payload.get("report_text")
            or (payload.get("recommended") or {}).get("one_liner")
            or "ok"
        )
        return 0

    print(f"ERROR: {payload.get('error')}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
