#!/usr/bin/env python3
# centaur/model_tests/run_all_model_tests.py
#
# Master test runner:
# - Creates one run directory for this invocation
# - Calls each model test script
# - Prints PASS/FAIL/ERROR per model
#
# Usage:
#   # Run all model tests (no dependency check)
#   python3 centaur/model_tests/run_all_model_tests.py --db data/centaurparting.db
#
#   # Explicit run directory
#   python3 centaur/model_tests/run_all_model_tests.py --db data/centaurparting.db \
#       --run-dir data/model_tests/test_results_20251220_123000
#
#   # Include dependency check
#   python3 centaur/model_tests/run_all_model_tests.py --db data/centaurparting.db -dep
#
#   # Only show failures (still runs everything)
#   python3 centaur/model_tests/run_all_model_tests.py --db data/centaurparting.db -of
#
#   # Only show failures + dependency check
#   python3 centaur/model_tests/run_all_model_tests.py --db data/centaurparting.db -of -dep
#
#   # If dependency check fails, dump its JSON for troubleshooting
#   python3 centaur/model_tests/run_all_model_tests.py --db data/centaurparting.db -dep -v
#   python3 centaur/model_tests/run_all_model_tests.py --db data/centaurparting.db -of -dep -v
#
# Flags:
#   -dep   include dependency check (test_model_dependencies.py)
#   -of    only show failures (less output; still runs everything)
#   -v     with -dep: if dependency check FAIL/ERROR, print its JSON (useful for troubleshooting)
#
# Notes:
# - IMPORTANT FIX: we DO NOT forward "--only-failures" to test_model_dependencies.py because it
#   doesn't accept it (argparse would exit 2 -> ERROR). "-of" is handled by this runner only.
# - Dependency test currently manages its own output directory and does not accept --run-dir here.

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def classify_status(code: int, stderr: str) -> str:
    """
    argparse uses exit=2 for CLI usage problems; treat those as ERROR.
    Otherwise, follow our convention:
      0 PASS
      2 FAIL
      else ERROR
    """
    low = (stderr or "").lower()
    if code == 2 and ("unrecognized arguments" in low or "usage:" in low):
        return "ERROR"
    if code == 0:
        return "PASS"
    if code == 2:
        return "FAIL"
    return "ERROR"


def _extract_dep_json_path(stdout: str, stderr: str) -> Optional[Path]:
    """
    Dependency test prints something like:
      [test_model_dependencies] wrote_json=path/to/test_model_dependencies.json
    """
    text = (stdout or "") + "\n" + (stderr or "")
    m = re.search(r"wrote_json\s*=\s*(\S+)", text)
    if not m:
        return None
    try:
        return Path(m.group(1)).expanduser()
    except Exception:
        return None


def _fallback_latest_dep_json() -> Optional[Path]:
    """
    If we couldn't parse wrote_json=..., pick the newest:
      data/model_tests/model_dependencies/test_results_*/test_model_dependencies.json
    """
    root = Path("data") / "model_tests" / "model_dependencies"
    if not root.exists():
        return None
    candidates = list(root.glob("test_results_*/test_model_dependencies.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.parent.stat().st_mtime)


def run_one(
    name: str,
    script_path: Path,
    db_path: Path,
    run_dir: Path,
    *,
    pass_run_dir: bool = True,
    extra_args: Optional[List[str]] = None,
) -> Tuple[str, int, str, str]:
    """
    Returns (name, exit_code, stdout, stderr).
      0 = PASS
      2 = FAIL (script-defined fail)
      1 = ERROR (script-defined error)
      other = ERROR
    """
    cmd = [sys.executable, str(script_path), "--db", str(db_path)]
    if pass_run_dir:
        cmd += ["--run-dir", str(run_dir)]
    if extra_args:
        cmd += list(extra_args)

    p = subprocess.run(cmd, text=True, capture_output=True)
    return name, p.returncode, p.stdout or "", p.stderr or ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all Centaur model tests")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Optional: explicitly set run directory. If blank, auto: data/model_tests/test_results_<STAMP>",
    )

    # Requested flags
    ap.add_argument(
        "-dep",
        "--dep",
        action="store_true",
        help="Include dependency check (test_model_dependencies).",
    )
    ap.add_argument(
        "-of",
        "--only-failures",
        action="store_true",
        help="Only show failures (still runs all tests).",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="With -dep: if dependency check fails, print its JSON (useful for troubleshooting).",
    )

    args = ap.parse_args()

    db_path = Path(args.db)
    stamp = utc_stamp()
    run_dir = (
        Path(args.run_dir)
        if args.run_dir.strip()
        else (Path("data") / "model_tests" / f"test_results_{stamp}")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # (name, rel_path, pass_run_dir)
    tests: List[Tuple[str, str, bool]] = [
        (
            "sky_basic_metrics",
            "centaur/model_tests/models/test_sky_basic_metrics.py",
            True,
        ),
        (
            "sky_background2d_metrics",
            "centaur/model_tests/models/test_sky_background2d_metrics.py",
            True,
        ),
        (
            "saturation_metrics",
            "centaur/model_tests/models/test_saturation_metrics.py",
            True,
        ),
        (
            "signal_structure_metrics",
            "centaur/model_tests/models/test_signal_structure_metrics.py",
            True,
        ),
        (
            "roi_signal_metrics",
            "centaur/model_tests/models/test_roi_signal_metrics.py",
            True,
        ),
        (
            "nebula_mask_metrics",
            "centaur/model_tests/models/test_nebula_mask_metrics.py",
            True,
        ),
        (
            "masked_signal_metrics",
            "centaur/model_tests/models/test_masked_signal_metrics.py",
            True,
        ),
        (
            "star_headroom_metrics",
            "centaur/model_tests/models/test_star_headroom_metrics.py",
            True,
        ),
        (
            "psf_detect_metrics",
            "centaur/model_tests/models/test_psf_detect_metrics.py",
            True,
        ),
        (
            "psf_basic_metrics",
            "centaur/model_tests/models/test_psf_basic_metrics.py",
            True,
        ),
        (
            "psf_grid_metrics",
            "centaur/model_tests/models/test_psf_grid_metrics.py",
            True,
        ),
        (
            "psf_model_metrics",
            "centaur/model_tests/models/test_psf_model_metrics.py",
            True,
        ),
        ("exposure_advice", "centaur/model_tests/models/test_exposure_advice.py", True),
        (
            "fits_header_core",
            "centaur/model_tests/models/test_fits_header_core.py",
            True,
        ),
        (
            "fits_header_full",
            "centaur/model_tests/models/test_fits_header_full.py",
            True,
        ),
        ("module_runs", "centaur/model_tests/models/test_module_runs.py", True),
        ("images_table", "centaur/model_tests/models/test_images.py", True),
        ("flat_profiles", "centaur/model_tests/models/test_flat_profiles.py", True),
        (
            "flat_capture_sets",
            "centaur/model_tests/models/test_flat_capture_sets.py",
            True,
        ),
        ("flat_metrics", "centaur/model_tests/models/test_flat_metrics.py", True),
        (
            "flat_frame_links",
            "centaur/model_tests/models/test_flat_frame_links.py",
            True,
        ),
        (
            "camera_constants",
            "centaur/model_tests/models/test_camera_constants.py",
            True,
        ),
        ("watch_roots", "centaur/model_tests/models/test_watch_roots.py", True),
        ("image_setups", "centaur/model_tests/models/test_image_setups.py", True),
        ("optical_setups", "centaur/model_tests/models/test_optical_setups.py", True),
        (
            "frame_quality",
            "centaur/model_tests/models/test_frame_quality_metrics.py",
            True,
        ),
    ]

    if args.dep:
        # Dependency script currently expects only "--db" and manages its own output directory.
        tests.append(
            (
                "model_dependencies",
                "centaur/model_tests/models/test_model_dependencies.py",
                False,
            )
        )

    print(f"[run_all_model_tests] db={db_path}")
    print(f"[run_all_model_tests] run_dir={run_dir}")
    print(
        f"[run_all_model_tests] dep={'ON' if args.dep else 'OFF'} "
        f"only_failures={'ON' if args.only_failures else 'OFF'} "
        f"verbose={'ON' if args.verbose else 'OFF'}"
    )
    print("")

    results: List[Tuple[str, str, int]] = []
    dep_json_path: Optional[Path] = None
    dep_status: Optional[str] = None

    for name, rel_path, pass_run_dir in tests:
        script_path = Path(rel_path)

        if not args.only_failures:
            print(f"[run_all_model_tests] running {name} ...")

        extra_args: List[str] = []

        # IMPORTANT: do NOT forward "--only-failures" or any other runner-only flags to
        # test_model_dependencies.py; it doesn't accept them and will argparse-error.
        if name == "model_dependencies":
            extra_args = []

        n, code, out, err = run_one(
            name,
            script_path,
            db_path,
            run_dir,
            pass_run_dir=pass_run_dir,
            extra_args=extra_args if extra_args else None,
        )
        status = classify_status(code, err)
        results.append((n, status, code))

        # Capture dependency JSON path for -v troubleshooting
        if name == "model_dependencies":
            dep_status = status
            dep_json_path = (
                _extract_dep_json_path(out, err) or _fallback_latest_dep_json()
            )

        # Output control:
        # - default: show everything
        # - -of: show only failing tests (and their stdout/stderr)
        if args.only_failures and status == "PASS":
            continue

        if out.strip():
            for line in out.rstrip().splitlines():
                print(f"    {line}")
        if err.strip():
            for line in err.rstrip().splitlines():
                print(f"    [stderr] {line}")

        print(f"[run_all_model_tests] {name}: {status} (exit={code})")
        print("")

    any_fail = any(s == "FAIL" for _, s, _ in results)
    any_error = any(s == "ERROR" for _, s, _ in results)

    # Summary: if -of, only show non-PASS lines (or "ALL PASS")
    print("=== SUMMARY ===")
    if args.only_failures:
        non_pass = [(n, s) for n, s, _ in results if s != "PASS"]
        if not non_pass:
            print("ALL PASS")
        else:
            for name, status in non_pass:
                print(f"{name:28s} {status}")
    else:
        for name, status, _code in results:
            print(f"{name:28s} {status}")

    # -v behavior: only with -dep, and only if dependency failed/error.
    # Enhancement: print the JSON path we found (even if we fail to parse it).
    if args.dep and args.verbose and dep_status in ("FAIL", "ERROR"):
        print("")
        print("=== DEPENDENCY DEBUG (-v) ===")
        if dep_json_path:
            print(f"dependency_json_candidate={dep_json_path}")
        if dep_json_path and dep_json_path.exists():
            try:
                data = json.loads(dep_json_path.read_text(encoding="utf-8"))
                print(json.dumps(data, indent=2, sort_keys=True))
            except Exception as e:
                print(
                    f"Could not read/parse dependency JSON at {dep_json_path}: "
                    f"{type(e).__name__}:{e}"
                )
        else:
            print(
                "Could not locate dependency JSON (no wrote_json=... line and no fallback file found)."
            )

    if any_error:
        return 1
    if any_fail:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
