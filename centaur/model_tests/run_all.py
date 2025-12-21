from __future__ import annotations

import argparse
from pathlib import Path

from centaur.model_tests.common import write_result, fail
from centaur.model_tests.model_registry import get_registry


def main() -> int:
    ap = argparse.ArgumentParser(description="Centaur model test runner (PASS/FAIL per model + artifacts).")
    ap.add_argument("--db", type=str, default="data/centaurparting.db")
    ap.add_argument("--outdir", type=str, default="data/model_test_results")
    args = ap.parse_args()

    db_path = Path(args.db)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tests = get_registry()
    any_fail = False

    print(f"[model_tests] db={db_path} tests={len(tests)} outdir={outdir}")

    for t in tests:
        out_path = outdir / f"{t.name}.json"
        try:
            res = t.test_func(db_path=db_path, out_path=out_path)
        except Exception as e:
            any_fail = True
            print(f"FAIL  {t.name}  exception={type(e).__name__}:{e}")
            res = fail(t.name, "exception", exception=f"{type(e).__name__}:{e}")
            write_result(out_path, res)
            continue

        if res.passed:
            print(f"PASS  {t.name}  {res.summary}")
        else:
            any_fail = True
            print(f"FAIL  {t.name}  {res.summary}  -> {out_path}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
