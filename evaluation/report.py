"""
evaluation/report.py
--------------------
Generates two reporting views from evaluation results:
  (a) Per-case pass/fail table with literal category from the JSON file.
  (b) Rolled-up summary table using README reporting categories (via CATEGORY_MAP).

Writes all data to evaluation/results.json for downstream consumption
(e.g. docs/README author can copy real numbers without re-running).

Usage:
    from evaluation.report import generate_report
    generate_report(results)   # results = list[dict] from run_eval.main()

Or run standalone (reads last results from results.json if it exists):
    python -m evaluation.report
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.run_eval import CATEGORY_MAP, README_CATEGORIES

_EVAL_DIR = Path(__file__).parent
_RESULTS_FILE = _EVAL_DIR / "results.json"

TOTAL_EXPECTED = 25


def _build_by_literal_category(results: list[dict]) -> dict[str, dict[str, int]]:
    """
    Aggregate pass/fail counts by literal case category (from JSON files).
    Returns: { "<literal_category>": {"passed": n, "total": n} }
    """
    by_cat: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"passed": 0, "total": 0}
        by_cat[cat]["total"] += 1
        if r["passed"]:
            by_cat[cat]["passed"] += 1
    return by_cat


def _build_by_readme_category(results: list[dict]) -> dict[str, dict[str, int]]:
    """
    Aggregate pass/fail counts by README reporting category (via CATEGORY_MAP).
    A case may be counted in more than one README bucket per CATEGORY_MAP.
    Returns: { "<readme_category>": {"passed": n, "total": n} }
    """
    by_readme: dict[str, dict[str, int]] = {
        cat: {"passed": 0, "total": 0} for cat in README_CATEGORIES
    }

    for r in results:
        literal_cat = r["category"]
        readme_cats = CATEGORY_MAP.get(literal_cat, [])
        for readme_cat in readme_cats:
            if readme_cat in by_readme:
                by_readme[readme_cat]["total"] += 1
                if r["passed"]:
                    by_readme[readme_cat]["passed"] += 1

    return by_readme


def _format_table_row(cells: list[str], widths: list[int]) -> str:
    parts = []
    for cell, w in zip(cells, widths):
        parts.append(cell.ljust(w))
    return "| " + " | ".join(parts) + " |"


def _format_separator(widths: list[int]) -> str:
    return "+-" + "-+-".join("-" * w for w in widths) + "-+"


def print_per_case_table(results: list[dict]) -> None:
    """Print per-case pass/fail table to stdout."""
    print("\n" + "=" * 90)
    print("PER-CASE RESULTS")
    print("=" * 90)

    headers = ["Case ID", "Category", "Result", "Duration", "Failures"]
    widths = [45, 22, 6, 9, 40]

    print(_format_separator(widths))
    print(_format_table_row(headers, widths))
    print(_format_separator(widths))

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        failures_str = "; ".join(r["failures"])[:38] if r["failures"] else ""
        row = [
            r["id"][:43],
            r["category"][:20],
            status,
            f"{r['duration_s']:.1f}s",
            failures_str,
        ]
        print(_format_table_row(row, widths))

    print(_format_separator(widths))


def print_literal_category_table(by_cat: dict[str, dict[str, int]]) -> None:
    """Print per-literal-category summary to stdout."""
    print("\n" + "=" * 50)
    print("BY LITERAL CATEGORY")
    print("=" * 50)

    headers = ["Category", "Passed", "Total", "Rate"]
    widths = [26, 7, 7, 8]

    print(_format_separator(widths))
    print(_format_table_row(headers, widths))
    print(_format_separator(widths))

    for cat in sorted(by_cat):
        d = by_cat[cat]
        rate = f"{100 * d['passed'] // d['total']}%" if d["total"] else "N/A"
        print(_format_table_row([cat, str(d["passed"]), str(d["total"]), rate], widths))

    print(_format_separator(widths))


def print_readme_category_table(by_readme: dict[str, dict[str, int]]) -> None:
    """Print README reporting category summary to stdout."""
    print("\n" + "=" * 50)
    print("BY README REPORTING CATEGORY")
    print("=" * 50)

    headers = ["README Category", "Passed", "Total", "Rate"]
    widths = [18, 7, 7, 8]

    print(_format_separator(widths))
    print(_format_table_row(headers, widths))
    print(_format_separator(widths))

    for cat in README_CATEGORIES:
        d = by_readme.get(cat, {"passed": 0, "total": 0})
        rate = f"{100 * d['passed'] // d['total']}%" if d["total"] else "N/A"
        print(_format_table_row([cat, str(d["passed"]), str(d["total"]), rate], widths))

    print(_format_separator(widths))


def generate_report(results: list[dict]) -> dict[str, Any]:
    """
    Generate and print the full evaluation report.

    Prints:
      (a) Per-case table with literal category.
      (b) Per-literal-category summary.
      (c) README reporting category summary.
      (d) Summary line: "X/25 cases passed."

    Writes evaluation/results.json with the complete structured output.

    Returns the results dict (also written to JSON).
    """
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    by_literal = _build_by_literal_category(results)
    by_readme = _build_by_readme_category(results)

    # ------------------------------------------------------------------
    # Print to stdout
    # ------------------------------------------------------------------
    print_per_case_table(results)
    print_literal_category_table(by_literal)
    print_readme_category_table(by_readme)

    print(f"\n{'=' * 50}")
    print(f"SUMMARY: {passed}/{len(results)} cases passed.")
    if len(results) != TOTAL_EXPECTED:
        print(f"[WARNING] Expected {TOTAL_EXPECTED} cases but ran {len(results)}")
    print("=" * 50)

    # Detailed failure listing
    failing = [r for r in results if not r["passed"]]
    if failing:
        print(f"\nFailing cases ({len(failing)}):")
        for r in failing:
            print(f"  FAIL: {r['id']} [{r['category']}]")
            for f in r["failures"]:
                print(f"        - {f}")

    # ------------------------------------------------------------------
    # Write results.json
    # ------------------------------------------------------------------
    output: dict[str, Any] = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "cases": [
            {
                "id": r["id"],
                "category": r["category"],
                "passed": r["passed"],
                "failures": r["failures"],
                "duration_s": r["duration_s"],
            }
            for r in results
        ],
        "by_literal_category": by_literal,
        "by_readme_category": by_readme,
    }

    _RESULTS_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResults written to {_RESULTS_FILE}")

    return output


if __name__ == "__main__":
    # Standalone: load existing results.json and reprint
    if _RESULTS_FILE.exists():
        data = json.loads(_RESULTS_FILE.read_text(encoding="utf-8"))
        results = data.get("cases", [])
        generate_report(results)
    else:
        print(f"No {_RESULTS_FILE} found. Run 'python -m evaluation.run_eval' first.")
