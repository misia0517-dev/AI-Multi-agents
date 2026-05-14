"""
Compare eval results across runs.

Usage:
    python -m evals.compare                              # compare all runs in eval_results/
    python -m evals.compare eval_results/eval_001.json eval_results/eval_002.json  # compare specific runs
"""
from __future__ import annotations

import glob
import json
import os
import sys


def load_results(path: str) -> list[dict]:
    """Load one or more eval results from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    # Flat single result (from main.py or --scenario)
    if isinstance(data, dict) and "overall_score" in data:
        return [data]

    # List of results (from multi-scenario run)
    if isinstance(data, list):
        return data

    # Legacy nested format {"scenario_name": {results}}
    if isinstance(data, dict):
        return list(data.values())

    return [data]


def print_comparison(results: list[tuple[str, dict]]):
    """Print a side-by-side comparison table."""

    # Header
    print(f"\n{'='*80}")
    print("  EVAL RESULTS COMPARISON")
    print(f"{'='*80}\n")

    # Column headers
    names = [label for label, _ in results]
    col_width = max(20, max(len(n) for n in names) + 2)

    header = f"  {'Validator':<30}"
    for name in names:
        header += f"  {name:>{col_width}}"
    print(header)
    print(f"  {'-'*30}" + f"  {'-'*col_width}" * len(names))

    # Collect all validator names from first result
    first_data = results[0][1]
    validators = []
    for r in first_data.get("deterministic_results", []):
        validators.append(r["name"])

    # Rows
    for vname in validators:
        row = f"  {vname:<30}"
        for _, data in results:
            det_results = {r["name"]: r for r in data.get("deterministic_results", [])}
            if vname in det_results:
                r = det_results[vname]
                score = r["score"]
                icon = "PASS" if r["passed"] else "FAIL"
                cell = f"{icon} {score:.0%}"
            else:
                cell = "N/A"
            row += f"  {cell:>{col_width}}"
        print(row)

    # Summary row
    print(f"  {'-'*30}" + f"  {'-'*col_width}" * len(names))

    row = f"  {'OVERALL':<30}"
    for _, data in results:
        score = data.get("overall_score", 0)
        passed = data.get("overall_passed", False)
        icon = "PASS" if passed else "FAIL"
        cell = f"{icon} {score:.1%}"
        row += f"  {cell:>{col_width}}"
    print(row)

    # Metadata
    print(f"\n  {'Agent / Model':<30}")
    for _, data in results:
        provider = data.get("agent_provider", "?")
        model = data.get("model_name", "?")
        cell = f"{provider}/{model}"
        print(f"    {cell}")

    print(f"\n{'='*80}\n")


def main():
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        results_dir = os.path.join(os.path.dirname(__file__), "..", "eval_results")
        if not os.path.isdir(results_dir):
            print("No eval_results/ folder found. Run your agent first to generate results.")
            sys.exit(1)
        paths = sorted(glob.glob(os.path.join(results_dir, "eval_*.json")))

    if not paths:
        print("No eval result files found.")
        sys.exit(1)

    if len(paths) == 1:
        print(f"Only 1 result found. Run your agent multiple times to compare.")

    results = []
    for p in paths:
        try:
            for entry in load_results(p):
                label = entry.get("scenario_name", os.path.basename(p))
                ts = entry.get("timestamp", "")
                if ts:
                    label = f"{label}_{ts[5:10]}"  # add date fragment
                results.append((label, entry))
        except Exception as e:
            print(f"Skipping {p}: {e}")

    if results:
        print_comparison(results)


if __name__ == "__main__":
    main()
