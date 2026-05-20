from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()  # must be before importing graph/agents
from tracing import init_langsmith_tracing
init_langsmith_tracing()  # must be before importing graph/agents
from graph import build_graph


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _run_embedded_evals(final_state: dict) -> None:
    from evals.adapter_mia import quick_eval_mia

    report = quick_eval_mia(
        final_state,
        provider="anthropic",
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        print_report=True,
    )

    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{timestamp}.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"\n=== EVAL RESULTS SAVED ===\n{out_path}")


if __name__ == "__main__":

    app = build_graph()

    state = {
        "pdf_path":      "data-for-enhancement/playbook_variant_02_weather_escalation_score2.md",
        "csv_path":      "data-for-enhancement/Incoming_shipments_14d_multi_corridor.csv",
        "resource_path": "data-for-enhancement/Resource_availability_48h.csv",
    }

    final = app.invoke(state)

    report_html = final.get("report_html", "")
    print("\n=== REPORT (first 2000 chars) ===\n")
    print(report_html[:2000])

    if _env_enabled("RUN_EVALS"):
        _run_embedded_evals(final)
