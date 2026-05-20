"""
Agent Output Adapters
======================
Bridges between real agent implementations and the eval framework.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .schemas import AgentPipelineOutput, parse_agent_output
from .validators import run_all_validators
from .llm_judges import JudgeFn
from .runner import EvalReport, evaluate_output


# ═══════════════════════════════════════════════════════════════════════════
# Path 1: LangGraph State → Eval
# ═══════════════════════════════════════════════════════════════════════════

def from_langgraph_state(
    state: Dict[str, Any],
    agent_provider: str = "unknown",
    model_name: str = "unknown",
) -> AgentPipelineOutput:
    """
    Convert SeeWeeS-Mia LangGraph final state into the eval schema.
    """

    # Your graph.py returns this as a dict:
    # {
    #   "C1_I95_NJ_BOS": {...risk...},
    #   "C2_NJ_PHL": {...risk...}
    # }
    weather_dict = state.get("corridor_weather_risk", {}) or {}

    corridor_weather_risks = [
        {
            "corridor_id": corridor_id,
            **risk,
        }
        for corridor_id, risk in weather_dict.items()
        if isinstance(risk, dict)
    ]

    validation_violations = state.get("validation_violations", []) or []
    planner_retry_count = state.get("planner_retry_count", 0) or 0

    mapped = {
        "dispatch_plan": state.get("dispatch_plan", ""),
        "report_html": state.get("report_html", ""),
        "business_context": state.get("business_context", ""),
        "ops_insights": state.get("ops_insights", ""),
        "csv_kpis": state.get("csv_kpis", {}),
        "anomalies_md": state.get("anomalies_md", ""),

        # Corridor list for evals
        "corridors": list(weather_dict.keys()),

        # Metadata
        "agent_provider": agent_provider,
        "model_name": model_name,

        # Fixed mapping: graph.py uses corridor_weather_risk, eval expects corridor_weather_risks
        "corridor_weather_risks": corridor_weather_risks,

        # Resource allocation already matches your graph.py
        "resource_allocation": state.get("resource_allocation", {}),

        # Fixed mapping: graph.py uses validation_violations + planner_retry_count
        "validation_result": {
            "passed": len(validation_violations) == 0,
            "violations": validation_violations,
            "retry_count": planner_retry_count,
        },
    }

    return parse_agent_output(mapped)


# ═══════════════════════════════════════════════════════════════════════════
# Path 2: JSON file → Eval
# ═══════════════════════════════════════════════════════════════════════════

def capture_state_to_json(state: Dict[str, Any], filepath: str, **metadata):
    """
    Save a LangGraph final state to JSON for offline eval.
    """

    serializable = {}

    for key, value in state.items():
        try:
            json.dumps(value)
            serializable[key] = value
        except (TypeError, ValueError):
            serializable[key] = str(value)

    serializable.update(metadata)

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"Saved agent output to {filepath}")
    print(f"Run eval with: python -m evals --input {filepath}")


# ═══════════════════════════════════════════════════════════════════════════
# Path 3: Python API
# ═══════════════════════════════════════════════════════════════════════════

def quick_eval(
    state: Dict[str, Any],
    provider: str = "unknown",
    model: str = "unknown",
    judge_fn: Optional[JudgeFn] = None,
    threshold: float = 0.70,
    print_report: bool = True,
) -> EvalReport:
    """
    One-line eval: pass in your LangGraph state, get a scored report.
    """

    output = from_langgraph_state(state, provider, model)
    report = evaluate_output(output, "live_run", judge_fn, threshold)

    if print_report:
        print(report.summary_str())

    return report


def quick_eval_json(
    json_path: str,
    judge_fn: Optional[JudgeFn] = None,
    threshold: float = 0.70,
    print_report: bool = True,
) -> EvalReport:
    """
    Eval from a saved JSON file.
    """

    with open(json_path) as f:
        raw = json.load(f)

    output = parse_agent_output(raw)

    report = evaluate_output(
        output,
        scenario_name=raw.get("scenario_name", "custom"),
        judge_fn=judge_fn,
        pass_threshold=threshold,
    )

    if print_report:
        print(report.summary_str())

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: validators-only
# ═══════════════════════════════════════════════════════════════════════════

def validators_only(state: Dict[str, Any], print_results: bool = True) -> list:
    """
    Run just the deterministic validators.
    """

    output = from_langgraph_state(state)
    results = run_all_validators(output)

    if print_results:
        for r in results:
            icon = "PASS" if r.passed else "FAIL"
            print(f"  [{icon}] {r.name}: {r.score:.1%} — {r.details[:80]}")

    return results