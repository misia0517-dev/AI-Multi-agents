"""
Eval Runner & Scoring Aggregator
==================================
Orchestrates all evaluation dimensions, computes weighted scores,
and produces a detailed pass/fail report.

Usage:
    python -m evals.runner                         # run all scenarios (deterministic only)
    python -m evals.runner --scenario happy_path   # run one scenario
    python -m evals.runner --with-judges openai    # include LLM judges
    python -m evals.runner --output results.json   # save results to file
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schemas import AgentPipelineOutput, parse_agent_output
from .validators import EvalResult, run_all_validators
from .llm_judges import (
    JudgeFn,
    run_all_judges,
    make_openai_judge,
    make_anthropic_judge,
    make_mock_judge,
)
from .fixtures import ALL_SCENARIOS


# ---------------------------------------------------------------------------
# Scoring aggregator
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    """Aggregated evaluation report for a single agent output."""
    scenario_name: str
    agent_provider: str
    model_name: str
    timestamp: str = ""

    # Results
    deterministic_results: List[EvalResult] = field(default_factory=list)
    judge_results: List[EvalResult] = field(default_factory=list)

    # Aggregate scores
    deterministic_score: float = 0.0
    judge_score: float = 0.0
    overall_score: float = 0.0
    overall_passed: bool = False

    # Thresholds
    pass_threshold: float = 0.70
    critical_validators: List[str] = field(default_factory=lambda: [
        "sla_compliance",
        "cold_chain_coverage",
        "weather_faithfulness",
    ])

    def compute_scores(self):
        """Compute weighted aggregate scores."""
        # Deterministic score (weighted average)
        det_weighted_sum = sum(r.score * r.weight for r in self.deterministic_results)
        det_weight_total = sum(r.weight for r in self.deterministic_results) or 1.0
        self.deterministic_score = det_weighted_sum / det_weight_total

        # Judge score (weighted average)
        if self.judge_results:
            j_weighted_sum = sum(r.score * r.weight for r in self.judge_results)
            j_weight_total = sum(r.weight for r in self.judge_results) or 1.0
            self.judge_score = j_weighted_sum / j_weight_total
        else:
            self.judge_score = -1.0  # not evaluated

        # Overall: 70% deterministic + 30% judges (if available)
        if self.judge_results:
            self.overall_score = 0.70 * self.deterministic_score + 0.30 * self.judge_score
        else:
            self.overall_score = self.deterministic_score

        # Pass/fail: score threshold + no critical failures
        critical_passed = all(
            r.passed for r in self.deterministic_results
            if r.name in self.critical_validators
        )
        self.overall_passed = (
            self.overall_score >= self.pass_threshold and critical_passed
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "agent_provider": self.agent_provider,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "overall_score": round(self.overall_score, 4),
            "overall_passed": self.overall_passed,
            "deterministic_score": round(self.deterministic_score, 4),
            "judge_score": round(self.judge_score, 4) if self.judge_score >= 0 else "N/A",
            "pass_threshold": self.pass_threshold,
            "deterministic_results": [r.to_dict() for r in self.deterministic_results],
            "judge_results": [r.to_dict() for r in self.judge_results],
        }

    def summary_str(self) -> str:
        """Human-readable summary."""
        status = "PASS" if self.overall_passed else "FAIL"
        lines = [
            f"\n{'='*60}",
            f"  EVAL REPORT: {self.scenario_name}",
            f"  Agent: {self.agent_provider}/{self.model_name}",
            f"  Status: {status} (score: {self.overall_score:.1%})",
            f"{'='*60}",
            f"",
            f"  Deterministic Score: {self.deterministic_score:.1%}",
        ]
        if self.judge_score >= 0:
            lines.append(f"  LLM Judge Score:    {self.judge_score:.1%}")
        lines.append(f"  Overall Score:      {self.overall_score:.1%} (threshold: {self.pass_threshold:.0%})")
        lines.append("")

        # Deterministic results
        lines.append("  DETERMINISTIC VALIDATORS:")
        for r in self.deterministic_results:
            icon = "PASS" if r.passed else "FAIL"
            lines.append(f"    [{icon}] {r.name}: {r.score:.1%} (w={r.weight}) — {r.details[:80]}")

        # Judge results
        if self.judge_results:
            lines.append("")
            lines.append("  LLM JUDGES:")
            for r in self.judge_results:
                icon = "PASS" if r.passed else "FAIL"
                lines.append(f"    [{icon}] {r.name}: {r.score:.1%} — {r.details[:80]}")

        lines.append(f"\n{'='*60}\n")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def evaluate_output(
    output: AgentPipelineOutput,
    scenario_name: str = "custom",
    judge_fn: Optional[JudgeFn] = None,
    pass_threshold: float = 0.70,
) -> EvalReport:
    """
    Run full evaluation on a single agent output.

    Args:
        output: Parsed agent output (AgentPipelineOutput)
        scenario_name: Name for this eval run
        judge_fn: Optional LLM judge function. If None, skip judge evals.
        pass_threshold: Minimum overall score to pass (0.0–1.0)

    Returns:
        EvalReport with all results and aggregate scores
    """
    report = EvalReport(
        scenario_name=scenario_name,
        agent_provider=output.agent_provider,
        model_name=output.model_name,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        pass_threshold=pass_threshold,
    )

    # Run deterministic validators
    report.deterministic_results = run_all_validators(output)

    # Run LLM judges (optional)
    if judge_fn:
        report.judge_results = run_all_judges(output, judge_fn)

    # Compute aggregate scores
    report.compute_scores()

    return report


def evaluate_scenario(
    scenario: Dict[str, Any],
    judge_fn: Optional[JudgeFn] = None,
    pass_threshold: float = 0.70,
) -> EvalReport:
    """Evaluate a fixture scenario dict."""
    output = parse_agent_output(scenario)
    return evaluate_output(
        output,
        scenario_name=scenario.get("scenario_name", "unknown"),
        judge_fn=judge_fn,
        pass_threshold=pass_threshold,
    )


def run_all_scenarios(
    judge_fn: Optional[JudgeFn] = None,
    pass_threshold: float = 0.70,
    scenarios: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, EvalReport]:
    """Run eval across all fixture scenarios. Returns dict of reports."""
    if scenarios is None:
        scenarios = ALL_SCENARIOS

    reports = {}
    for name, scenario in scenarios.items():
        reports[name] = evaluate_scenario(scenario, judge_fn, pass_threshold)

    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SeeWeeS Agent-Agnostic Evaluation Runner"
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Run a specific scenario (e.g., happy_path, sla_breach)",
    )
    parser.add_argument(
        "--with-judges", type=str, default=None,
        choices=["openai", "anthropic", "mock"],
        help="Include LLM-as-judge evaluators using specified provider",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.70,
        help="Pass/fail threshold (0.0–1.0, default 0.70)",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Evaluate a custom agent output from JSON file instead of fixtures",
    )

    args = parser.parse_args()

    # Set up judge
    judge_fn = None
    if args.with_judges == "openai":
        judge_fn = make_openai_judge()
    elif args.with_judges == "anthropic":
        judge_fn = make_anthropic_judge()
    elif args.with_judges == "mock":
        judge_fn = make_mock_judge()

    # Run evals
    if args.input:
        # Evaluate custom agent output
        with open(args.input) as f:
            raw = json.load(f)
        output = parse_agent_output(raw)
        report = evaluate_output(output, "custom", judge_fn, args.threshold)
        reports = {"custom": report}
    elif args.scenario:
        if args.scenario not in ALL_SCENARIOS:
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {', '.join(ALL_SCENARIOS.keys())}")
            sys.exit(1)
        report = evaluate_scenario(ALL_SCENARIOS[args.scenario], judge_fn, args.threshold)
        reports = {args.scenario: report}
    else:
        reports = run_all_scenarios(judge_fn, args.threshold)

    # Print results
    all_passed = True
    for name, report in reports.items():
        print(report.summary_str())
        if not report.overall_passed:
            all_passed = False

    # Summary
    total = len(reports)
    passed = sum(1 for r in reports.values() if r.overall_passed)
    print(f"\nSUITE SUMMARY: {passed}/{total} scenarios passed")

    # Save to file
    if args.output:
        if len(reports) == 1:
            # Single scenario — save flat (consistent with main.py's format)
            results_dict = next(iter(reports.values())).to_dict()
        else:
            # Multiple scenarios — save as list for compare tool
            results_dict = [r.to_dict() for r in reports.values()]
        with open(args.output, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results saved to {args.output}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
