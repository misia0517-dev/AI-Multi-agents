"""
Deterministic Validators
========================
Pure-function constraint checkers that require NO LLM calls.
Each validator returns an EvalResult with pass/fail, score, and details.

These are agent-agnostic: they test the output contract, not the implementation.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .schemas import AgentPipelineOutput, CorridorWeatherRisk, CorridorAllocation


# ---------------------------------------------------------------------------
# Eval result type
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of a single evaluation check."""
    name: str
    passed: bool
    score: float          # 0.0 – 1.0
    weight: float = 1.0   # importance weight for aggregation
    details: str = ""
    sub_results: List["EvalResult"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "passed": self.passed,
            "score": round(self.score, 4),
            "weight": self.weight,
            "details": self.details,
        }
        if self.sub_results:
            d["sub_results"] = [r.to_dict() for r in self.sub_results]
        return d


# ---------------------------------------------------------------------------
# Domain constants (from SeeWeeS Playbook)
# ---------------------------------------------------------------------------

BUFFER_POLICY = {0: 0.0, 1: 0.10, 2: 0.25, 3: 0.40}
TRUCK_CAPACITY = 10
PACKING_BUFFER = 0.10
EFFECTIVE_CAPACITY = TRUCK_CAPACITY * (1 - PACKING_BUFFER)  # 9.0

SLA_TIERS = {"Tier 1": 6.0, "Tier 2": 12.0}  # max hours

WEATHER_THRESHOLDS = {
    "heavy_rain": 15.0,     # mm/day
    "high_wind": 45.0,      # km/h
    "freezing": 0.0,        # °C (≤ 0)
}

COST_CEILING_PER_CORRIDOR = 3000.0  # $/day
UTILIZATION_WARNING = 95.0           # %
UTILIZATION_TARGET_LOW = 75.0
UTILIZATION_TARGET_HIGH = 90.0
FAIRNESS_RATIO_FLOOR = 0.7


# ═══════════════════════════════════════════════════════════════════════════
# 1. SLA Compliance Validator
# ═══════════════════════════════════════════════════════════════════════════

def validate_sla_compliance(output: AgentPipelineOutput) -> EvalResult:
    """
    Check: For every corridor with Tier 1 shipments, adjusted_travel_hrs ≤ 6.
    For Tier 2, ≤ 12 hours.
    """
    issues = []
    corridor_count = 0

    for wr in output.corridor_weather_risks:
        corridor_count += 1
        travel = wr.adjusted_travel_hrs
        if travel is None:
            issues.append(f"{wr.corridor_id}: missing adjusted_travel_hrs")
            continue

        # Tier 1 check (6 hr SLA)
        if travel > SLA_TIERS["Tier 1"]:
            issues.append(
                f"{wr.corridor_id}: adjusted_travel_hrs={travel:.1f}h exceeds "
                f"Tier 1 SLA ({SLA_TIERS['Tier 1']}h)"
            )

    if corridor_count == 0:
        return EvalResult(
            name="sla_compliance",
            passed=False, score=0.0, weight=2.0,
            details="No corridor weather data to check SLA against",
        )

    # Check dispatch plan mentions SLA breaches
    plan_lower = output.dispatch_plan.lower()
    sla_mentioned = any(term in plan_lower for term in ["sla", "breach", "tier 1", "tier-1", "6 hour", "6-hour", "6hr"])

    breach_corridors = [i for i in issues if "exceeds" in i]
    plan_flags_breaches = sla_mentioned if breach_corridors else True

    score = 1.0
    if breach_corridors:
        # Partial credit if the plan at least acknowledges the breach
        score = 0.5 if plan_flags_breaches else 0.0

    if not plan_flags_breaches and breach_corridors:
        issues.append("Dispatch plan does NOT flag SLA breach — critical omission")

    passed = len(breach_corridors) == 0 or plan_flags_breaches
    return EvalResult(
        name="sla_compliance",
        passed=passed,
        score=score,
        weight=2.0,
        details="; ".join(issues) if issues else "All corridors within SLA limits",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Cold-Chain Coverage Validator
# ═══════════════════════════════════════════════════════════════════════════

def validate_cold_chain(output: AgentPipelineOutput) -> EvalResult:
    """
    Check: temp_truck_capacity ≥ cold_chain_demand * (1 + buffer)
    for every corridor. Coverage must be 100%.
    """
    if not output.resource_allocation:
        return EvalResult(
            name="cold_chain_coverage",
            passed=False, score=0.0, weight=2.0,
            details="No resource allocation data available",
        )

    issues = []
    scores = []

    for cid, alloc in output.resource_allocation.allocations.items():
        cold_demand = alloc.cold_chain_demand
        buffer = alloc.buffer_pct
        buffered_cold = cold_demand * (1 + buffer)
        temp_cap = alloc.temp_truck_capacity

        coverage = min(temp_cap / max(buffered_cold, 0.001), 1.0) * 100

        if coverage < 100.0:
            issues.append(
                f"{cid}: cold-chain coverage={coverage:.1f}% "
                f"(need {buffered_cold:.1f} vol, have {temp_cap:.1f} cap)"
            )
            scores.append(coverage / 100.0)
        else:
            scores.append(1.0)

    avg_score = sum(scores) / max(len(scores), 1)
    passed = len(issues) == 0
    return EvalResult(
        name="cold_chain_coverage",
        passed=passed,
        score=avg_score,
        weight=2.0,
        details="; ".join(issues) if issues else "100% cold-chain coverage across all corridors",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Buffer Policy Compliance Validator
# ═══════════════════════════════════════════════════════════════════════════

def validate_buffer_policy(output: AgentPipelineOutput) -> EvalResult:
    """
    Check: buffer_pct matches the risk score per the playbook policy.
    risk 0→0%, 1→10%, 2→25%, 3→40%.
    """
    if not output.resource_allocation:
        return EvalResult(
            name="buffer_policy",
            passed=False, score=0.0, weight=1.5,
            details="No resource allocation data",
        )

    issues = []
    for cid, alloc in output.resource_allocation.allocations.items():
        risk = alloc.weather_risk_score
        expected_buffer = BUFFER_POLICY.get(risk, 0.0)
        actual_buffer = alloc.buffer_pct

        if abs(actual_buffer - expected_buffer) > 0.001:
            issues.append(
                f"{cid}: risk={risk} → expected buffer={expected_buffer:.0%}, "
                f"got {actual_buffer:.0%}"
            )

    passed = len(issues) == 0
    score = 1.0 if passed else max(0.0, 1.0 - len(issues) * 0.5)
    return EvalResult(
        name="buffer_policy",
        passed=passed,
        score=score,
        weight=1.5,
        details="; ".join(issues) if issues else "Buffer policy correctly applied",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Resource Utilization Validator
# ═══════════════════════════════════════════════════════════════════════════

def validate_utilization(output: AgentPipelineOutput) -> EvalResult:
    """
    Check: utilization 75-90% is ideal. >95% is critical overload.
    Driver count ≥ total trucks assigned.
    """
    if not output.resource_allocation:
        return EvalResult(
            name="resource_utilization",
            passed=False, score=0.0, weight=1.5,
            details="No resource allocation data",
        )

    issues = []
    scores = []

    for cid, alloc in output.resource_allocation.allocations.items():
        util = alloc.utilization_pct
        total_trucks = alloc.truck_standard + alloc.truck_temp_controlled
        drivers = alloc.driver

        # Utilization scoring
        if util > UTILIZATION_WARNING:
            issues.append(f"{cid}: critically overloaded at {util:.1f}% utilization")
            scores.append(0.3)
        elif UTILIZATION_TARGET_LOW <= util <= UTILIZATION_TARGET_HIGH:
            scores.append(1.0)
        elif util < UTILIZATION_TARGET_LOW:
            scores.append(0.7)  # under-utilized but acceptable
        else:
            scores.append(0.8)  # 90-95%, slightly high

        # Driver sufficiency
        if drivers < total_trucks:
            issues.append(
                f"{cid}: driver shortage — {drivers} drivers for {total_trucks} trucks"
            )
            scores.append(0.0)

    avg_score = sum(scores) / max(len(scores), 1)
    passed = all("critically" not in i and "shortage" not in i for i in issues)
    return EvalResult(
        name="resource_utilization",
        passed=passed,
        score=avg_score,
        weight=1.5,
        details="; ".join(issues) if issues else "Utilization within acceptable range",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Cost Guardrail Validator
# ═══════════════════════════════════════════════════════════════════════════

def validate_cost_guardrails(output: AgentPipelineOutput) -> EvalResult:
    """
    Check: estimated_daily_cost ≤ $3,000 per corridor.
    """
    if not output.resource_allocation:
        return EvalResult(
            name="cost_guardrails",
            passed=False, score=0.0, weight=1.0,
            details="No resource allocation data",
        )

    issues = []
    for cid, alloc in output.resource_allocation.allocations.items():
        cost = alloc.estimated_daily_cost
        if cost > COST_CEILING_PER_CORRIDOR:
            overage = cost - COST_CEILING_PER_CORRIDOR
            issues.append(
                f"{cid}: cost=${cost:.0f}/day exceeds ceiling "
                f"(${COST_CEILING_PER_CORRIDOR:.0f}), over by ${overage:.0f}"
            )

    passed = len(issues) == 0
    score = 1.0 if passed else max(0.0, 1.0 - len(issues) * 0.25)
    return EvalResult(
        name="cost_guardrails",
        passed=passed,
        score=score,
        weight=1.0,
        details="; ".join(issues) if issues else "All corridors within cost ceiling",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Fairness Metrics Validator
# ═══════════════════════════════════════════════════════════════════════════

def validate_fairness(output: AgentPipelineOutput) -> EvalResult:
    """
    Check: max-min fairness ratio ≥ 0.7 (from academic reference).
    NSW score should be > 0.
    """
    if not output.resource_allocation:
        return EvalResult(
            name="fairness_metrics",
            passed=False, score=0.0, weight=1.0,
            details="No resource allocation data",
        )

    ratio = output.resource_allocation.max_min_fairness_ratio
    nsw = output.resource_allocation.nsw_score

    issues = []
    if ratio < FAIRNESS_RATIO_FLOOR:
        issues.append(
            f"Max-min fairness ratio={ratio:.3f} < {FAIRNESS_RATIO_FLOOR} — unfair allocation"
        )
    if nsw <= 0:
        issues.append(f"NSW score={nsw:.4f} ≤ 0 — degenerate allocation")

    passed = len(issues) == 0
    score = min(ratio / FAIRNESS_RATIO_FLOOR, 1.0) if ratio > 0 else 0.0
    return EvalResult(
        name="fairness_metrics",
        passed=passed,
        score=score,
        weight=1.0,
        details="; ".join(issues) if issues else f"Fair allocation (ratio={ratio:.3f}, NSW={nsw:.4f})",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7. Weather Risk Consistency Validator
# ═══════════════════════════════════════════════════════════════════════════

def validate_weather_consistency(output: AgentPipelineOutput) -> EvalResult:
    """
    Check: weather risk scores are consistent with raw weather data.
    - precip ≥ 15mm → heavy_rain flag should be present
    - wind ≥ 45km/h → high_wind flag should be present
    - temp ≤ 0°C → freezing flag should be present
    - risk_score_0_3 = count of active flags
    """
    issues = []
    checked = 0

    for wr in output.corridor_weather_risks:
        flags = wr.risk_flags or []
        expected_flags = []

        if wr.max_precip_mm_day is not None and wr.max_precip_mm_day >= WEATHER_THRESHOLDS["heavy_rain"]:
            expected_flags.append("heavy_rain")
        if wr.max_wind_gust_kmh is not None and wr.max_wind_gust_kmh >= WEATHER_THRESHOLDS["high_wind"]:
            expected_flags.append("high_wind")
        if wr.min_temp_c is not None and wr.min_temp_c <= WEATHER_THRESHOLDS["freezing"]:
            expected_flags.append("freezing")

        # Check flag consistency
        for expected in expected_flags:
            if expected not in flags:
                issues.append(f"{wr.corridor_id}: missing expected flag '{expected}'")

        # Check score consistency
        expected_score = len(expected_flags)
        if wr.route_risk_score_0_3 != expected_score and expected_flags:
            issues.append(
                f"{wr.corridor_id}: risk_score={wr.route_risk_score_0_3} but "
                f"expected {expected_score} based on flags {expected_flags}"
            )

        checked += 1

    if checked == 0:
        return EvalResult(
            name="weather_consistency",
            passed=False, score=0.0, weight=1.0,
            details="No weather data to validate",
        )

    passed = len(issues) == 0
    score = max(0.0, 1.0 - len(issues) * 0.2)
    return EvalResult(
        name="weather_consistency",
        passed=passed,
        score=score,
        weight=1.0,
        details="; ".join(issues) if issues else "Weather flags and scores are consistent",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 8. Plan Completeness Validator
# ═══════════════════════════════════════════════════════════════════════════

def validate_plan_completeness(output: AgentPipelineOutput) -> EvalResult:
    """
    Check: The dispatch plan mentions all corridors and covers required sections:
    - Per-corridor dispatch actions
    - Buffer/weather references
    - Resource/truck assignments
    - Contingency triggers
    - KPI impacts
    """
    plan = output.dispatch_plan.lower()
    if not plan.strip():
        return EvalResult(
            name="plan_completeness",
            passed=False, score=0.0, weight=1.5,
            details="Dispatch plan is empty",
        )

    sub_results = []

    # Check corridor coverage
    corridors = output.corridors or [wr.corridor_id for wr in output.corridor_weather_risks]
    missing_corridors = [c for c in corridors if c.lower() not in plan]
    corridor_score = max(0.0, 1.0 - len(missing_corridors) / max(len(corridors), 1))
    sub_results.append(EvalResult(
        name="corridor_coverage",
        passed=len(missing_corridors) == 0,
        score=corridor_score,
        details=f"Missing corridors: {missing_corridors}" if missing_corridors else "All corridors covered",
    ))

    # Check required sections
    required_sections = {
        "weather_reference": ["weather", "risk", "precip", "wind", "temperature", "forecast"],
        "buffer_mention": ["buffer", "%", "percent", "adjustment", "travel time"],
        "resource_assignment": ["truck", "driver", "resource", "capacity", "allocation"],
        "contingency": ["contingency", "escalat", "fallback", "backup", "if ", "monitor"],
        "kpi_impact": ["kpi", "sla", "utilization", "cost", "compliance", "on-time"],
    }

    for section_name, keywords in required_sections.items():
        found = any(kw in plan for kw in keywords)
        sub_results.append(EvalResult(
            name=section_name,
            passed=found,
            score=1.0 if found else 0.0,
            details=f"{'Found' if found else 'MISSING'} in dispatch plan",
        ))

    overall_score = sum(r.score for r in sub_results) / len(sub_results)
    overall_passed = all(r.passed for r in sub_results)
    return EvalResult(
        name="plan_completeness",
        passed=overall_passed,
        score=overall_score,
        weight=1.5,
        sub_results=sub_results,
        details=f"Completeness score: {overall_score:.1%}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 9. Weather Faithfulness Validator (no hallucination)
# ═══════════════════════════════════════════════════════════════════════════

def validate_weather_faithfulness(output: AgentPipelineOutput) -> EvalResult:
    """
    Check: The dispatch plan does NOT hallucinate weather data.
    Specifically, it should NOT mention snow/snowfall, visibility,
    or hourly thresholds unless the raw weather data contains them.
    """
    plan = output.dispatch_plan.lower()
    report = output.report_html.lower()
    combined = plan + " " + report

    hallucination_terms = [
        "snowfall", "snow accumulation", "inches of snow",
        "visibility", "fog", "ice storm",
    ]

    found_hallucinations = [term for term in hallucination_terms if term in combined]

    # Check if any weather data actually contains these
    has_snow_data = False
    for wr in output.corridor_weather_risks:
        if wr.per_waypoint:
            for wp in wr.per_waypoint:
                if any(k for k in wp.keys() if "snow" in k.lower()):
                    has_snow_data = True

    issues = []
    if found_hallucinations and not has_snow_data:
        issues = [f"Hallucinated weather term: '{term}'" for term in found_hallucinations]

    passed = len(issues) == 0
    score = 1.0 if passed else max(0.0, 1.0 - len(issues) * 0.3)
    return EvalResult(
        name="weather_faithfulness",
        passed=passed,
        score=score,
        weight=1.5,
        details="; ".join(issues) if issues else "No weather hallucinations detected",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 10. Report Quality Validator (structural)
# ═══════════════════════════════════════════════════════════════════════════

def validate_report_structure(output: AgentPipelineOutput) -> EvalResult:
    """
    Check: HTML report contains required sections per playbook §12:
    - Weather risk summary
    - Applied travel buffers
    - Valid vs excluded shipment counts
    - SLA risk flags
    Plus multi-corridor additions:
    - Corridor comparison table
    - Resource allocation table
    - Fairness metrics
    """
    html = output.report_html.lower()
    if not html.strip():
        return EvalResult(
            name="report_structure",
            passed=False, score=0.0, weight=1.0,
            details="Report HTML is empty",
        )

    required_elements = {
        "weather_summary": ["weather", "risk"],
        "travel_buffer": ["buffer", "travel"],
        "shipment_counts": ["shipment", "volume", "count"],
        "sla_flags": ["sla", "tier"],
        "html_table": ["<table", "<tr", "<td"],
        "corridor_data": [],  # filled dynamically
    }

    # Dynamic corridor check
    corridors = output.corridors or [wr.corridor_id for wr in output.corridor_weather_risks]
    required_elements["corridor_data"] = [c.lower() for c in corridors]

    sub_results = []
    for name, keywords in required_elements.items():
        found = any(kw in html for kw in keywords) if keywords else True
        sub_results.append(EvalResult(
            name=name, passed=found, score=1.0 if found else 0.0,
            details=f"{'Found' if found else 'MISSING'} in report",
        ))

    overall_score = sum(r.score for r in sub_results) / len(sub_results)
    return EvalResult(
        name="report_structure",
        passed=all(r.passed for r in sub_results),
        score=overall_score,
        weight=1.0,
        sub_results=sub_results,
        details=f"Report structure score: {overall_score:.1%}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 11. Validation Loop Coherence
# ═══════════════════════════════════════════════════════════════════════════

def validate_loop_coherence(output: AgentPipelineOutput) -> EvalResult:
    """
    Check: If the validation agent found issues, the dispatch plan should
    have been revised (or the issues should be acknowledged).
    """
    if not output.validation_result:
        return EvalResult(
            name="validation_loop",
            passed=True, score=0.5, weight=1.0,
            details="No validation result — cannot check loop coherence",
        )

    val = output.validation_result
    if val.is_valid:
        return EvalResult(
            name="validation_loop",
            passed=True, score=1.0, weight=1.0,
            details="Validation passed — plan is valid",
        )

    # Plan is invalid — check if issues are at least acknowledged
    plan_lower = output.dispatch_plan.lower()
    acknowledged = 0
    for issue in val.issues:
        # Check if key terms from the issue appear in the plan
        terms = [w for w in issue.lower().split() if len(w) > 4]
        if any(t in plan_lower for t in terms[:3]):
            acknowledged += 1

    ack_ratio = acknowledged / max(len(val.issues), 1)
    return EvalResult(
        name="validation_loop",
        passed=ack_ratio > 0.5,
        score=ack_ratio,
        weight=1.0,
        details=f"Plan acknowledged {acknowledged}/{len(val.issues)} validation issues",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Validator Registry
# ═══════════════════════════════════════════════════════════════════════════

ALL_VALIDATORS: List[Callable[[AgentPipelineOutput], EvalResult]] = [
    validate_sla_compliance,
    validate_cold_chain,
    validate_buffer_policy,
    validate_utilization,
    validate_cost_guardrails,
    validate_fairness,
    validate_weather_consistency,
    validate_plan_completeness,
    validate_weather_faithfulness,
    validate_report_structure,
    validate_loop_coherence,
]


def run_all_validators(output: AgentPipelineOutput) -> List[EvalResult]:
    """Run all deterministic validators and return results."""
    return [v(output) for v in ALL_VALIDATORS]
