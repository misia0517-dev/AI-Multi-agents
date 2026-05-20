"""
Mia-Specific Validators
========================
Additional deterministic validators for Mia's unique features:
  - Penalty model (shortfall detection, penalty scoring)
  - Escalation check (risk_score_48h ≥ 3 → escalation_required)
  - Day-level resource feasibility (Day0, Day1 separately)
  - Planner retry coherence (violations addressed after retry)
  - Cold-chain over-assignment guard

These run IN ADDITION to the standard 11 validators.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .validators import EvalResult


# ═══════════════════════════════════════════════════════════════════════════
# 12. Penalty Model Consistency
# ═══════════════════════════════════════════════════════════════════════════

def validate_penalty_model(state: Dict[str, Any]) -> EvalResult:
    """
    Check: If penalty_score > 0, allocation_feasible should be False.
    If penalty_score == 0, allocation_feasible should be True.
    Verify penalty math: corridor_penalty sums = day_total_penalty.
    """
    alloc = state.get("resource_allocation", {})
    if not alloc:
        return EvalResult(
            name="penalty_model",
            passed=False, score=0.0, weight=1.5,
            details="No resource allocation data",
        )

    summary = alloc.get("summary_48h", {})
    total_penalty = summary.get("total_penalty_score", 0)
    feasible = summary.get("allocation_feasible", True)

    issues = []

    # Consistency check: penalty vs feasibility
    if total_penalty > 0 and feasible:
        issues.append(
            f"Penalty={total_penalty} but allocation_feasible=True — inconsistent"
        )
    if total_penalty == 0 and not feasible:
        issues.append(
            f"Penalty=0 but allocation_feasible=False — inconsistent"
        )

    # Verify penalty sums
    computed_total = 0
    for day in ["Day0", "Day1"]:
        day_data = alloc.get(day, {})
        day_penalty = day_data.get("day_total_penalty", 0)
        corridor_sum = sum(
            c.get("corridor_penalty", 0)
            for c in day_data.get("corridors", {}).values()
            if isinstance(c, dict)
        )
        if corridor_sum != day_penalty:
            issues.append(
                f"{day}: corridor penalties sum={corridor_sum} != day_total_penalty={day_penalty}"
            )
        computed_total += corridor_sum

    if computed_total != total_penalty:
        issues.append(
            f"Computed total penalty={computed_total} != summary total={total_penalty}"
        )

    passed = len(issues) == 0
    score = 1.0 if passed else max(0.0, 1.0 - len(issues) * 0.25)
    return EvalResult(
        name="penalty_model",
        passed=passed,
        score=score,
        weight=1.5,
        details="; ".join(issues) if issues else f"Penalty model consistent (total={total_penalty})",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 13. Shortfall Detection
# ═══════════════════════════════════════════════════════════════════════════

def validate_shortfall_detection(state: Dict[str, Any]) -> EvalResult:
    """
    Check: If any corridor has shortfall > 0, it should be flagged
    (can_dispatch_all=False) and the dispatch plan should mention it.
    """
    alloc = state.get("resource_allocation", {})
    plan = state.get("dispatch_plan", "").lower()

    if not alloc:
        return EvalResult(
            name="shortfall_detection",
            passed=True, score=0.5, weight=1.5,
            details="No allocation data to check shortfalls",
        )

    shortfall_corridors = []
    all_ok = True

    for day in ["Day0", "Day1"]:
        corridors = alloc.get(day, {}).get("corridors", {})
        for cid, stats in corridors.items():
            if not isinstance(stats, dict):
                continue

            has_shortfall = (
                stats.get("shortfall_temp_trucks", 0) > 0
                or stats.get("shortfall_std_trucks", 0) > 0
                or stats.get("shortfall_drivers", 0) > 0
            )

            if has_shortfall:
                all_ok = False
                # Check can_dispatch_all is correctly False
                if stats.get("can_dispatch_all", True):
                    shortfall_corridors.append(
                        f"{day}/{cid}: has shortfall but can_dispatch_all=True"
                    )

                # Check plan mentions it
                cid_lower = cid.lower().replace("_", "")
                plan_clean = plan.replace("_", "").replace(" ", "")
                if cid_lower not in plan_clean:
                    shortfall_corridors.append(
                        f"{day}/{cid}: shortfall exists but corridor not mentioned in plan"
                    )

    if all_ok:
        return EvalResult(
            name="shortfall_detection",
            passed=True, score=1.0, weight=1.5,
            details="No shortfalls detected — all corridors fully served",
        )

    passed = len(shortfall_corridors) == 0
    score = 1.0 if passed else max(0.0, 1.0 - len(shortfall_corridors) * 0.25)
    return EvalResult(
        name="shortfall_detection",
        passed=passed,
        score=score,
        weight=1.5,
        details="; ".join(shortfall_corridors) if shortfall_corridors else "Shortfalls correctly flagged and addressed in plan",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 14. Escalation Check
# ═══════════════════════════════════════════════════════════════════════════

def validate_escalation(state: Dict[str, Any]) -> EvalResult:
    """
    Check: If any corridor has risk_score_48h >= 3, escalation_required
    should be True AND the dispatch plan should explicitly address it.
    """
    weather = state.get("corridor_weather_risk", {})
    plan = state.get("dispatch_plan", "").lower()

    if not weather:
        return EvalResult(
            name="escalation_check",
            passed=True, score=0.5, weight=1.0,
            details="No weather data to check escalation",
        )

    issues = []
    for cid, risk in weather.items():
        score = risk.get("risk_score_48h", 0)
        escalation = risk.get("escalation_required", False)

        if score >= 3 and not escalation:
            issues.append(f"{cid}: risk={score} but escalation_required=False")

        if score >= 3:
            # Check plan mentions escalation for this corridor
            cid_token = cid.lower().replace("_", "")
            plan_token = plan.replace("_", "")
            has_escalation_mention = (
                "escalat" in plan and cid_token in plan_token
            )
            if not has_escalation_mention:
                issues.append(f"{cid}: risk={score} requires escalation but plan doesn't mention it")

    passed = len(issues) == 0
    score_val = 1.0 if passed else max(0.0, 1.0 - len(issues) * 0.3)
    return EvalResult(
        name="escalation_check",
        passed=passed,
        score=score_val,
        weight=1.0,
        details="; ".join(issues) if issues else "Escalation flags correctly set",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 15. Cold-Chain Over-Assignment Guard
# ═══════════════════════════════════════════════════════════════════════════

def validate_cold_chain_overassign(state: Dict[str, Any]) -> EvalResult:
    """
    Check: Total temp trucks assigned across corridors on the same day
    must not exceed available temp trucks for that day.
    """
    alloc = state.get("resource_allocation", {})
    if not alloc:
        return EvalResult(
            name="cold_chain_overassign",
            passed=True, score=0.5, weight=1.5,
            details="No allocation data",
        )

    issues = []
    for day in ["Day0", "Day1"]:
        day_data = alloc.get(day, {})
        available_temp = day_data.get("available", {}).get("truck_temp_controlled", 999)
        total_assigned = sum(
            c.get("allocated_temp_trucks", 0)
            for c in day_data.get("corridors", {}).values()
            if isinstance(c, dict)
        )
        if total_assigned > available_temp:
            issues.append(
                f"{day}: {total_assigned} temp trucks assigned but only {available_temp} available"
            )

    passed = len(issues) == 0
    return EvalResult(
        name="cold_chain_overassign",
        passed=passed,
        score=1.0 if passed else 0.0,
        weight=1.5,
        details="; ".join(issues) if issues else "No temp truck over-assignment",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 16. Planner Retry Coherence
# ═══════════════════════════════════════════════════════════════════════════

def validate_retry_coherence(state: Dict[str, Any]) -> EvalResult:
    """
    Check: If planner_retry_count > 0, the final dispatch plan should
    address the violations that triggered the retry.
    """
    retry_count = state.get("planner_retry_count", 0)
    violations = state.get("validation_violations", [])

    if retry_count == 0:
        return EvalResult(
            name="retry_coherence",
            passed=True, score=1.0, weight=1.0,
            details="No retries needed — plan passed on first attempt",
        )

    plan = state.get("dispatch_plan", "").lower()

    if not violations:
        return EvalResult(
            name="retry_coherence",
            passed=True, score=1.0, weight=1.0,
            details=f"Retried {retry_count} time(s), all violations resolved",
        )

    # Check if unresolved violations are at least mentioned in the plan
    addressed = 0
    for v in violations:
        terms = [w for w in v.lower().split() if len(w) > 4][:4]
        if any(t in plan for t in terms):
            addressed += 1

    ratio = addressed / max(len(violations), 1)
    return EvalResult(
        name="retry_coherence",
        passed=ratio >= 0.5,
        score=ratio,
        weight=1.0,
        details=f"After {retry_count} retry(s), plan addresses {addressed}/{len(violations)} unresolved violations",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 17. Day-Level Feasibility
# ═══════════════════════════════════════════════════════════════════════════

def validate_day_feasibility(state: Dict[str, Any]) -> EvalResult:
    """
    Check: Remaining resource pool after allocation should be >= 0
    for all resource types on both days.
    """
    alloc = state.get("resource_allocation", {})
    if not alloc:
        return EvalResult(
            name="day_feasibility",
            passed=True, score=0.5, weight=1.0,
            details="No allocation data",
        )

    issues = []
    for day in ["Day0", "Day1"]:
        remaining = alloc.get(day, {}).get("remaining_pool", {})
        for rtype, count in remaining.items():
            if count < 0:
                issues.append(f"{day}: {rtype} over-allocated (remaining={count})")

    passed = len(issues) == 0
    return EvalResult(
        name="day_feasibility",
        passed=passed,
        score=1.0 if passed else 0.0,
        weight=1.0,
        details="; ".join(issues) if issues else "Resources not over-allocated on any day",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

MIA_VALIDATORS = [
    validate_penalty_model,
    validate_shortfall_detection,
    validate_escalation,
    validate_cold_chain_overassign,
    validate_retry_coherence,
    validate_day_feasibility,
]


def run_mia_validators(state: Dict[str, Any]) -> List[EvalResult]:
    """Run all Mia-specific validators on the raw state dict."""
    return [v(state) for v in MIA_VALIDATORS]
