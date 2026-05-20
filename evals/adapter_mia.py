"""
Adapter for Mia's SeeWeeS implementation
==========================================
Mia's code uses a different state shape, corridor naming, weather format,
and resource allocation model (penalty-based greedy vs our NSW fairness).

This adapter translates her final LangGraph state into our eval schema
so the same validators + judges can score her output.

Usage in Mia's main.py:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "SeeWeeS"))

    from evals.adapter_mia import quick_eval_mia
    report = quick_eval_mia(final_state)
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from .schemas import (
    AgentPipelineOutput,
    CorridorWeatherRisk,
    CorridorAllocation,
    ResourceAllocationOutput,
    ValidationOutput,
)
from .validators import EvalResult, run_all_validators
from .validators_mia import run_mia_validators
from .llm_judges import JudgeFn, run_all_judges
from .runner import EvalReport, evaluate_output


# ---------------------------------------------------------------------------
# Constants from Mia's code
# ---------------------------------------------------------------------------
BUFFER_MAP = {0: 0.0, 1: 0.10, 2: 0.25, 3: 0.40}
EFFECTIVE_CAPACITY = 9.0  # 10 * 0.9 (TRUCK_CAPACITY / PACKING_BUFFER)
BASE_TRAVEL_HOURS = {
    "C1_I95_NJ_BOS": 4.6,
    "C2_NJ_PHL": 1.8,
}


# ---------------------------------------------------------------------------
# Weather translation
# ---------------------------------------------------------------------------

def _translate_weather(mia_state: Dict[str, Any]) -> List[CorridorWeatherRisk]:
    """
    Mia stores weather as:
        corridor_weather_risk = {
            "C1_I95_NJ_BOS": {
                "risk_score_48h": 1,
                "risk_flags_48h": {"heavy_rain_risk": True, ...},
                "travel_buffer_pct": 10,
                "escalation_required": False,
                "waypoints": [...],
                "day_risk": [0, 1],
            },
            ...
        }
    We translate to our list of CorridorWeatherRisk.
    """
    weather_dict = mia_state.get("corridor_weather_risk", {})
    results = []

    for corridor_id, risk in weather_dict.items():
        # Extract max weather values from waypoint data
        max_precip = None
        max_wind = None
        min_temp = None

        waypoints = risk.get("waypoints", [])
        for wp in waypoints:
            precip_days = wp.get("precip_by_day", [])
            if precip_days:
                p = max(precip_days)
                if max_precip is None or p > max_precip:
                    max_precip = p

            gust_days = wp.get("gusts_by_day", [])
            if gust_days:
                g = max(gust_days)
                if max_wind is None or g > max_wind:
                    max_wind = g

            tmin_days = wp.get("tmin_by_day", [])
            if tmin_days:
                t = min(tmin_days)
                if min_temp is None or t < min_temp:
                    min_temp = t

        # Translate flags
        flags_dict = risk.get("risk_flags_48h", {})
        flags_list = []
        if flags_dict.get("heavy_rain_risk"):
            flags_list.append("heavy_rain")
        if flags_dict.get("high_wind_risk"):
            flags_list.append("high_wind")
        if flags_dict.get("freezing_risk"):
            flags_list.append("freezing")

        risk_score = risk.get("risk_score_48h", 0)
        buffer_pct = risk.get("travel_buffer_pct", 0) / 100.0  # 10 → 0.10
        base_travel_hrs = BASE_TRAVEL_HOURS.get(corridor_id)
        adjusted_travel_hrs = (
            round(base_travel_hrs * (1 + buffer_pct), 2)
            if base_travel_hrs is not None
            else None
        )

        results.append(CorridorWeatherRisk(
            corridor_id=corridor_id,
            route_risk_score_0_3=risk_score,
            max_precip_mm_day=max_precip,
            max_wind_gust_kmh=max_wind,
            min_temp_c=min_temp,
            risk_flags=flags_list,
            adjusted_travel_hrs=adjusted_travel_hrs,
            base_travel_hrs=base_travel_hrs,
            total_distance_km=None,
            worst_waypoint=None,
            per_waypoint=[{"waypoint": wp.get("waypoint_id", ""), "city": wp.get("city", "")}
                          for wp in waypoints],
        ))

    return results


# ---------------------------------------------------------------------------
# Resource allocation translation
# ---------------------------------------------------------------------------

def _translate_allocation(mia_state: Dict[str, Any]) -> Optional[ResourceAllocationOutput]:
    """
    Mia stores allocation as:
        resource_allocation = {
            "Day0": {
                "available": {"driver": 7, "truck_standard": 4, ...},
                "corridors": {
                    "C1_I95_NJ_BOS": {
                        "allocated_temp_trucks": 3,
                        "allocated_std_trucks": 2,
                        "allocated_drivers": 5,
                        "shortfall_temp_trucks": 0,
                        "total_units": 50,
                        "can_dispatch_all": True,
                        "corridor_penalty": 0,
                        "weather_risk_score": 1,
                        "travel_buffer_pct": 10,
                        ...
                    },
                },
            },
            "Day1": { ... },
            "summary_48h": {
                "total_penalty_score": 0,
                "allocation_feasible": True,
                ...
            },
        }

    We aggregate across Day0+Day1 into our per-corridor CorridorAllocation format.
    """
    alloc_raw = mia_state.get("resource_allocation", {})
    if not alloc_raw:
        return None

    # Aggregate per corridor across both days
    corridor_totals: Dict[str, Dict[str, Any]] = {}

    for day in ["Day0", "Day1"]:
        day_data = alloc_raw.get(day, {})
        corridors = day_data.get("corridors", {})

        for cid, stats in corridors.items():
            if not isinstance(stats, dict):
                continue

            if cid not in corridor_totals:
                corridor_totals[cid] = {
                    "truck_standard": 0,
                    "truck_temp_controlled": 0,
                    "driver": 0,
                    "demand_volume": 0,
                    "cold_chain_demand": 0,
                    "cost": 0.0,
                    "risk": stats.get("weather_risk_score", 0),
                    "shortfall_temp": 0,
                    "shortfall_std": 0,
                    "shortfall_drivers": 0,
                    "can_dispatch_all": True,
                }

            t = corridor_totals[cid]
            t["truck_standard"] += stats.get("allocated_std_trucks", 0)
            t["truck_temp_controlled"] += stats.get("allocated_temp_trucks", 0)
            t["driver"] += stats.get("allocated_drivers", 0)
            t["demand_volume"] += stats.get("total_units", 0)
            t["cold_chain_demand"] += stats.get("temp_controlled_units",
                                                  stats.get("total_units", 0) // 2)  # estimate if missing
            t["cost"] += stats.get("corridor_penalty", 0)  # penalty as cost proxy
            t["shortfall_temp"] += stats.get("shortfall_temp_trucks", 0)
            t["shortfall_std"] += stats.get("shortfall_std_trucks", 0)
            t["shortfall_drivers"] += stats.get("shortfall_drivers", 0)
            if not stats.get("can_dispatch_all", True):
                t["can_dispatch_all"] = False

    # Build CorridorAllocation objects
    allocations: Dict[str, CorridorAllocation] = {}
    for cid, t in corridor_totals.items():
        risk = t["risk"]
        buffer = BUFFER_MAP.get(risk, 0.0)
        total_cap = (t["truck_standard"] + t["truck_temp_controlled"]) * EFFECTIVE_CAPACITY
        temp_cap = t["truck_temp_controlled"] * EFFECTIVE_CAPACITY
        buffered_demand = t["demand_volume"] * (1 + buffer)
        buffered_cold = t["cold_chain_demand"] * (1 + buffer)

        allocations[cid] = CorridorAllocation(
            corridor_id=cid,
            truck_standard=t["truck_standard"],
            truck_temp_controlled=t["truck_temp_controlled"],
            driver=t["driver"],
            weather_risk_score=risk,
            buffer_pct=buffer,
            demand_volume=t["demand_volume"],
            buffered_demand=round(buffered_demand, 1),
            total_capacity=round(total_cap, 1),
            cold_chain_demand=t["cold_chain_demand"],
            temp_truck_capacity=round(temp_cap, 1),
            utilization_pct=round(buffered_demand / max(total_cap, 1) * 100, 1),
            cold_chain_coverage_pct=round(
                min(temp_cap / max(buffered_cold, 0.001), 1.0) * 100, 1
            ),
            estimated_daily_cost=t["cost"],
        )

    # Summary
    summary_raw = alloc_raw.get("summary_48h", {})
    total_penalty = summary_raw.get("total_penalty_score", 0)

    # Compute fairness as ratio of min/max allocated capacity across corridors
    caps = [
        (a.truck_standard + a.truck_temp_controlled) * EFFECTIVE_CAPACITY
        for a in allocations.values()
    ]
    fairness = round(min(caps) / max(max(caps), 0.001), 3) if caps else 0.0

    return ResourceAllocationOutput(
        allocations=allocations,
        total_daily_cost=total_penalty,  # Mia uses penalty instead of cost
        nsw_score=1.0 if total_penalty == 0 else max(0.0, 1.0 - total_penalty / 1000),
        max_min_fairness_ratio=fairness,
        per_corridor_utility={},
    )


# ---------------------------------------------------------------------------
# Validation translation
# ---------------------------------------------------------------------------

def _translate_validation(mia_state: Dict[str, Any]) -> Optional[ValidationOutput]:
    """Mia stores validation as a list of violation strings."""
    violations = mia_state.get("validation_violations", [])

    return ValidationOutput(
        is_valid=len(violations) == 0,
        issues=violations,
        suggestions=[],
        notes="",
    )


# ---------------------------------------------------------------------------
# Main adapter: Mia state → AgentPipelineOutput
# ---------------------------------------------------------------------------

def from_mia_state(
    state: Dict[str, Any],
    agent_provider: str = "anthropic",
    model_name: str = "claude-sonnet-4-6",
) -> AgentPipelineOutput:
    """
    Convert Mia's LangGraph AppState into our eval schema.
    """
    weather = _translate_weather(state)
    allocation = _translate_allocation(state)
    validation = _translate_validation(state)

    corridors = list(state.get("corridor_weather_risk", {}).keys())

    return AgentPipelineOutput(
        dispatch_plan=state.get("dispatch_plan", ""),
        report_html=state.get("report_html", ""),
        validation_result=validation,
        corridor_weather_risks=weather,
        resource_allocation=allocation,
        business_context=state.get("business_context", ""),
        ops_insights=state.get("ops_insights", ""),
        csv_kpis=state.get("csv_kpis", {}),
        anomalies_md=state.get("anomalies_md", ""),
        corridors=corridors,
        agent_provider=agent_provider,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Convenience one-liners (same API as adapter.py)
# ---------------------------------------------------------------------------

def quick_eval_mia(
    state: Dict[str, Any],
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    judge_fn: Optional[JudgeFn] = None,
    threshold: float = 0.70,
    print_report: bool = True,
) -> EvalReport:
    """
    One-line eval for Mia's agent output.

    Usage in Mia's main.py:
        from evals.adapter_mia import quick_eval_mia
        report = quick_eval_mia(final_state)
    """
    output = from_mia_state(state, provider, model)

    # Run standard validators
    report = EvalReport(
        scenario_name="mia_live_run",
        agent_provider=output.agent_provider,
        model_name=output.model_name,
        pass_threshold=threshold,
    )

    import time
    report.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Standard deterministic validators
    report.deterministic_results = run_all_validators(output)

    # Mia-specific validators (penalty model, shortfalls, escalation)
    mia_results = run_mia_validators(state)
    report.deterministic_results.extend(mia_results)

    # LLM judges
    if judge_fn:
        report.judge_results = run_all_judges(output, judge_fn)

    report.compute_scores()

    if print_report:
        print(report.summary_str())

    return report


def validators_only_mia(state: Dict[str, Any], print_results: bool = True) -> list:
    """Run all validators (standard + Mia-specific) without LLM judges."""
    output = from_mia_state(state)
    results = run_all_validators(output)
    results.extend(run_mia_validators(state))

    if print_results:
        for r in results:
            icon = "PASS" if r.passed else "FAIL"
            print(f"  [{icon}] {r.name}: {r.score:.1%} — {r.details[:80]}")

    return results
