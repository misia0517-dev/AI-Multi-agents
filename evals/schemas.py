"""
Agent-agnostic output schemas.

Any agent that produces dispatch plans must conform to these schemas
so the eval framework can validate its outputs. The schemas define
the *contract* — not the implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Agent Output Contract — what the eval framework expects as input
# ---------------------------------------------------------------------------

@dataclass
class CorridorWeatherRisk:
    """Weather risk for a single corridor (from the weather stage)."""
    corridor_id: str
    route_risk_score_0_3: int  # 0–3
    max_precip_mm_day: Optional[float] = None
    max_wind_gust_kmh: Optional[float] = None
    min_temp_c: Optional[float] = None
    risk_flags: Optional[List[str]] = None
    adjusted_travel_hrs: Optional[float] = None
    base_travel_hrs: Optional[float] = None
    total_distance_km: Optional[float] = None
    worst_waypoint: Optional[Dict[str, Any]] = None
    per_waypoint: Optional[List[Dict[str, Any]]] = None


@dataclass
class CorridorAllocation:
    """Resource allocation for a single corridor."""
    corridor_id: str
    truck_standard: int = 0
    truck_temp_controlled: int = 0
    driver: int = 0
    weather_risk_score: int = 0
    buffer_pct: float = 0.0
    demand_volume: int = 0
    buffered_demand: float = 0.0
    total_capacity: float = 0.0
    cold_chain_demand: int = 0
    temp_truck_capacity: float = 0.0
    utilization_pct: float = 0.0
    cold_chain_coverage_pct: float = 0.0
    estimated_daily_cost: float = 0.0


@dataclass
class ResourceAllocationOutput:
    """Complete resource allocation output."""
    allocations: Dict[str, CorridorAllocation] = field(default_factory=dict)
    total_daily_cost: float = 0.0
    nsw_score: float = 0.0
    max_min_fairness_ratio: float = 0.0
    per_corridor_utility: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationOutput:
    """Validation agent output."""
    is_valid: bool = True
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class AgentPipelineOutput:
    """
    The complete output contract for any dispatch planning agent pipeline.
    This is what the eval framework evaluates.

    An agent implementation must produce this (or a dict that can be parsed
    into this) for the eval framework to score it.
    """
    # Core outputs
    dispatch_plan: str = ""
    report_html: str = ""
    validation_result: Optional[ValidationOutput] = None

    # Intermediate outputs (needed for constraint checking)
    corridor_weather_risks: List[CorridorWeatherRisk] = field(default_factory=list)
    resource_allocation: Optional[ResourceAllocationOutput] = None

    # Context (for faithfulness checks)
    business_context: str = ""
    ops_insights: str = ""
    csv_kpis: Dict[str, Any] = field(default_factory=dict)
    anomalies_md: str = ""

    # Metadata
    corridors: List[str] = field(default_factory=list)
    agent_provider: str = "unknown"  # e.g. "openai", "anthropic", "google"
    model_name: str = "unknown"


def parse_agent_output(raw: Dict[str, Any]) -> AgentPipelineOutput:
    """
    Parse a raw dict (from any agent) into the standard AgentPipelineOutput.
    Tolerant of missing fields — fills defaults.
    """
    # Parse weather risks
    weather_risks = []
    for wr in raw.get("corridor_weather_risks", []):
        if isinstance(wr, dict):
            weather_risks.append(CorridorWeatherRisk(
                corridor_id=wr.get("corridor_id", "unknown"),
                route_risk_score_0_3=wr.get("route_risk_score_0_3",
                                             wr.get("risk_score_0_3", 0)),
                max_precip_mm_day=wr.get("max_precip_mm_day"),
                max_wind_gust_kmh=wr.get("max_wind_gust_kmh"),
                min_temp_c=wr.get("min_temp_c"),
                risk_flags=wr.get("risk_flags"),
                adjusted_travel_hrs=wr.get("adjusted_travel_hrs"),
                base_travel_hrs=wr.get("base_travel_hrs"),
                total_distance_km=wr.get("total_distance_km"),
                worst_waypoint=wr.get("worst_waypoint"),
                per_waypoint=wr.get("per_waypoint"),
            ))

    # Parse resource allocation
    resource_alloc = None
    ra_raw = raw.get("resource_allocation", {})
    if ra_raw:
        allocs = {}
        for cid, a in ra_raw.get("allocations", {}).items():
            if isinstance(a, dict):
                allocs[cid] = CorridorAllocation(
                    corridor_id=cid,
                    truck_standard=a.get("truck_standard", 0),
                    truck_temp_controlled=a.get("truck_temp_controlled", 0),
                    driver=a.get("driver", 0),
                    weather_risk_score=a.get("weather_risk_score", 0),
                    buffer_pct=a.get("buffer_pct", 0.0),
                    demand_volume=a.get("demand_volume", 0),
                    buffered_demand=a.get("buffered_demand", 0.0),
                    total_capacity=a.get("total_capacity", 0.0),
                    cold_chain_demand=a.get("cold_chain_demand", 0),
                    temp_truck_capacity=a.get("temp_truck_capacity", 0.0),
                    utilization_pct=a.get("utilization_pct", 0.0),
                    cold_chain_coverage_pct=a.get("cold_chain_coverage_pct", 0.0),
                    estimated_daily_cost=a.get("estimated_daily_cost", 0.0),
                )

        summary = ra_raw.get("summary", {})
        resource_alloc = ResourceAllocationOutput(
            allocations=allocs,
            total_daily_cost=summary.get("total_daily_cost", 0.0),
            nsw_score=summary.get("nsw_score", 0.0),
            max_min_fairness_ratio=summary.get("max_min_fairness_ratio", 0.0),
            per_corridor_utility=summary.get("per_corridor_utility", {}),
        )

    # Parse validation
    val_raw = raw.get("validation_result", {})
    validation = None
    if val_raw:
        validation = ValidationOutput(
            is_valid=val_raw.get("is_valid", True),
            issues=val_raw.get("issues", []),
            suggestions=val_raw.get("suggestions", []),
            notes=val_raw.get("notes", ""),
        )

    return AgentPipelineOutput(
        dispatch_plan=raw.get("dispatch_plan", ""),
        report_html=raw.get("report_html", ""),
        validation_result=validation,
        corridor_weather_risks=weather_risks,
        resource_allocation=resource_alloc,
        business_context=raw.get("business_context", ""),
        ops_insights=raw.get("ops_insights", ""),
        csv_kpis=raw.get("csv_kpis", {}),
        anomalies_md=raw.get("anomalies_md", ""),
        corridors=raw.get("corridors", []),
        agent_provider=raw.get("agent_provider", "unknown"),
        model_name=raw.get("model_name", "unknown"),
    )
