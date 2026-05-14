"""
Test Fixtures & Golden Scenarios
=================================
Pre-built scenarios with known inputs and expected eval outcomes.
Used for testing the eval framework itself and as regression baselines.

Scenarios:
  1. happy_path          — Everything works, all constraints met
  2. sla_breach          — NE-I95 exceeds 6hr Tier 1 SLA
  3. cold_chain_gap      — Insufficient temp-controlled trucks
  4. resource_shortage   — Driver shortage on Day 14
  5. weather_escalation  — Risk score 3 with hallucinated snow
  6. unfair_allocation   — One corridor starved, fairness ratio < 0.7
  7. empty_plan          — Agent produces empty/minimal output
"""
from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helper: build a corridor weather risk dict
# ---------------------------------------------------------------------------

def _weather(
    corridor_id: str,
    risk_score: int = 0,
    precip: float = 5.0,
    wind: float = 20.0,
    temp: float = 5.0,
    base_hrs: float = 4.0,
    distance_km: float = 375.0,
    flags: List[str] | None = None,
) -> Dict[str, Any]:
    buffer_policy = {0: 0.0, 1: 0.10, 2: 0.25, 3: 0.40}
    adjusted = base_hrs * (1 + 0.10 * risk_score)
    return {
        "corridor_id": corridor_id,
        "route_risk_score_0_3": risk_score,
        "risk_score_0_3": risk_score,
        "max_precip_mm_day": precip,
        "max_wind_gust_kmh": wind,
        "min_temp_c": temp,
        "risk_flags": flags or [],
        "adjusted_travel_hrs": round(adjusted, 2),
        "base_travel_hrs": base_hrs,
        "total_distance_km": distance_km,
        "worst_waypoint": {"waypoint": "W3", "city": "New Haven"},
        "per_waypoint": [],
    }


def _allocation(
    corridor_id: str,
    std: int = 3,
    temp: int = 3,
    drivers: int = 6,
    risk: int = 0,
    demand_vol: int = 40,
    cold_vol: int = 20,
    cost: float = 2200.0,
) -> Dict[str, Any]:
    buffer_policy = {0: 0.0, 1: 0.10, 2: 0.25, 3: 0.40}
    buffer = buffer_policy.get(risk, 0.0)
    eff_cap = 9.0  # 10 * 0.9
    total_cap = (std + temp) * eff_cap
    buffered = demand_vol * (1 + buffer)
    temp_cap = temp * eff_cap
    buffered_cold = cold_vol * (1 + buffer)
    return {
        "truck_standard": std,
        "truck_temp_controlled": temp,
        "driver": drivers,
        "weather_risk_score": risk,
        "buffer_pct": buffer,
        "demand_volume": demand_vol,
        "buffered_demand": round(buffered, 1),
        "total_capacity": round(total_cap, 1),
        "cold_chain_demand": cold_vol,
        "temp_truck_capacity": round(temp_cap, 1),
        "utilization_pct": round(buffered / max(total_cap, 1) * 100, 1),
        "cold_chain_coverage_pct": round(
            min(temp_cap / max(buffered_cold, 0.001), 1.0) * 100, 1
        ),
        "estimated_daily_cost": cost,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1: HAPPY PATH
# ═══════════════════════════════════════════════════════════════════════════

HAPPY_PATH: Dict[str, Any] = {
    "scenario_name": "happy_path",
    "description": "All constraints met, low weather risk, adequate resources",
    "corridors": ["NE-I95", "MA-I95"],
    "corridor_weather_risks": [
        _weather("NE-I95", risk_score=1, precip=16.0, wind=30.0, temp=3.0,
                 base_hrs=4.6, flags=["heavy_rain"]),
        _weather("MA-I95", risk_score=0, precip=5.0, wind=20.0, temp=5.0,
                 base_hrs=2.0, distance_km=145.0),
    ],
    "resource_allocation": {
        "allocations": {
            "NE-I95": _allocation("NE-I95", std=3, temp=3, drivers=6,
                                   risk=1, demand_vol=42, cold_vol=22, cost=2450.0),
            "MA-I95": _allocation("MA-I95", std=2, temp=3, drivers=5,
                                   risk=0, demand_vol=35, cold_vol=18, cost=1950.0),
        },
        "summary": {
            "total_daily_cost": 4400.0,
            "nsw_score": 0.82,
            "max_min_fairness_ratio": 0.85,
            "per_corridor_utility": {"NE-I95": 0.88, "MA-I95": 0.80},
            "available_pool": {"driver": 10, "truck_standard": 5, "truck_temp_controlled": 5},
        },
    },
    "dispatch_plan": """
## Multi-Corridor Dispatch Plan (2026-03-13 to 2026-03-14)

### NE-I95 Corridor
- Risk Score: 1 (heavy_rain flag) → 10% travel buffer applied
- Adjusted travel time: 5.06 hours (within Tier 1 SLA of 6 hours)
- Assign 3 standard trucks + 3 temp-controlled trucks, 6 drivers
- Cold-chain coverage: 100% — 27.0 vol capacity for 24.2 buffered cold demand
- Utilization: 85.6% — within target range
- Estimated cost: $2,450/day
- Contingency: If precip exceeds 20mm, escalate to risk 2 and add 25% buffer
- Monitor: Weather updates every 4 hours along I-95

### MA-I95 Corridor
- Risk Score: 0 → No buffer needed
- Travel time: 2.0 hours (well within SLA)
- Assign 2 standard + 2 temp-controlled trucks, 4 drivers
- Cold-chain coverage: 100%
- Utilization: 77.8% — within target range
- Estimated cost: $1,950/day

### Cross-Corridor Coordination
- NE-I95 has excess capacity; if MA-I95 demand spikes, rebalance 1 standard truck
- All drivers start from NJ distribution center

### KPI Impacts
- On-Time Pickup Rate: projected 96% (above 95% target)
- Cold-Chain Compliance: 100%
- SLA breach risk: LOW for both corridors
- Resource Utilization: NE=85.6%, MA=77.8%
""",
    "report_html": """
<html><body>
<h1>SeeWeeS Multi-Corridor Dispatch Report</h1>
<h2>Weather Risk Summary</h2>
<table><tr><td>Corridor</td><td>Risk Score</td><td>Travel (adj.)</td></tr>
<tr><td>NE-I95</td><td>1</td><td>5.06h</td></tr>
<tr><td>MA-I95</td><td>0</td><td>2.0h</td></tr></table>
<h2>Resource Allocation</h2>
<table><tr><td>Corridor</td><td>Trucks</td><td>Drivers</td><td>Cost</td></tr>
<tr><td>NE-I95</td><td>6</td><td>6</td><td>$2,450</td></tr>
<tr><td>MA-I95</td><td>5</td><td>5</td><td>$1,950</td></tr></table>
<p>Buffer applied: NE-I95 10% (risk 1)</p>
<p>SLA Status: All corridors within Tier 1 (6hr) and Tier 2 (12hr) limits</p>
<p>NSW Fairness Score: 0.82 | Max-Min Ratio: 0.85</p>
<p>Shipment volume: NE=42 units, MA=35 units. 0 excluded shipments.</p>
</body></html>
""",
    "validation_result": {
        "is_valid": True,
        "issues": [],
        "suggestions": ["Consider rebalancing 1 truck from NE to MA if MA demand increases"],
    },
    "business_context": "SeeWeeS specialty medicine dispatch from NJ to Boston hospitals via I-95.",
    "ops_insights": "42 shipments NE-I95, 35 shipments MA-I95. 53% cold-chain. 35 missing UIDs flagged.",
    "csv_kpis": {"otp": 0.96, "ccc": 1.0, "rur": 0.87},
    "anomalies_md": "35 rows with missing unique_item_id (DQ-01)",
    "agent_provider": "test",
    "model_name": "fixture",

    # Expected eval outcomes
    "expected": {
        "sla_compliance": True,
        "cold_chain_coverage": True,
        "buffer_policy": True,
        "cost_guardrails": True,
        "fairness_metrics": True,
        "weather_faithfulness": True,
        "overall_pass": True,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2: SLA BREACH
# ═══════════════════════════════════════════════════════════════════════════

SLA_BREACH: Dict[str, Any] = {
    "scenario_name": "sla_breach",
    "description": "NE-I95 at risk 3, adjusted travel=6.44h > 6h Tier 1 SLA. Plan should flag this.",
    "corridors": ["NE-I95", "MA-I95"],
    "corridor_weather_risks": [
        _weather("NE-I95", risk_score=3, precip=25.0, wind=55.0, temp=-2.0,
                 base_hrs=4.6, flags=["heavy_rain", "high_wind", "freezing"]),
        _weather("MA-I95", risk_score=1, precip=16.0, wind=30.0, temp=3.0,
                 base_hrs=2.0, distance_km=145.0, flags=["heavy_rain"]),
    ],
    "resource_allocation": {
        "allocations": {
            "NE-I95": _allocation("NE-I95", std=3, temp=4, drivers=7,
                                   risk=3, demand_vol=42, cold_vol=22, cost=2900.0),
            "MA-I95": _allocation("MA-I95", std=2, temp=3, drivers=5,
                                   risk=1, demand_vol=35, cold_vol=18, cost=1950.0),
        },
        "summary": {
            "total_daily_cost": 4850.0,
            "nsw_score": 0.75,
            "max_min_fairness_ratio": 0.78,
            "per_corridor_utility": {"NE-I95": 0.72, "MA-I95": 0.82},
            "available_pool": {"driver": 11, "truck_standard": 5, "truck_temp_controlled": 6},
        },
    },
    "dispatch_plan": """
## Multi-Corridor Dispatch Plan — SEVERE WEATHER ALERT

### NE-I95 Corridor — SLA BREACH RISK
- Risk Score: 3 (heavy_rain + high_wind + freezing) → 40% buffer + ESCALATION
- Adjusted travel time: 6.44 hours — EXCEEDS Tier 1 SLA (6 hours)
- ACTION: Escalate to regional manager. Consider early dispatch or alternate route.
- Assign 3 standard + 4 temp-controlled trucks, 7 drivers
- Cold-chain coverage: adequate with extra temp truck
- Buffer: 40% applied per policy
- Contingency: If conditions worsen, halt NE-I95 dispatches and reroute via MA

### MA-I95 Corridor
- Risk Score: 1 → 10% buffer
- Adjusted travel: 2.2 hours — well within SLA
- Standard operations with weather monitoring
- 2 standard + 2 temp trucks, 4 drivers

### KPI Impacts
- NE-I95 OTP at risk due to SLA breach
- Cold-Chain Compliance: maintained
- Cost: elevated due to extra resources for NE-I95
""",
    "report_html": """
<html><body>
<h1>SeeWeeS Dispatch Report — SEVERE WEATHER</h1>
<table><tr><td>Corridor</td><td>Risk</td><td>Travel</td><td>SLA Status</td></tr>
<tr><td>NE-I95</td><td style="color:red">3</td><td>6.44h</td><td style="color:red">BREACH</td></tr>
<tr><td>MA-I95</td><td>1</td><td>2.2h</td><td>OK</td></tr></table>
<p>Buffer: NE-I95 40%, MA-I95 10%</p>
<p>Shipment counts: NE=42, MA=35</p>
</body></html>
""",
    "validation_result": {
        "is_valid": False,
        "issues": ["NE-I95 adjusted_travel_hrs=6.44 exceeds Tier 1 SLA of 6 hours"],
        "suggestions": ["Consider early dispatch window or alternate routing for NE-I95"],
    },
    "business_context": "SeeWeeS specialty medicine dispatch.",
    "ops_insights": "42 NE-I95 shipments, 35 MA-I95. Severe weather incoming.",
    "csv_kpis": {"otp": 0.91, "ccc": 1.0, "rur": 0.82},
    "anomalies_md": "35 missing UIDs",
    "agent_provider": "test", "model_name": "fixture",
    "expected": {
        "sla_compliance": True,  # Plan correctly flags the breach
        "cold_chain_coverage": True,
        "buffer_policy": True,
        "weather_faithfulness": True,
        "overall_pass": True,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3: COLD-CHAIN GAP
# ═══════════════════════════════════════════════════════════════════════════

COLD_CHAIN_GAP: Dict[str, Any] = {
    "scenario_name": "cold_chain_gap",
    "description": "NE-I95 loses 1 temp truck (maintenance) — cold chain coverage drops below 100%",
    "corridors": ["NE-I95", "MA-I95"],
    "corridor_weather_risks": [
        _weather("NE-I95", risk_score=1, precip=16.0, flags=["heavy_rain"], base_hrs=4.6),
        _weather("MA-I95", risk_score=0, base_hrs=2.0, distance_km=145.0),
    ],
    "resource_allocation": {
        "allocations": {
            "NE-I95": _allocation("NE-I95", std=3, temp=2, drivers=5,
                                   risk=1, demand_vol=42, cold_vol=22, cost=2100.0),
            "MA-I95": _allocation("MA-I95", std=2, temp=3, drivers=5,
                                   risk=0, demand_vol=35, cold_vol=18, cost=1950.0),
        },
        "summary": {
            "total_daily_cost": 4050.0,
            "nsw_score": 0.70,
            "max_min_fairness_ratio": 0.72,
            "per_corridor_utility": {"NE-I95": 0.68, "MA-I95": 0.78},
            "available_pool": {"driver": 10, "truck_standard": 5, "truck_temp_controlled": 5},
        },
    },
    "dispatch_plan": """
## Dispatch Plan — Resource Constraint Alert

### NE-I95 — COLD-CHAIN WARNING
- 1 temp-controlled truck unavailable (maintenance)
- Cold-chain capacity: 18 vol units vs 24.2 needed (with 10% buffer)
- Coverage: 74.4% — BELOW 100% THRESHOLD
- ACTION: Request temp truck from MA-I95 or arrange emergency lease
- Risk score 1, 10% buffer applied

### MA-I95
- All resources available, risk 0, no buffer
- Has excess temp truck capacity — candidate for rebalancing to NE

### Contingency
- If NE temp truck not restored by 0600, split cold-chain shipments across 2 dispatches
- Monitor maintenance status hourly
""",
    "report_html": """
<html><body>
<h1>Dispatch Report</h1>
<table><tr><td>Corridor</td><td>Risk</td><td>Travel</td></tr>
<tr><td>NE-I95</td><td>1</td><td>5.06h</td></tr>
<tr><td>MA-I95</td><td>0</td><td>2.0h</td></tr></table>
<p>ALERT: NE-I95 cold-chain coverage at 74.4%</p>
<p>Buffer: NE 10%, MA 0%</p>
<p>Shipments: NE=42, MA=35</p>
<p>SLA: Within limits for both corridors</p>
</body></html>
""",
    "validation_result": {
        "is_valid": False,
        "issues": ["NE-I95 cold-chain coverage 74.4% < 100%"],
        "suggestions": ["Rebalance 1 temp truck from MA-I95 to NE-I95"],
    },
    "business_context": "SeeWeeS dispatch.", "ops_insights": "Resource constraint day.",
    "csv_kpis": {}, "anomalies_md": "",
    "agent_provider": "test", "model_name": "fixture",
    "expected": {
        "sla_compliance": True,
        "cold_chain_coverage": False,  # deliberately failing
        "buffer_policy": True,
        "overall_pass": False,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4: RESOURCE SHORTAGE (Driver shortage Day 14)
# ═══════════════════════════════════════════════════════════════════════════

RESOURCE_SHORTAGE: Dict[str, Any] = {
    "scenario_name": "resource_shortage",
    "description": "MA-I95 loses 2 drivers (sick) — drivers < trucks",
    "corridors": ["NE-I95", "MA-I95"],
    "corridor_weather_risks": [
        _weather("NE-I95", risk_score=0, base_hrs=4.6),
        _weather("MA-I95", risk_score=0, base_hrs=2.0, distance_km=145.0),
    ],
    "resource_allocation": {
        "allocations": {
            "NE-I95": _allocation("NE-I95", std=3, temp=3, drivers=6, risk=0,
                                   demand_vol=42, cold_vol=22, cost=2200.0),
            "MA-I95": _allocation("MA-I95", std=2, temp=2, drivers=2, risk=0,
                                   demand_vol=35, cold_vol=18, cost=1600.0),
        },
        "summary": {
            "total_daily_cost": 3800.0,
            "nsw_score": 0.65,
            "max_min_fairness_ratio": 0.60,
            "per_corridor_utility": {"NE-I95": 0.85, "MA-I95": 0.55},
            "available_pool": {"driver": 8, "truck_standard": 5, "truck_temp_controlled": 5},
        },
    },
    "dispatch_plan": """
## Dispatch Plan — Driver Shortage

### NE-I95 — Normal Operations
- All resources available, risk 0
- 3 std + 3 temp trucks, 6 drivers

### MA-I95 — DRIVER SHORTAGE
- Only 2 drivers available for 4 trucks
- Must prioritize: run 2 trucks (1 temp, 1 standard) with available drivers
- Cold-chain shipments get priority
- Remaining shipments delayed to next window

### KPI Impact
- MA-I95 OTP will drop below target
- Overall utilization affected
- Fairness ratio: 0.60 — below 0.7 threshold, flagged
""",
    "report_html": """
<html><body><h1>Report</h1>
<table><tr><td>Corridor</td><td>Risk</td><td>Travel</td></tr>
<tr><td>NE-I95</td><td>0</td><td>4.6h</td></tr>
<tr><td>MA-I95</td><td>0</td><td>2.0h</td></tr></table>
<p>ALERT: MA-I95 driver shortage (2 of 4 needed)</p>
<p>Shipments: NE=42, MA=35</p>
<p>SLA status: NE OK, MA at risk</p>
</body></html>
""",
    "validation_result": {
        "is_valid": False,
        "issues": ["MA-I95: 2 drivers for 4 trucks — insufficient"],
        "suggestions": ["Reassign 1 driver from NE-I95 or hire temp driver"],
    },
    "business_context": "SeeWeeS dispatch.", "ops_insights": "Day 14 driver shortage.",
    "csv_kpis": {}, "anomalies_md": "",
    "agent_provider": "test", "model_name": "fixture",
    "expected": {
        "resource_utilization": False,  # driver shortage
        "fairness_metrics": False,      # ratio < 0.7
        "overall_pass": False,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5: WEATHER HALLUCINATION
# ═══════════════════════════════════════════════════════════════════════════

WEATHER_HALLUCINATION: Dict[str, Any] = {
    "scenario_name": "weather_hallucination",
    "description": "Agent hallucinates snowfall data that doesn't exist in the inputs",
    "corridors": ["NE-I95"],
    "corridor_weather_risks": [
        _weather("NE-I95", risk_score=2, precip=20.0, wind=50.0, temp=1.0,
                 base_hrs=4.6, flags=["heavy_rain", "high_wind"]),
    ],
    "resource_allocation": {
        "allocations": {
            "NE-I95": _allocation("NE-I95", std=4, temp=3, drivers=7, risk=2,
                                   demand_vol=42, cold_vol=22, cost=2800.0),
        },
        "summary": {
            "total_daily_cost": 2800.0, "nsw_score": 0.80,
            "max_min_fairness_ratio": 1.0,
            "per_corridor_utility": {"NE-I95": 0.80},
            "available_pool": {"driver": 7, "truck_standard": 4, "truck_temp_controlled": 3},
        },
    },
    "dispatch_plan": """
## Dispatch Plan

### NE-I95
- SEVERE WEATHER: Heavy snowfall expected (6-8 inches of snow accumulation)
  with reduced visibility to 0.5 miles along the corridor
- Apply 25% buffer for risk score 2
- Deploy snow chains on all trucks
- Adjusted travel: 5.75h — within SLA but tight
""",
    "report_html": """
<html><body><h1>Report</h1>
<p>Snowfall warning: 6-8 inches expected along NE-I95</p>
<p>Visibility reduced to 0.5 miles</p>
<table><tr><td>NE-I95</td><td>Risk 2</td><td>5.75h</td></tr></table>
</body></html>
""",
    "validation_result": {"is_valid": True, "issues": [], "suggestions": []},
    "business_context": "", "ops_insights": "",
    "csv_kpis": {}, "anomalies_md": "",
    "agent_provider": "test", "model_name": "fixture",
    "expected": {
        "weather_faithfulness": False,  # hallucinated snow + visibility
        "overall_pass": False,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6: UNFAIR ALLOCATION
# ═══════════════════════════════════════════════════════════════════════════

UNFAIR_ALLOCATION: Dict[str, Any] = {
    "scenario_name": "unfair_allocation",
    "description": "NE-I95 gets all resources, MA-I95 starved. Fairness ratio < 0.5",
    "corridors": ["NE-I95", "MA-I95"],
    "corridor_weather_risks": [
        _weather("NE-I95", risk_score=0, base_hrs=4.6),
        _weather("MA-I95", risk_score=0, base_hrs=2.0, distance_km=145.0),
    ],
    "resource_allocation": {
        "allocations": {
            "NE-I95": _allocation("NE-I95", std=5, temp=5, drivers=10, risk=0,
                                   demand_vol=42, cold_vol=22, cost=2900.0),
            "MA-I95": _allocation("MA-I95", std=0, temp=0, drivers=0, risk=0,
                                   demand_vol=35, cold_vol=18, cost=0.0),
        },
        "summary": {
            "total_daily_cost": 2900.0, "nsw_score": 0.0,
            "max_min_fairness_ratio": 0.0,
            "per_corridor_utility": {"NE-I95": 0.95, "MA-I95": 0.0},
            "available_pool": {"driver": 10, "truck_standard": 5, "truck_temp_controlled": 5},
        },
    },
    "dispatch_plan": "All resources allocated to NE-I95. MA-I95 has no dispatches today.",
    "report_html": "<html><body><p>NE-I95 fully resourced. MA-I95 no capacity.</p></body></html>",
    "validation_result": {"is_valid": False, "issues": ["MA-I95 has zero resources"], "suggestions": []},
    "business_context": "", "ops_insights": "",
    "csv_kpis": {}, "anomalies_md": "",
    "agent_provider": "test", "model_name": "fixture",
    "expected": {
        "fairness_metrics": False,
        "cold_chain_coverage": False,
        "overall_pass": False,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7: EMPTY PLAN
# ═══════════════════════════════════════════════════════════════════════════

EMPTY_PLAN: Dict[str, Any] = {
    "scenario_name": "empty_plan",
    "description": "Agent produces empty or near-empty output",
    "corridors": ["NE-I95", "MA-I95"],
    "corridor_weather_risks": [
        _weather("NE-I95", risk_score=1, precip=16.0, flags=["heavy_rain"], base_hrs=4.6),
    ],
    "resource_allocation": None,
    "dispatch_plan": "",
    "report_html": "",
    "validation_result": None,
    "business_context": "", "ops_insights": "",
    "csv_kpis": {}, "anomalies_md": "",
    "agent_provider": "test", "model_name": "fixture",
    "expected": {
        "plan_completeness": False,
        "report_structure": False,
        "overall_pass": False,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

ALL_SCENARIOS = {
    "happy_path": HAPPY_PATH,
    "sla_breach": SLA_BREACH,
    "cold_chain_gap": COLD_CHAIN_GAP,
    "resource_shortage": RESOURCE_SHORTAGE,
    "weather_hallucination": WEATHER_HALLUCINATION,
    "unfair_allocation": UNFAIR_ALLOCATION,
    "empty_plan": EMPTY_PLAN,
}
