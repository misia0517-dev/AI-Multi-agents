from __future__ import annotations
import math
from typing import Dict, Any, List
import pandas as pd


PENALTY = {
    "tier1_sla_violation":   100,
    "tier2_sla_violation":    40,
    "cold_chain_violation":   80,
    "non_sla_delay":          10,
}

CORRIDOR_SLA_TIER = {
    "C1_I95_NJ_BOS": "Tier 1",
    "C2_NJ_PHL":     "Tier 2",
}

CORRIDOR_PRIORITY = ["C1_I95_NJ_BOS", "C2_NJ_PHL"]

TRUCK_CAPACITY = 10
PACKING_BUFFER = 1.10


def load_resource_availability(csv_path: str) -> Dict[str, Dict[str, int]]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    availability: Dict[str, Dict[str, int]] = {}
    for _, row in df.iterrows():
        day   = str(row["day"]).strip()
        rtype = str(row["resource_type"]).strip()
        count = int(row["available_count"])
        availability.setdefault(day, {})[rtype] = count

    return availability


def allocate_resources(
    corridor_day_summary: Dict[str, Any],
    availability: Dict[str, Dict[str, int]],
    corridor_weather_risk: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    allocation_plan: Dict[str, Any] = {}
    total_penalty = 0
    tier1_units_impacted = 0

    for day in ["Day0", "Day1"]:
        avail = dict(availability.get(day, {}))
        day_plan: Dict[str, Any] = {
            "available": dict(avail),
            "corridors": {},
            "day_total_penalty": 0,
            "resource_shortfall": {},
        }

        for corridor_id in CORRIDOR_PRIORITY:
            if corridor_id not in corridor_day_summary:
                continue

            stats    = corridor_day_summary[corridor_id].get(day, {})
            sla_tier = CORRIDOR_SLA_TIER.get(corridor_id, "Tier 2")

            need_temp   = stats.get("required_temp_trucks", 0)
            need_std    = stats.get("required_std_trucks",  0)
            need_drv    = stats.get("required_drivers",     0)
            temp_units  = stats.get("temp_controlled_units", 0)
            std_units   = stats.get("standard_units", 0)
            total_units = stats.get("total_valid_units", 0)

            wx_risk = 0
            if corridor_weather_risk and corridor_id in corridor_weather_risk:
                wx_risk = corridor_weather_risk[corridor_id].get("risk_score_48h", 0)

            allocated_temp = min(need_temp, avail.get("truck_temp_controlled", 0))
            avail["truck_temp_controlled"] = max(0, avail.get("truck_temp_controlled", 0) - allocated_temp)
            temp_shortfall = need_temp - allocated_temp

            allocated_std = min(need_std, avail.get("truck_standard", 0))
            avail["truck_standard"] = max(0, avail.get("truck_standard", 0) - allocated_std)
            std_shortfall = need_std - allocated_std

            drivers_needed_actual = allocated_temp + allocated_std
            allocated_drv = min(drivers_needed_actual, avail.get("driver", 0))
            avail["driver"] = max(0, avail.get("driver", 0) - allocated_drv)
            drv_shortfall = drivers_needed_actual - allocated_drv

            units_per_truck = math.floor(TRUCK_CAPACITY / PACKING_BUFFER)

            undelivered_temp = min(temp_units, temp_shortfall * units_per_truck)
            undelivered_std  = min(std_units,  std_shortfall  * units_per_truck)
            undelivered_drv  = drv_shortfall * units_per_truck

            total_undelivered = min(total_units, undelivered_temp + undelivered_std + undelivered_drv)

            sla_penalty_rate  = PENALTY["tier1_sla_violation"] if sla_tier == "Tier 1" else PENALTY["tier2_sla_violation"]
            cold_penalty_rate = PENALTY["cold_chain_violation"]

            penalty_sla      = int(total_undelivered) * sla_penalty_rate
            penalty_cold     = int(undelivered_temp)  * cold_penalty_rate
            corridor_penalty = penalty_sla + penalty_cold

            total_penalty += corridor_penalty
            if sla_tier == "Tier 1":
                tier1_units_impacted += int(total_undelivered)

            can_dispatch_all = (temp_shortfall == 0 and std_shortfall == 0 and drv_shortfall == 0)

            day_plan["corridors"][corridor_id] = {
                "sla_tier":              sla_tier,
                "total_units":           total_units,
                "allocated_temp_trucks": allocated_temp,
                "allocated_std_trucks":  allocated_std,
                "allocated_drivers":     allocated_drv,
                "shortfall_temp_trucks": temp_shortfall,
                "shortfall_std_trucks":  std_shortfall,
                "shortfall_drivers":     drv_shortfall,
                "undelivered_units":     int(total_undelivered),
                "corridor_penalty":      corridor_penalty,
                "can_dispatch_all":      can_dispatch_all,
                "weather_risk_score":    wx_risk,
                "travel_buffer_pct":     _travel_buffer(wx_risk),
                "escalation_required":   wx_risk >= 3,
            }
            day_plan["day_total_penalty"] += corridor_penalty

        day_plan["remaining_pool"] = dict(avail)
        allocation_plan[day] = day_plan

    allocation_plan["summary_48h"] = {
        "total_penalty_score":    total_penalty,
        "tier1_units_impacted":   tier1_units_impacted,
        "allocation_feasible":    total_penalty == 0,
        "recommendation":         _summarise_recommendation(allocation_plan, total_penalty),
    }

    return allocation_plan


def _travel_buffer(risk_score: int) -> int:
    return {0: 0, 1: 10, 2: 25, 3: 40}.get(risk_score, 40)


def _summarise_recommendation(plan: Dict[str, Any], total_penalty: int) -> str:
    if total_penalty == 0:
        return "All corridors can be fully served within available resources."

    lines = [f"Total penalty score: {total_penalty}. Shortfalls detected:"]
    for day in ["Day0", "Day1"]:
        for corridor_id, stats in plan.get(day, {}).get("corridors", {}).items():
            if not stats.get("can_dispatch_all"):
                lines.append(
                    f"  [{day}] {corridor_id} ({stats['sla_tier']}): "
                    f"{stats['undelivered_units']} units undelivered. "
                    f"Penalty: {stats['corridor_penalty']} pts."
                )
    return " ".join(lines)
