from __future__ import annotations

import math
from typing import Any, Dict

import pandas as pd

from tools.policy_tools import DispatchPolicy, load_dispatch_policy


def load_resource_availability(csv_path: str) -> Dict[str, Dict[str, int]]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    availability: Dict[str, Dict[str, int]] = {}
    for _, row in df.iterrows():
        day = str(row["day"]).strip()
        resource_type = str(row["resource_type"]).strip()
        availability.setdefault(day, {})[resource_type] = int(row["available_count"])

    return availability


def allocate_resources(
    corridor_day_summary: Dict[str, Any],
    availability: Dict[str, Dict[str, int]],
    corridor_weather_risk: Dict[str, Any] | None = None,
    policy_path: str | None = None,
) -> Dict[str, Any]:
    policy = load_dispatch_policy(policy_path)
    allocation: Dict[str, Any] = {}
    total_penalty = 0
    tier1_units_impacted = 0

    for day in ["Day0", "Day1"]:
        remaining = dict(availability.get(day, {}))
        day_plan: Dict[str, Any] = {
                  "available": dict(remaining),
                  "corridors": {},
                  "day_total_penalty": 0,
                  "resource_shortfall": {},
       }


        corridor_order = [c for c in policy.corridor_priority if c in corridor_day_summary]
        corridor_order.extend(c for c in corridor_day_summary if c not in corridor_order)

        for corridor_id in corridor_order:
            stats = corridor_day_summary.get(corridor_id, {}).get(day, {})
            if not stats:
                continue

            sla_tier = stats.get("sla_tier", "Tier 2")
            need_temp = int(stats.get("required_temp_trucks", 0))
            need_std = int(stats.get("required_std_trucks", 0))
            temp_units = int(stats.get("temp_controlled_units", 0))
            std_units = int(stats.get("standard_units", 0))
            total_units = int(stats.get("total_valid_units", 0))

            allocated_temp = min(need_temp, int(remaining.get("truck_temp_controlled", 0)))
            remaining["truck_temp_controlled"] = int(remaining.get("truck_temp_controlled", 0)) - allocated_temp
            temp_shortfall = need_temp - allocated_temp

            allocated_std = min(need_std, int(remaining.get("truck_standard", 0)))
            remaining["truck_standard"] = int(remaining.get("truck_standard", 0)) - allocated_std
            std_shortfall = need_std - allocated_std

            drivers_needed = allocated_temp + allocated_std
            allocated_drivers = min(drivers_needed, int(remaining.get("driver", 0)))
            remaining["driver"] = int(remaining.get("driver", 0)) - allocated_drivers
            driver_shortfall = drivers_needed - allocated_drivers

            units_per_truck = max(1, math.floor(policy.truck_capacity / policy.packing_buffer))
            undelivered_temp = min(temp_units, temp_shortfall * units_per_truck)
            undelivered_std = min(std_units, std_shortfall * units_per_truck)
            undelivered_driver = min(total_units, driver_shortfall * units_per_truck)
            undelivered_units = min(total_units, undelivered_temp + undelivered_std + undelivered_driver)

            sla_penalty_rate = (
                policy.penalty["tier1_sla_violation"]
                if sla_tier == "Tier 1"
                else policy.penalty["tier2_sla_violation"]
            )
            corridor_penalty = (
                undelivered_units * sla_penalty_rate
                + undelivered_temp * policy.penalty["cold_chain_violation"]
            )

            wx = (corridor_weather_risk or {}).get(corridor_id, {})
            weather_score = int(wx.get("risk_score_48h", wx.get("risk_score_0_3", 0)))

            if sla_tier == "Tier 1":
                tier1_units_impacted += undelivered_units

            total_penalty += corridor_penalty
            day_plan["day_total_penalty"] += corridor_penalty
            day_plan["corridors"][corridor_id] = {
                "sla_tier": sla_tier,
                "total_units": total_units,
                "allocated_temp_trucks": allocated_temp,
                "allocated_std_trucks": allocated_std,
                "allocated_drivers": allocated_drivers,
                "shortfall_temp_trucks": temp_shortfall,
                "shortfall_std_trucks": std_shortfall,
                "shortfall_drivers": driver_shortfall,
                "undelivered_units": undelivered_units,
                "corridor_penalty": corridor_penalty,
                "can_dispatch_all": corridor_penalty == 0,
                "weather_risk_score": weather_score,
                "travel_buffer_pct": _travel_buffer(weather_score, policy),
                "escalation_required": weather_score >= policy.escalation_score,
            }

        day_plan["remaining_pool"] = dict(remaining)
        allocation[day] = day_plan

    allocation["summary_48h"] = {
        "total_penalty_score": total_penalty,
        "tier1_units_impacted": tier1_units_impacted,
        "allocation_feasible": total_penalty == 0,
        "recommendation": _summarise_recommendation(allocation, total_penalty),
    }
    return allocation


def _travel_buffer(risk_score: int, policy: DispatchPolicy) -> int:
    return policy.travel_buffer_by_score.get(
        risk_score,
        max(policy.travel_buffer_by_score.values()),
    )


def _summarise_recommendation(allocation: Dict[str, Any], total_penalty: int) -> str:
    if total_penalty == 0:
        return "All Day0/Day1 corridor demand can be served within available resources."

    lines = [f"Total penalty score: {total_penalty}. Shortfalls detected:"]
    for day in ["Day0", "Day1"]:
        for corridor_id, stats in allocation.get(day, {}).get("corridors", {}).items():
            if not stats.get("can_dispatch_all"):
                lines.append(
                    f"{day} {corridor_id}: {stats['undelivered_units']} units undelivered "
                    f"with {stats['corridor_penalty']} penalty points."
                )
    return " ".join(lines)
