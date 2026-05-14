from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))


def test_csv_reconciliation_and_48h_kpis():
    from tools.csv_tools import analyze_csv

    result = analyze_csv(str(ROOT / "data-for-enhancement" / "Incoming_shipments_14d_multi_corridor.csv"))

    assert result.summary["rows_original"] == 129
    assert result.summary["rows_planning_window"] == 30
    assert result.summary["rows_excluded_dq01"] == 3
    assert result.kpis["total_excluded_dq01"] == 5
    assert result.kpis["total_flagged_dq02"] == 0
    assert result.kpis["total_flagged_dq03"] == 0
    assert result.kpis["total_flagged_dq04"] == 0

    assert result.kpis["C1_I95_NJ_BOS_total_units_48h"] == 16
    assert result.kpis["C2_NJ_PHL_total_units_48h"] == 14

    c1_day0 = result.corridor_day_summary["C1_I95_NJ_BOS"]["Day0"]
    assert c1_day0["sla_tier"] == "Tier 1"
    assert c1_day0["total_valid_units"] == 8
    assert c1_day0["temp_controlled_units"] == 5
    assert c1_day0["required_temp_trucks"] == 1
    assert c1_day0["required_std_trucks"] == 1

    fixed_legacy = result.planning_df[result.planning_df["unique_item_id"] == "ALB-2026-1501"].iloc[0]
    assert fixed_legacy["canonical_item_id"] == "ALB-INH"
    assert fixed_legacy["reconcile_confidence"] == "LEGACY_ID_MAP"

    fixed_alias = result.planning_df[result.planning_df["unique_item_id"] == "PBR-2026-0302"].iloc[0]
    assert fixed_alias["canonical_item_id"] == "PMB-KEY"
    assert fixed_alias["reconcile_confidence"] == "ALIAS_MATCH"


def test_weather_risk_thresholds_and_corridor_aggregation(monkeypatch):
    from tools import weather_tools

    forecasts_by_lat = {
        "1.0": {
            "daily": {
                "precipitation_sum": [0.0, 16.0],
                "wind_gusts_10m_max": [20.0, 30.0],
                "temperature_2m_min": [5.0, 4.0],
            }
        },
        "2.0": {
            "daily": {
                "precipitation_sum": [0.0, 0.0],
                "wind_gusts_10m_max": [46.0, 10.0],
                "temperature_2m_min": [2.0, -1.0],
            }
        },
    }

    def fake_forecast(lat: str, lon: str, tz: str = weather_tools.TZ):
        return forecasts_by_lat[lat]

    monkeypatch.setattr(weather_tools, "get_weather_forecast", fake_forecast)

    risk = weather_tools.get_all_corridors_weather_risk(
        corridors={
            "TEST": [
                {"waypoint_id": "W1", "city": "A", "lat": 1.0, "lon": 1.0},
                {"waypoint_id": "W2", "city": "B", "lat": 2.0, "lon": 2.0},
            ]
        }
    )["TEST"]

    assert risk["day_risk"] == [1, 1]
    assert risk["risk_score_48h"] == 1
    assert risk["travel_buffer_pct"] == 10
    assert risk["escalation_required"] is False
    assert risk["risk_flags_48h"] == {
        "heavy_rain_risk": True,
        "high_wind_risk": True,
        "freezing_risk": True,
    }


def test_resource_allocator_prioritizes_tier1_when_capacity_is_scarce():
    from tools.resource_tools import allocate_resources

    corridor_day_summary = {
        "C1_I95_NJ_BOS": {
            "Day0": {
                "total_valid_units": 10,
                "temp_controlled_units": 5,
                "standard_units": 5,
                "required_temp_trucks": 1,
                "required_std_trucks": 1,
                "required_drivers": 2,
            },
            "Day1": {
                "total_valid_units": 0,
                "temp_controlled_units": 0,
                "standard_units": 0,
                "required_temp_trucks": 0,
                "required_std_trucks": 0,
                "required_drivers": 0,
            },
        },
        "C2_NJ_PHL": {
            "Day0": {
                "total_valid_units": 10,
                "temp_controlled_units": 5,
                "standard_units": 5,
                "required_temp_trucks": 1,
                "required_std_trucks": 1,
                "required_drivers": 2,
            },
            "Day1": {
                "total_valid_units": 0,
                "temp_controlled_units": 0,
                "standard_units": 0,
                "required_temp_trucks": 0,
                "required_std_trucks": 0,
                "required_drivers": 0,
            },
        },
    }
    availability = {
        "Day0": {"driver": 2, "truck_standard": 1, "truck_temp_controlled": 1},
        "Day1": {"driver": 0, "truck_standard": 0, "truck_temp_controlled": 0},
    }

    allocation = allocate_resources(corridor_day_summary, availability)
    c1 = allocation["Day0"]["corridors"]["C1_I95_NJ_BOS"]
    c2 = allocation["Day0"]["corridors"]["C2_NJ_PHL"]

    assert c1["can_dispatch_all"] is True
    assert c1["corridor_penalty"] == 0
    assert c2["can_dispatch_all"] is False
    assert c2["undelivered_units"] == 10
    assert c2["corridor_penalty"] == 800
    assert allocation["summary_48h"]["total_penalty_score"] == 800
    assert allocation["summary_48h"]["tier1_units_impacted"] == 0


def test_validator_retries_when_escalated_corridor_is_missing_from_plan():
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-local-evals")

    import graph

    state = {
        "resource_allocation": {
            "Day0": {
                "available": {"truck_temp_controlled": 1},
                "corridors": {
                    "C1_I95_NJ_BOS": {
                        "shortfall_temp_trucks": 0,
                        "shortfall_std_trucks": 0,
                        "shortfall_drivers": 0,
                        "allocated_temp_trucks": 1,
                    }
                },
            },
            "Day1": {
                "available": {"truck_temp_controlled": 1},
                "corridors": {},
            },
        },
        "corridor_weather_risk": {
            "C2_NJ_PHL": {"risk_score_48h": 3, "escalation_required": True}
        },
        "dispatch_plan": "Prioritize Boston shipments with normal monitoring.",
        "planner_retry_count": 0,
    }

    result = graph.node_validate(state)

    assert result["planner_retry_count"] == 1
    assert len(result["validation_violations"]) == 1
    assert "C2_NJ_PHL requires escalation" in result["validation_violations"][0]
    assert graph.route_after_validate(result) == "planner"


def test_embedded_eval_adapter_translates_project_state():
    from evals.adapter_mia import from_mia_state
    from evals.validators_mia import run_mia_validators

    state = {
        "dispatch_plan": (
            "Day0 and Day1 cover C1_I95_NJ_BOS and C2_NJ_PHL. "
            "C2_NJ_PHL requires escalation, weather monitoring, trucks, drivers, "
            "SLA tracking, contingency triggers, and KPI review."
        ),
        "report_html": (
            "<html><body><h2>Weather Risk Summary</h2><table><tr><td>corridor</td></tr>"
            "<tr><td>C1_I95_NJ_BOS</td></tr><tr><td>C2_NJ_PHL</td></tr></table>"
            "<h2>Resource Allocation</h2><p>travel buffer, shipment count, SLA tier</p>"
            "</body></html>"
        ),
        "corridor_weather_risk": {
            "C1_I95_NJ_BOS": {
                "risk_score_48h": 1,
                "risk_flags_48h": {
                    "heavy_rain_risk": True,
                    "high_wind_risk": False,
                    "freezing_risk": False,
                },
                "travel_buffer_pct": 10,
                "escalation_required": False,
                "waypoints": [
                    {
                        "waypoint_id": "C1_W1",
                        "city": "Newark NJ",
                        "precip_by_day": [16.0, 0.0],
                        "gusts_by_day": [20.0, 20.0],
                        "tmin_by_day": [5.0, 5.0],
                    }
                ],
            },
            "C2_NJ_PHL": {
                "risk_score_48h": 3,
                "risk_flags_48h": {
                    "heavy_rain_risk": True,
                    "high_wind_risk": True,
                    "freezing_risk": True,
                },
                "travel_buffer_pct": 40,
                "escalation_required": True,
                "waypoints": [
                    {
                        "waypoint_id": "C2_W4",
                        "city": "Philadelphia PA",
                        "precip_by_day": [16.0, 0.0],
                        "gusts_by_day": [46.0, 20.0],
                        "tmin_by_day": [-1.0, 5.0],
                    }
                ],
            },
        },
        "resource_allocation": {
            "Day0": {
                "available": {"driver": 4, "truck_standard": 2, "truck_temp_controlled": 2},
                "corridors": {
                    "C1_I95_NJ_BOS": {
                        "total_units": 8,
                        "temp_controlled_units": 4,
                        "allocated_temp_trucks": 1,
                        "allocated_std_trucks": 1,
                        "allocated_drivers": 2,
                        "shortfall_temp_trucks": 0,
                        "shortfall_std_trucks": 0,
                        "shortfall_drivers": 0,
                        "corridor_penalty": 0,
                        "can_dispatch_all": True,
                        "weather_risk_score": 1,
                    },
                    "C2_NJ_PHL": {
                        "total_units": 7,
                        "temp_controlled_units": 3,
                        "allocated_temp_trucks": 1,
                        "allocated_std_trucks": 1,
                        "allocated_drivers": 2,
                        "shortfall_temp_trucks": 0,
                        "shortfall_std_trucks": 0,
                        "shortfall_drivers": 0,
                        "corridor_penalty": 0,
                        "can_dispatch_all": True,
                        "weather_risk_score": 3,
                    },
                },
                "day_total_penalty": 0,
                "remaining_pool": {"driver": 0, "truck_standard": 0, "truck_temp_controlled": 0},
            },
            "Day1": {
                "available": {"driver": 0, "truck_standard": 0, "truck_temp_controlled": 0},
                "corridors": {},
                "day_total_penalty": 0,
                "remaining_pool": {"driver": 0, "truck_standard": 0, "truck_temp_controlled": 0},
            },
            "summary_48h": {
                "total_penalty_score": 0,
                "allocation_feasible": True,
            },
        },
        "validation_violations": [],
        "planner_retry_count": 0,
    }

    output = from_mia_state(state)
    assert output.corridors == ["C1_I95_NJ_BOS", "C2_NJ_PHL"]
    assert output.corridor_weather_risks[0].adjusted_travel_hrs == 5.06
    assert output.corridor_weather_risks[1].adjusted_travel_hrs == 2.52
    assert output.resource_allocation is not None
    assert set(output.resource_allocation.allocations) == {"C1_I95_NJ_BOS", "C2_NJ_PHL"}

    mia_results = {result.name: result for result in run_mia_validators(state)}
    assert mia_results["penalty_model"].passed
    assert mia_results["escalation_check"].passed
    assert mia_results["day_feasibility"].passed
