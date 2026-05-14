from __future__ import annotations
from typing import Dict, Any, List
import requests


###Harish changes###
from pathlib import Path
from typing import Any, Dict, List

from tools.policy_tools import DEFAULT_POLICY_PATH, load_dispatch_policy


TZ = "America/New_York"


def _policy_path(policy_path: str | Path | None = None) -> str:
    return str(policy_path or DEFAULT_POLICY_PATH)


def _weather_config(policy_path: str | Path | None = None) -> Dict[str, Any]:
    policy = load_dispatch_policy(_policy_path(policy_path))
    return {
        "corridors": policy.corridor_waypoints,
        "precip_threshold": policy.weather_thresholds["precipitation_sum_min_mm"],
        "gust_threshold": policy.weather_thresholds["wind_gusts_10m_max_min_kmh"],
        "freeze_threshold": policy.weather_thresholds["temperature_2m_min_max_c"],
        "forecast_days": policy.forecast_days,
        "travel_buffer_by_score": policy.travel_buffer_by_score,
        "escalation_score": policy.escalation_score,
    }


# Compatibility export for Pranit's graph/tests.
# Loaded from the default playbook, not hardcoded in this file.
CORRIDOR_WAYPOINTS: Dict[str, List[Dict[str, Any]]] = _weather_config()["corridors"]


def get_weather_forecast(
    lat: str,
    lon: str,
    tz: str = TZ,
    forecast_days: int | None = None,
    policy_path: str | Path | None = None,
) -> Dict[str, Any]:
    config = _weather_config(policy_path)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m",
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,wind_gusts_10m_max",
        "timezone": tz,
        "forecast_days": forecast_days or config["forecast_days"],
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def derive_dispatch_weather_risk(
    forecast: Dict[str, Any],
    policy_path: str | Path | None = None,
) -> Dict[str, Any]:
    config = _weather_config(policy_path)
    daily = forecast.get("daily", {})
    precip = daily.get("precipitation_sum", []) or []
    gusts = daily.get("wind_gusts_10m_max", []) or []
    tmin = daily.get("temperature_2m_min", []) or []

    max_precip = max(precip) if precip else 0.0
    max_gusts = max(gusts) if gusts else 0.0
    min_temp = min(tmin) if tmin else None

    flags = {
        "heavy_rain_risk": max_precip >= config["precip_threshold"],
        "high_wind_risk": max_gusts >= config["gust_threshold"],
        "freezing_risk": min_temp is not None and min_temp <= config["freeze_threshold"],
    }
    score = sum(int(v) for v in flags.values())

    return {
        "max_precip_mm_day": float(max_precip),
        "max_wind_gust_kmh": float(max_gusts),
        "min_temp_c": float(min_temp) if min_temp is not None else None,
        "risk_flags": flags,
        "risk_score_0_3": score,
    }


def _waypoint_day_scores(
    forecast: Dict[str, Any],
    config: Dict[str, Any],
) -> List[int]:
    daily = forecast.get("daily", {})
    precip = daily.get("precipitation_sum", []) or []
    gusts = daily.get("wind_gusts_10m_max", []) or []
    tmin = daily.get("temperature_2m_min", []) or []

    scores = []
    n_days = max(len(precip), len(gusts), len(tmin))
    for i in range(n_days):
        p = precip[i] if i < len(precip) else 0.0
        g = gusts[i] if i < len(gusts) else 0.0
        t = tmin[i] if i < len(tmin) else None

        scores.append(
            int((p or 0.0) >= config["precip_threshold"])
            + int((g or 0.0) >= config["gust_threshold"])
            + int(t is not None and t <= config["freeze_threshold"])
        )
    return scores


def get_single_corridor_weather_risk(
    corridor_id: str,
    corridors: Dict[str, List[Dict[str, Any]]] | None = None,
    tz: str = TZ,
    policy_path: str | Path | None = None,
) -> Dict[str, Any]:
    config = _weather_config(policy_path)
    active_corridors = corridors or config["corridors"]

    if corridor_id not in active_corridors:
        return {
            "error": f"Unknown corridor: {corridor_id}",
            "risk_score_48h": 0,
            "travel_buffer_pct": 0,
            "escalation_required": False,
        }

    return get_all_corridors_weather_risk(
        corridors={corridor_id: active_corridors[corridor_id]},
        tz=tz,
        policy_path=policy_path,
    ).get(corridor_id, {})


def get_all_corridors_weather_risk(
    corridors: Dict[str, List[Dict[str, Any]]] | None = None,
    tz: str = TZ,
    policy_path: str | Path | None = None,
) -> Dict[str, Any]:
    config = _weather_config(policy_path)
    active_corridors = corridors or config["corridors"]
    result: Dict[str, Any] = {}

    for corridor_id, waypoints in active_corridors.items():
        waypoint_details = []
        all_day_scores: List[List[int]] = []

        for wp in waypoints:
            try:
                forecast = get_weather_forecast(
                    lat=str(wp["lat"]),
                    lon=str(wp["lon"]),
                    tz=tz,
                    policy_path=policy_path,
                )
                day_scores = _waypoint_day_scores(forecast, config)
                daily = forecast.get("daily", {})
                waypoint_details.append({
                    "waypoint_id": wp["waypoint_id"],
                    "city": wp["city"],
                    "day_scores": day_scores,
                    "precip_by_day": daily.get("precipitation_sum", []),
                    "gusts_by_day": daily.get("wind_gusts_10m_max", []),
                    "tmin_by_day": daily.get("temperature_2m_min", []),
                })
                all_day_scores.append(day_scores)
            except Exception as e:
                print(f"  [weather] WARNING: failed to fetch {wp['waypoint_id']} ({wp['city']}): {e}")
                waypoint_details.append({
                    "waypoint_id": wp["waypoint_id"],
                    "city": wp["city"],
                    "day_scores": [0] * config["forecast_days"],
                    "error": str(e),
                })
                all_day_scores.append([0] * config["forecast_days"])

        n_days = max((len(s) for s in all_day_scores), default=config["forecast_days"])
        day_risk = [
            max(scores[day] if day < len(scores) else 0 for scores in all_day_scores)
            for day in range(n_days)
        ]
        risk_score_48h = max(day_risk) if day_risk else 0

        all_precip = [v for wp in waypoint_details for v in (wp.get("precip_by_day") or [])]
        all_gusts = [v for wp in waypoint_details for v in (wp.get("gusts_by_day") or [])]
        all_tmin = [v for wp in waypoint_details for v in (wp.get("tmin_by_day") or [])]

        risk_flags_48h = {
            "heavy_rain_risk": any((v or 0) >= config["precip_threshold"] for v in all_precip),
            "high_wind_risk": any((v or 0) >= config["gust_threshold"] for v in all_gusts),
            "freezing_risk": any(v is not None and v <= config["freeze_threshold"] for v in all_tmin),
        }

        result[corridor_id] = {
            "waypoints": waypoint_details,
            "day_risk": day_risk,
            "risk_score_48h": risk_score_48h,
            "risk_flags_48h": risk_flags_48h,
            "travel_buffer_pct": config["travel_buffer_by_score"].get(
                risk_score_48h,
                max(config["travel_buffer_by_score"].values()),
            ),
            "escalation_required": risk_score_48h >= config["escalation_score"],
        }

    return result