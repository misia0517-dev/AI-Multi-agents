from __future__ import annotations
from typing import Dict, Any, List
import requests


# ---------------------------------------------------------------------------
# Corridor waypoints (from SeeWeeS Specialty Dispatch Playbook, Section 3.2)
# ---------------------------------------------------------------------------
CORRIDOR_WAYPOINTS: Dict[str, List[Dict[str, Any]]] = {
    "C1_I95_NJ_BOS": [
        {"waypoint_id": "C1_W1", "city": "Newark NJ",      "lat": 40.7357, "lon": -74.1724},
        {"waypoint_id": "C1_W2", "city": "Bronx NY",       "lat": 40.8448, "lon": -73.8648},
        {"waypoint_id": "C1_W3", "city": "New Haven CT",   "lat": 41.3083, "lon": -72.9279},
        {"waypoint_id": "C1_W4", "city": "Providence RI",  "lat": 41.8240, "lon": -71.4128},
        {"waypoint_id": "C1_W5", "city": "Boston MA",      "lat": 42.3601, "lon": -71.0589},
    ],
    "C2_NJ_PHL": [
        {"waypoint_id": "C2_W1", "city": "Newark NJ",         "lat": 40.7357, "lon": -74.1724},
        {"waypoint_id": "C2_W2", "city": "New Brunswick NJ",  "lat": 40.4862, "lon": -74.4518},
        {"waypoint_id": "C2_W3", "city": "Trenton NJ",        "lat": 40.2204, "lon": -74.7643},
        {"waypoint_id": "C2_W4", "city": "Philadelphia PA",   "lat": 39.9526, "lon": -75.1652},
    ],
}

TZ = "America/New_York"

PRECIP_THRESHOLD_MM  = 15.0
GUST_THRESHOLD_KMH   = 45.0
FREEZE_THRESHOLD_C   =  0.0


def get_weather_forecast(lat: str, lon: str, tz: str = TZ) -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m",
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,wind_gusts_10m_max",
        "timezone": tz,
        "forecast_days": 2,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def derive_dispatch_weather_risk(forecast: Dict[str, Any]) -> Dict[str, Any]:
    daily = forecast.get("daily", {})
    precip = daily.get("precipitation_sum", []) or []
    gusts  = daily.get("wind_gusts_10m_max", []) or []
    tmin   = daily.get("temperature_2m_min", []) or []

    max_precip = max(precip) if precip else 0.0
    max_gusts  = max(gusts)  if gusts  else 0.0
    min_temp   = min(tmin)   if tmin   else None

    flags = {
        "heavy_rain_risk": max_precip >= PRECIP_THRESHOLD_MM,
        "high_wind_risk":  max_gusts  >= GUST_THRESHOLD_KMH,
        "freezing_risk":   (min_temp is not None and min_temp <= FREEZE_THRESHOLD_C),
    }
    score = sum(int(v) for v in flags.values())

    return {
        "max_precip_mm_day": float(max_precip),
        "max_wind_gust_kmh": float(max_gusts),
        "min_temp_c":        float(min_temp) if min_temp is not None else None,
        "risk_flags":        flags,
        "risk_score_0_3":    score,
    }


def _waypoint_day_scores(forecast: Dict[str, Any]) -> List[int]:
    daily  = forecast.get("daily", {})
    precip = daily.get("precipitation_sum", []) or []
    gusts  = daily.get("wind_gusts_10m_max", []) or []
    tmin   = daily.get("temperature_2m_min", []) or []

    scores = []
    n_days = max(len(precip), len(gusts), len(tmin))
    for i in range(n_days):
        p = precip[i] if i < len(precip) else 0.0
        g = gusts[i]  if i < len(gusts)  else 0.0
        t = tmin[i]   if i < len(tmin)   else None

        s = (
            int((p or 0.0) >= PRECIP_THRESHOLD_MM)
            + int((g or 0.0) >= GUST_THRESHOLD_KMH)
            + int(t is not None and t <= FREEZE_THRESHOLD_C)
        )
        scores.append(s)
    return scores


def get_all_corridors_weather_risk(
    corridors: Dict[str, List[Dict[str, Any]]] = CORRIDOR_WAYPOINTS,
    tz: str = TZ,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    for corridor_id, waypoints in corridors.items():
        waypoint_details = []
        all_day_scores: List[List[int]] = []

        for wp in waypoints:
            try:
                forecast = get_weather_forecast(
                    lat=str(wp["lat"]),
                    lon=str(wp["lon"]),
                    tz=tz,
                )
                day_scores = _waypoint_day_scores(forecast)
                daily = forecast.get("daily", {})
                waypoint_details.append({
                    "waypoint_id":   wp["waypoint_id"],
                    "city":          wp["city"],
                    "day_scores":    day_scores,
                    "precip_by_day": daily.get("precipitation_sum", []),
                    "gusts_by_day":  daily.get("wind_gusts_10m_max", []),
                    "tmin_by_day":   daily.get("temperature_2m_min", []),
                })
                all_day_scores.append(day_scores)
            except Exception as e:
                print(f"  [weather] WARNING: failed to fetch {wp['waypoint_id']} ({wp['city']}): {e}")
                waypoint_details.append({
                    "waypoint_id": wp["waypoint_id"],
                    "city":        wp["city"],
                    "day_scores":  [0, 0],
                    "error":       str(e),
                })
                all_day_scores.append([0, 0])

        n_days = max((len(s) for s in all_day_scores), default=2)
        day_risk = []
        for d in range(n_days):
            day_risk.append(max(s[d] if d < len(s) else 0 for s in all_day_scores))

        risk_score_48h = max(day_risk) if day_risk else 0

        all_precip = [v for wp in waypoint_details for v in (wp.get("precip_by_day") or [])]
        all_gusts  = [v for wp in waypoint_details for v in (wp.get("gusts_by_day")  or [])]
        all_tmin   = [v for wp in waypoint_details for v in (wp.get("tmin_by_day")   or [])]

        risk_flags_48h = {
            "heavy_rain_risk": any((v or 0) >= PRECIP_THRESHOLD_MM for v in all_precip),
            "high_wind_risk":  any((v or 0) >= GUST_THRESHOLD_KMH  for v in all_gusts),
            "freezing_risk":   any(v is not None and v <= FREEZE_THRESHOLD_C for v in all_tmin),
        }

        buffer_map = {0: 0, 1: 10, 2: 25, 3: 40}
        travel_buffer_pct = buffer_map.get(risk_score_48h, 40)

        result[corridor_id] = {
            "waypoints":           waypoint_details,
            "day_risk":            day_risk,
            "risk_score_48h":      risk_score_48h,
            "risk_flags_48h":      risk_flags_48h,
            "travel_buffer_pct":   travel_buffer_pct,
            "escalation_required": risk_score_48h >= 3,
        }

    return result
