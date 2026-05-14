from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_POLICY_PATH = Path("data-for-enhancement/SeeWeeS Specialty Dispatch Playbook.md")


@dataclass(frozen=True)
class DispatchPolicy:
    canonical_items: Dict[str, Dict[str, Any]]
    name_alias: Dict[str, str]
    legacy_id_map: Dict[int, str]
    corridor_sla_tier: Dict[str, str]
    corridor_priority: List[str]
    corridor_waypoints: Dict[str, List[Dict[str, Any]]]
    special_case_item_ids: List[int]
    truck_capacity: int
    packing_buffer: float
    penalty: Dict[str, int]
    weather_thresholds: Dict[str, float]
    travel_buffer_by_score: Dict[int, int]
    escalation_score: int
    forecast_days: int


def load_dispatch_policy(policy_path: str | Path | None = None) -> DispatchPolicy:
    path = Path(policy_path or DEFAULT_POLICY_PATH)
    return _load_dispatch_policy_cached(str(path))


@lru_cache(maxsize=8)
def _load_dispatch_policy_cached(policy_path: str) -> DispatchPolicy:
    path = Path(policy_path)
    text = path.read_text(encoding="utf-8")

    canonical_items = _parse_canonical_items(text)
    _merge_item_truth_names(text, canonical_items)

    return DispatchPolicy(
        canonical_items=canonical_items,
        name_alias=_parse_alias_map(text),
        legacy_id_map=_parse_legacy_id_map(text),
        corridor_sla_tier=_parse_corridor_sla_tier(text),
        corridor_priority=list(_parse_corridor_sla_tier(text).keys()),
        corridor_waypoints=_parse_corridor_waypoints(text),
        special_case_item_ids=_parse_special_case_item_ids(text),
        truck_capacity=_parse_truck_capacity(text),
        packing_buffer=_parse_packing_buffer(text),
        penalty=_parse_penalties(text),
        weather_thresholds=_parse_weather_thresholds(text),
        travel_buffer_by_score=_parse_travel_buffers(text),
        escalation_score=_parse_escalation_score(text),
        forecast_days=_parse_forecast_days(text),
    )


def _parse_markdown_table_after(text: str, marker: str) -> List[Dict[str, str]]:
    idx = text.find(marker)
    if idx == -1:
        return []

    lines = text[idx:].splitlines()
    table_lines: List[str] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
            in_table = True
        elif in_table:
            break

    return _parse_markdown_table(table_lines)


def _parse_markdown_table(table_lines: List[str]) -> List[Dict[str, str]]:
    if len(table_lines) < 2:
        return []

    headers = [_clean_cell(c) for c in table_lines[0].strip("|").split("|")]
    rows: List[Dict[str, str]] = []
    for line in table_lines[2:]:
        cells = [_clean_cell(c) for c in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def _clean_cell(value: str) -> str:
    return value.strip().strip("`").strip()


def _parse_canonical_items(text: str) -> Dict[str, Dict[str, Any]]:
    rows = _parse_markdown_table_after(text, "| canonical_item_id | item_id | canonical_item_name |")
    if not rows:
        raise ValueError("Policy playbook is missing the canonical item table.")

    items: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        canonical_id = row["canonical_item_id"]
        items[canonical_id] = {
            "item_id": int(row["item_id"]),
            "item_name": row["canonical_item_name"],
            "temp_control": row["temp_control"],
            "medicine_type": row.get("medicine_type", ""),
            "product_class": row.get("product_class", ""),
            "accepted_names": {row["canonical_item_name"].strip().lower()},
        }
    return items


def _merge_item_truth_names(text: str, canonical_items: Dict[str, Dict[str, Any]]) -> None:
    truth_rows = _parse_markdown_table_after(text, "| item_id | item_name | medicine_type | temp_control |")
    for row in truth_rows:
        item_id = int(row["item_id"])
        item_name = row["item_name"].strip()
        temp_control = row["temp_control"].strip()
        for meta in canonical_items.values():
            if int(meta["item_id"]) == item_id and meta["temp_control"] == temp_control:
                meta.setdefault("accepted_names", set()).add(item_name.lower())


def _parse_alias_map(text: str) -> Dict[str, str]:
    rows = _parse_markdown_table_after(text, "| alias_name | canonical_item_id |")
    return {
        row["alias_name"].strip().lower(): row["canonical_item_id"].strip()
        for row in rows
    }


def _parse_legacy_id_map(text: str) -> Dict[int, str]:
    rows = _parse_markdown_table_after(text, "| legacy_item_id | canonical_item_id |")
    return {
        int(row["legacy_item_id"]): row["canonical_item_id"].strip()
        for row in rows
    }


def _parse_special_case_item_ids(text: str) -> List[int]:
    rows = _parse_markdown_table_after(text, "| legacy_item_id | canonical_item_id |")
    return [
        int(row["legacy_item_id"])
        for row in rows
        if row.get("rule", "").strip().upper() == "SPECIAL_CASE"
    ]


def _parse_corridor_sla_tier(text: str) -> Dict[str, str]:
    rows = _parse_markdown_table_after(text, "| corridor_id | corridor_name |")
    if not rows:
        raise ValueError("Policy playbook is missing the corridor table.")
    return {
        row["corridor_id"].strip(): row["default_sla_tier"].strip()
        for row in rows
    }


def _parse_corridor_waypoints(text: str) -> Dict[str, List[Dict[str, Any]]]:
    waypoints: Dict[str, List[Dict[str, Any]]] = {}
    current_corridor: str | None = None
    table_lines: List[str] = []

    for line in text.splitlines():
        bold = re.match(r"\*\*([^*]+)\*\*", line.strip())
        if bold:
            if current_corridor and table_lines:
                waypoints[current_corridor] = _waypoint_rows(table_lines)
            current_corridor = bold.group(1).strip()
            table_lines = []
            continue

        if current_corridor and line.strip().startswith("|"):
            table_lines.append(line.strip())
        elif current_corridor and table_lines:
            waypoints[current_corridor] = _waypoint_rows(table_lines)
            current_corridor = None
            table_lines = []

    if current_corridor and table_lines:
        waypoints[current_corridor] = _waypoint_rows(table_lines)

    return {k: v for k, v in waypoints.items() if v}


def _waypoint_rows(table_lines: List[str]) -> List[Dict[str, Any]]:
    rows = _parse_markdown_table(table_lines)
    return [
        {
            "waypoint_id": row["waypoint_id"],
            "city": row["city"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
        }
        for row in rows
        if {"waypoint_id", "city", "lat", "lon"}.issubset(row)
    ]


def _parse_truck_capacity(text: str) -> int:
    match = re.search(r"Standard truck capacity:\s*\*\*(\d+)\s+volume units\*\*", text)
    if not match:
        raise ValueError("Policy playbook is missing standard truck capacity.")
    return int(match.group(1))


def _parse_packing_buffer(text: str) -> float:
    match = re.search(r"Packing inefficiency buffer:\s*\*\*\+?(\d+(?:\.\d+)?)%\*\*", text)
    if not match:
        raise ValueError("Policy playbook is missing packing inefficiency buffer.")
    return 1.0 + (float(match.group(1)) / 100.0)


def _parse_penalties(text: str) -> Dict[str, int]:
    rows = _parse_markdown_table_after(text, "| Violation Type | Penalty per Unit |")
    penalties: Dict[str, int] = {}
    for row in rows:
        violation = row["Violation Type"].lower()
        value = _first_int(row["Penalty per Unit"])
        if "tier 1" in violation:
            penalties["tier1_sla_violation"] = value
        elif "tier 2" in violation:
            penalties["tier2_sla_violation"] = value
        elif "cold-chain" in violation:
            penalties["cold_chain_violation"] = value
        elif "non-sla" in violation:
            penalties["non_sla_delay"] = value

    required = {"tier1_sla_violation", "tier2_sla_violation", "cold_chain_violation"}
    if not required.issubset(penalties):
        raise ValueError("Policy playbook is missing required penalty values.")
    return penalties


def _parse_weather_thresholds(text: str) -> Dict[str, float]:
    rows = _parse_markdown_table_after(text, "| Condition | Open-Meteo Daily Variable | Threshold |")
    thresholds: Dict[str, float] = {}
    for row in rows:
        condition = row["Condition"].lower()
        value = _first_float(row["Threshold"])
        if "precipitation" in condition:
            thresholds["precipitation_sum_min_mm"] = value
        elif "wind" in condition:
            thresholds["wind_gusts_10m_max_min_kmh"] = value
        elif "freezing" in condition:
            thresholds["temperature_2m_min_max_c"] = value

    required = {
        "precipitation_sum_min_mm",
        "wind_gusts_10m_max_min_kmh",
        "temperature_2m_min_max_c",
    }
    if not required.issubset(thresholds):
        raise ValueError("Policy playbook is missing required weather thresholds.")
    return thresholds


def _parse_travel_buffers(text: str) -> Dict[int, int]:
    rows = _parse_markdown_table_after(text, "| risk_score_0_3 | Travel Time Adjustment |")
    buffers: Dict[int, int] = {}
    for row in rows:
        score = int(row["risk_score_0_3"])
        adjustment = row["Travel Time Adjustment"].lower()
        buffers[score] = 0 if "no buffer" in adjustment else _first_int(adjustment)
    if not buffers:
        raise ValueError("Policy playbook is missing travel buffer policy.")
    return buffers


def _parse_escalation_score(text: str) -> int:
    rows = _parse_markdown_table_after(text, "| risk_score_0_3 | Travel Time Adjustment |")
    for row in rows:
        if "escalation" in row["Travel Time Adjustment"].lower():
            return int(row["risk_score_0_3"])
    return max(_parse_travel_buffers(text))


def _parse_forecast_days(text: str) -> int:
    match = re.search(r"forecast_days\s*=\s*(\d+)", text)
    if not match:
        raise ValueError("Policy playbook is missing forecast_days.")
    return int(match.group(1))


def _first_int(value: str) -> int:
    match = re.search(r"\d+", value)
    if not match:
        raise ValueError(f"No integer found in policy value: {value}")
    return int(match.group(0))


def _first_float(value: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        raise ValueError(f"No number found in policy value: {value}")
    return float(match.group(0))
