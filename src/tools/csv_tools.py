from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
import math

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

from tools.policy_tools import DispatchPolicy, load_dispatch_policy


@dataclass
class MultiCorridorResult:
    rows_original: int
    dq_report: Dict[str, Any]
    excluded_df: pd.DataFrame
    planning_df: pd.DataFrame
    history_df: pd.DataFrame
    corridor_day_summary: Dict[str, Any]
    kpis: Dict[str, Any]
    anomalies: pd.DataFrame
    summary: Dict[str, Any]

    # Harish compatibility / generic analyzer fields
    cleaned_shape: Tuple[int, int] = (0, 0)
    numeric_cols: List[str] | None = None


CsvAnalysisResult = MultiCorridorResult


def _item_id_to_canonical(policy: DispatchPolicy) -> Dict[int, List[str]]:
    mapping: Dict[int, List[str]] = {}
    for canonical_id, meta in policy.canonical_items.items():
        mapping.setdefault(int(meta["item_id"]), []).append(canonical_id)
    return mapping


def _resolve_item(
    item_id: Any,
    item_name: str,
    policy: DispatchPolicy,
) -> Tuple[Optional[str], str, List[str]]:
    dq_flags: List[str] = []
    name_lower = str(item_name).strip().lower()
    item_map = _item_id_to_canonical(policy)

    try:
        iid = int(item_id)
    except (TypeError, ValueError):
        iid = None

    if iid is not None and iid in policy.legacy_id_map and iid not in item_map:
        return policy.legacy_id_map[iid], "LEGACY_ID_MAP", dq_flags

    if iid is not None and iid in item_map:
        candidates = item_map[iid]
        for canonical_id in candidates:
            accepted_names = policy.canonical_items[canonical_id].get("accepted_names", set())
            if name_lower in accepted_names:
                return canonical_id, "EXACT_MATCH", dq_flags

        if name_lower in policy.name_alias and policy.name_alias[name_lower] in candidates:
            return policy.name_alias[name_lower], "ALIAS_MATCH", dq_flags

        if len(candidates) > 1:
            dq_flags.append("DQ-03")

        return candidates[0], "ITEM_ID_MATCH_NAME_VARIANT", dq_flags

    if name_lower in policy.name_alias:
        return policy.name_alias[name_lower], "ALIAS_MATCH", dq_flags

    dq_flags.append("DQ-02")
    return None, "UNRESOLVED", dq_flags


def _required_trucks(unit_count: int, policy: DispatchPolicy) -> int:
    if unit_count <= 0:
        return 0
    return math.ceil((unit_count * policy.packing_buffer) / policy.truck_capacity)


def analyze_csv(
    csv_path: str,
    policy_path: str | None = None,
    df_override: pd.DataFrame | None = None,
) -> MultiCorridorResult:
    policy = load_dispatch_policy(policy_path)

    df = df_override.copy() if df_override is not None else pd.read_csv(csv_path)
    original_shape = df.shape
    rows_original = int(original_shape[0])

    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all").copy()

    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")

    dq_counts = {"DQ-01": 0, "DQ-02": 0, "DQ-03": 0, "DQ-04": 0}
    corridor_day_summary: Dict[str, Any] = {}

    missing_uid = df["unique_item_id"].isna() | (
        df["unique_item_id"].astype(str).str.strip() == ""
    )
    df.loc[missing_uid, "_dq_flag"] = "DQ-01"
    dq_counts["DQ-01"] = int(missing_uid.sum())

    valid_uid = df.loc[~missing_uid, "unique_item_id"]
    duplicate_uid = df["unique_item_id"].isin(valid_uid[valid_uid.duplicated()])
    df.loc[duplicate_uid & ~missing_uid, "_dq_flag"] = (
        df.loc[duplicate_uid & ~missing_uid, "_dq_flag"].fillna("DQ-04")
    )
    dq_counts["DQ-04"] = int((duplicate_uid & ~missing_uid).sum())

    resolved_ids, confidences, dq_extra = [], [], []
    for _, row in df.iterrows():
        cid, confidence, flags = _resolve_item(
            row.get("item_id"),
            row.get("item_name", ""),
            policy,
        )
        resolved_ids.append(cid)
        confidences.append(confidence)
        dq_extra.append(flags)

    df["canonical_item_id"] = resolved_ids
    df["reconcile_confidence"] = confidences
    df["reconcile_reason"] = confidences

    for flags in dq_extra:
        for flag in flags:
            dq_counts[flag] = dq_counts.get(flag, 0) + 1

    df["temp_control"] = df["canonical_item_id"].map(
        lambda cid: policy.canonical_items.get(cid, {}).get("temp_control", "Unknown")
        if cid
        else "Unknown"
    )
    df["needs_temp_truck"] = df["temp_control"].str.lower().str.contains("cold", na=False)
    df["sla_tier"] = df["corridor_id"].map(policy.corridor_sla_tier).fillna("Unknown")

    if "is_planning_window" in df.columns:
        planning_mask = df["is_planning_window"].astype(str).str.strip().isin(
            ["1", "1.0", "True", "true"]
        )
    else:
        planning_mask = df["planning_day"].isin(["Day0", "Day1"])

    exclude_mask = planning_mask & missing_uid
    excluded_df = df[exclude_mask].copy()
    planning_df = df[planning_mask & ~exclude_mask].copy()
    history_df = df[~planning_mask].copy()

    for corridor_id in sorted(planning_df["corridor_id"].dropna().unique()):
        corridor_day_summary[corridor_id] = {}
        for day in ["Day0", "Day1"]:
            sub = planning_df[
                (planning_df["corridor_id"] == corridor_id)
                & (planning_df["planning_day"] == day)
            ]

            temp_units = int(sub["needs_temp_truck"].sum())
            total_units = int(len(sub))
            standard_units = total_units - temp_units
            required_temp = _required_trucks(temp_units, policy)
            required_standard = _required_trucks(standard_units, policy)

            corridor_day_summary[corridor_id][day] = {
                "sla_tier": policy.corridor_sla_tier.get(corridor_id, "Unknown"),
                "total_valid_units": total_units,
                "temp_controlled_units": temp_units,
                "standard_units": standard_units,
                "required_temp_trucks": required_temp,
                "required_std_trucks": required_standard,
                "required_drivers": required_temp + required_standard,
                "hospitals": sorted(sub["dispatch_location"].dropna().unique().tolist())
                if "dispatch_location" in sub.columns
                else [],
            }

    dq_report = {
        "missing_unique_item_id_dq01": int(missing_uid.sum()),
        "duplicate_unique_item_id_dq04": int((duplicate_uid & ~missing_uid).sum()),
        "unresolved_item_dq02": int(df["canonical_item_id"].isna().sum()),
        "name_or_legacy_reconciled": int(
            df["reconcile_reason"].isin(
                ["ALIAS_MATCH", "LEGACY_ID_MAP", "ITEM_ID_MATCH_NAME_VARIANT"]
            ).sum()
        ),
    }

    kpis: Dict[str, Any] = {
        # Pranit-compatible KPI names
        "total_planning_units": int(len(planning_df)),
        "total_excluded_dq01": int(len(excluded_df)),
        "total_flagged_dq02": int(dq_counts.get("DQ-02", 0)),
        "total_flagged_dq03": int(dq_counts.get("DQ-03", 0)),
        "total_flagged_dq04": int(dq_counts.get("DQ-04", 0)),
        "exclusion_rate_pct": round(len(excluded_df) / max(rows_original, 1) * 100, 2),
        "corridor_day_summary": corridor_day_summary,

        # Harish/non-hardcoding KPI names
        "rows_history": int(len(history_df)),
        "rows_planning_window_valid": int(len(planning_df)),
        "rows_excluded_missing_unique_item_id": int(len(excluded_df)),
        "dq_report": dq_report,
    }

    for corridor_id, days in corridor_day_summary.items():
        total_units = sum(d["total_valid_units"] for d in days.values())
        kpis[f"{corridor_id}_total_units_48h"] = total_units
        kpis[f"{corridor_id}_sla_tier"] = policy.corridor_sla_tier.get(corridor_id, "Unknown")

    numeric_cols = planning_df.select_dtypes(include=[np.number]).columns.tolist()
    anomalies = pd.DataFrame()
    if len(numeric_cols) >= 2 and len(planning_df) >= 20:
        X = planning_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
        preds = model.fit_predict(X)
        scores = model.decision_function(X)

        tmp = planning_df.copy()
        tmp["is_anomaly"] = preds == -1
        tmp["anomaly_score"] = scores
        anomalies = tmp[tmp["is_anomaly"]].sort_values("anomaly_score").head(25)

    summary = {
        "rows_original": rows_original,
        "cols_original": int(original_shape[1]),
        "rows_after_drop_empty": int(df.shape[0]),
        "rows_planning_window": int(len(planning_df)),
        "rows_planning_window_valid": int(len(planning_df)),
        "rows_history": int(len(history_df)),
        "rows_excluded_dq01": int(len(excluded_df)),
        "rows_excluded_missing_unique_item_id": int(len(excluded_df)),
        "dq_report": dq_report,
        "corridors_found": sorted(df["corridor_id"].dropna().unique().tolist()),
        "columns": list(df.columns),
        "missingness_top": df.isna().mean().sort_values(ascending=False).head(10).to_dict(),
        "column_dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }

    return MultiCorridorResult(
        rows_original=rows_original,
        dq_report=dq_report,
        excluded_df=excluded_df,
        planning_df=planning_df,
        history_df=history_df,
        corridor_day_summary=corridor_day_summary,
        kpis=kpis,
        anomalies=anomalies,
        summary=summary,
        cleaned_shape=df.shape,
        numeric_cols=numeric_cols,
    )


def analyze_csv_with_overrides(
    csv_path: str,
    policy_path: str | None = None,
    hold_item_ids: List[int] | None = None,
    hold_corridors: List[str] | None = None,
) -> MultiCorridorResult:
    hold_item_ids = hold_item_ids or []
    hold_corridors = hold_corridors or []

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all").copy()

    planning_mask = df["is_planning_window"].astype(str).str.strip().isin(
        ["1", "1.0", "True", "true"]
    )
    retry_hold_mask = planning_mask & (
        df["item_id"].isin(hold_item_ids) | df["corridor_id"].isin(hold_corridors)
    )

    df.loc[retry_hold_mask, "unique_item_id"] = pd.NA

    result = analyze_csv(csv_path, policy_path=policy_path, df_override=df)
    if retry_hold_mask.any():
        result.kpis["human_retry_holds"] = {
            "hold_item_ids": hold_item_ids,
            "hold_corridors": hold_corridors,
            "rows_held_from_planning": int(retry_hold_mask.sum()),
        }
        result.summary["human_retry_holds"] = result.kpis["human_retry_holds"]

    return result


def detect_planning_special_case_items(
    csv_path: str,
    policy_path: str | None = None,
    hold_item_ids: List[int] | None = None,
    hold_corridors: List[str] | None = None,
) -> Dict[str, Any]:
    policy = load_dispatch_policy(policy_path)
    hold_item_ids = hold_item_ids or []
    hold_corridors = hold_corridors or []

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = {"planning_day", "is_planning_window", "item_id", "unique_item_id", "corridor_id"}
    if not required.issubset(df.columns):
        return {"present": False, "items": [], "rows": []}

    planning_mask = df["is_planning_window"].astype(str).str.strip().isin(
        ["1", "1.0", "True", "true"]
    )
    held_mask = df["item_id"].isin(hold_item_ids) | df["corridor_id"].isin(hold_corridors)
    special_mask = planning_mask & ~held_mask & df["item_id"].isin(policy.special_case_item_ids)

    rows = df.loc[
        special_mask,
        ["planning_day", "corridor_id", "item_id", "item_name", "unique_item_id"],
    ]

    return {
        "present": bool(special_mask.any()),
        "items": sorted({int(x) for x in df.loc[special_mask, "item_id"].dropna().tolist()}),
        "rows": rows.to_dict(orient="records"),
    }
