from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import math
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# Item Master — Appendix A
# ---------------------------------------------------------------------------
CANONICAL_ITEMS: Dict[str, Dict[str, Any]] = {
    "RMD-100":    {"item_id": 10021, "item_name": "Remdesivir 100mg",               "temp_control": "Cold (2-8C)",              "product_class": "Antiviral"},
    "RMD-200":    {"item_id": 10021, "item_name": "Remdesivir 200mg",               "temp_control": "Cold (2-8C)",              "product_class": "Antiviral"},
    "INS-LIS":    {"item_id": 10022, "item_name": "Insulin Lispro",                 "temp_control": "Cold (2-8C)",              "product_class": "Endocrine"},
    "PMB-KEY":    {"item_id": 10035, "item_name": "Pembrolizumab",                  "temp_control": "Cold (2-8C)",              "product_class": "Oncology Biologic"},
    "EPI-AI":     {"item_id": 10040, "item_name": "Epinephrine Auto-Injector",      "temp_control": "Room Temp (20-25C)",       "product_class": "Emergency"},
    "HEP-SOD":    {"item_id": 10050, "item_name": "Heparin Sodium",                 "temp_control": "Room Temp (20-25C)",       "product_class": "Anticoagulant"},
    "MOR-SUL":    {"item_id": 10060, "item_name": "Morphine Sulfate",               "temp_control": "Controlled Storage",       "product_class": "Controlled"},
    "ALB-INH":    {"item_id": 10070, "item_name": "Albuterol Inhaler",              "temp_control": "Room Temp (20-25C)",       "product_class": "Respiratory"},
    "EXP-ONC-CT": {"item_id": 99999, "item_name": "Experimental Oncology Drug",    "temp_control": "Strict Cold Chain (-20C)", "product_class": "Clinical Trial"},
    "LEV-INH":    {"item_id": 10071, "item_name": "Levalbuterol Inhaler",           "temp_control": "Room Temp (20-25C)",       "product_class": "Respiratory"},
    "INS-ASP":    {"item_id": 10023, "item_name": "Insulin Aspart",                 "temp_control": "Cold (2-8C)",              "product_class": "Endocrine"},
}

NAME_ALIAS: Dict[str, str] = {
    "remdesivir 100 mg":          "RMD-100",
    "remdesivir 200 mg":          "RMD-200",
    "pembrolizumab (keytruda)":   "PMB-KEY",
    "epipen auto injector":       "EPI-AI",
    "heparin na":                 "HEP-SOD",
    "morphine sulphate":          "MOR-SUL",
    "albuterol inhaler 90mcg":    "ALB-INH",
}

LEGACY_ID_MAP: Dict[int, str] = {
    10020: "RMD-100",
    20021: "RMD-200",
    1070:  "ALB-INH",
    99999: "EXP-ONC-CT",
}

_ITEM_ID_TO_CANONICAL: Dict[int, List[str]] = {}
for cid, meta in CANONICAL_ITEMS.items():
    _ITEM_ID_TO_CANONICAL.setdefault(meta["item_id"], []).append(cid)

CORRIDOR_SLA_TIER: Dict[str, str] = {
    "C1_I95_NJ_BOS": "Tier 1",
    "C2_NJ_PHL":     "Tier 2",
}

COLD_KEYWORDS = {"cold", "strict cold chain"}
TRUCK_CAPACITY = 10
PACKING_BUFFER = 1.10


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
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


def _resolve_item(item_id: Any, item_name: str) -> Tuple[Optional[str], str, List[str]]:
    dq_flags: List[str] = []
    name_lower = str(item_name).strip().lower()

    try:
        iid = int(item_id)
    except (ValueError, TypeError):
        iid = None

    if iid is not None and iid in LEGACY_ID_MAP and iid not in _ITEM_ID_TO_CANONICAL:
        return LEGACY_ID_MAP[iid], "LEGACY_ID_MAP", dq_flags

    if iid is not None and iid in _ITEM_ID_TO_CANONICAL:
        candidates = _ITEM_ID_TO_CANONICAL[iid]
        for cid in candidates:
            if CANONICAL_ITEMS[cid]["item_name"].lower() == name_lower:
                return cid, "EXACT_MATCH", dq_flags
        if name_lower in NAME_ALIAS:
            alias_cid = NAME_ALIAS[name_lower]
            if alias_cid in candidates:
                return alias_cid, "ALIAS_MATCH", dq_flags
        if len(candidates) > 1:
            dq_flags.append("DQ-03")
            return candidates[0], "ALIAS_MATCH", dq_flags
        return candidates[0], "ALIAS_MATCH", dq_flags

    if name_lower in NAME_ALIAS:
        return NAME_ALIAS[name_lower], "ALIAS_MATCH", dq_flags

    dq_flags.append("DQ-02")
    return None, "UNRESOLVED", dq_flags


def _required_trucks(n_units: int) -> int:
    if n_units == 0:
        return 0
    return math.ceil(n_units * PACKING_BUFFER / TRUCK_CAPACITY)


def analyze_csv(csv_path: str) -> MultiCorridorResult:
    df = pd.read_csv(csv_path)
    rows_original = len(df)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(how="all").copy()

    for c in df.columns:
        if "date" in c.lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")

    dq_counts = {"DQ-01": 0, "DQ-02": 0, "DQ-03": 0, "DQ-04": 0}

    missing_uid = df["unique_item_id"].isna() | (df["unique_item_id"].astype(str).str.strip() == "")
    df.loc[missing_uid, "_dq_flag"] = "DQ-01"
    dq_counts["DQ-01"] = int(missing_uid.sum())

    valid_uid = df.loc[~missing_uid, "unique_item_id"]
    dup_mask = df["unique_item_id"].isin(valid_uid[valid_uid.duplicated()])
    df.loc[dup_mask & ~missing_uid, "_dq_flag"] = df.loc[dup_mask & ~missing_uid, "_dq_flag"].fillna("DQ-04")
    dq_counts["DQ-04"] = int((dup_mask & ~missing_uid).sum())

    resolved_ids, confidences, dq_extra = [], [], []
    for _, row in df.iterrows():
        cid, conf, flags = _resolve_item(row.get("item_id"), row.get("item_name", ""))
        resolved_ids.append(cid)
        confidences.append(conf)
        dq_extra.append(flags)

    df["canonical_item_id"] = resolved_ids
    df["reconcile_confidence"] = confidences

    for i, flags in enumerate(dq_extra):
        for f in flags:
            dq_counts[f] = dq_counts.get(f, 0) + 1

    df["temp_control"] = df["canonical_item_id"].map(
        lambda cid: CANONICAL_ITEMS.get(cid, {}).get("temp_control", "Unknown") if cid else "Unknown"
    )
    df["needs_temp_truck"] = df["temp_control"].str.lower().apply(
        lambda t: any(k in t for k in COLD_KEYWORDS)
    )
    df["sla_tier"] = df["corridor_id"].map(CORRIDOR_SLA_TIER).fillna("Unknown")

    if "is_planning_window" in df.columns:
        planning_mask = df["is_planning_window"].astype(str).str.strip().isin(["1", "1.0", "True"])
    else:
        planning_mask = df["planning_day"].isin(["Day0", "Day1"])

    exclude_mask = planning_mask & (df.get("_dq_flag", pd.Series("", index=df.index)) == "DQ-01")
    excluded_df = df[exclude_mask].copy()
    planning_df = df[planning_mask & ~exclude_mask].copy()
    history_df  = df[~planning_mask].copy()

    corridor_day_summary: Dict[str, Any] = {}

    for corridor_id in df["corridor_id"].dropna().unique():
        corridor_day_summary[corridor_id] = {}
        sla_tier = CORRIDOR_SLA_TIER.get(corridor_id, "Unknown")

        for day in ["Day0", "Day1"]:
            mask = (planning_df["corridor_id"] == corridor_id) & (planning_df["planning_day"] == day)
            sub = planning_df[mask]

            n_total  = len(sub)
            n_temp   = int(sub["needs_temp_truck"].sum())
            n_std    = n_total - n_temp
            req_temp = _required_trucks(n_temp)
            req_std  = _required_trucks(n_std)
            req_drv  = req_temp + req_std

            corridor_day_summary[corridor_id][day] = {
                "sla_tier":              sla_tier,
                "total_valid_units":     n_total,
                "temp_controlled_units": n_temp,
                "standard_units":        n_std,
                "required_temp_trucks":  req_temp,
                "required_std_trucks":   req_std,
                "required_drivers":      req_drv,
                "hospitals":             sorted(sub["dispatch_location"].dropna().unique().tolist()),
            }

    total_planning_units = len(planning_df)
    kpis: Dict[str, Any] = {
        "total_planning_units":  total_planning_units,
        "total_excluded_dq01":   dq_counts["DQ-01"],
        "total_flagged_dq02":    dq_counts["DQ-02"],
        "total_flagged_dq03":    dq_counts.get("DQ-03", 0),
        "total_flagged_dq04":    dq_counts["DQ-04"],
        "exclusion_rate_pct":    round(len(excluded_df) / max(rows_original, 1) * 100, 2),
        "corridor_day_summary":  corridor_day_summary,
    }
    for corridor_id, days in corridor_day_summary.items():
        total_units = sum(d["total_valid_units"] for d in days.values())
        kpis[f"{corridor_id}_total_units_48h"] = total_units
        kpis[f"{corridor_id}_sla_tier"]        = CORRIDOR_SLA_TIER.get(corridor_id, "Unknown")

    anomalies = pd.DataFrame()
    num_cols = planning_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2 and len(planning_df) >= 20:
        X = planning_df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
        preds  = model.fit_predict(X)
        scores = model.decision_function(X)
        tmp = planning_df.copy()
        tmp["is_anomaly"]    = (preds == -1)
        tmp["anomaly_score"] = scores
        anomalies = tmp[tmp["is_anomaly"]].sort_values("anomaly_score").head(25)

    summary = {
        "rows_original":        rows_original,
        "rows_planning_window": len(planning_df),
        "rows_history":         len(history_df),
        "rows_excluded_dq01":   len(excluded_df),
        "dq_report":            dq_counts,
        "corridors_found":      list(corridor_day_summary.keys()),
        "columns":              list(df.columns),
    }

    return MultiCorridorResult(
        rows_original=rows_original,
        dq_report=dq_counts,
        excluded_df=excluded_df,
        planning_df=planning_df,
        history_df=history_df,
        corridor_day_summary=corridor_day_summary,
        kpis=kpis,
        anomalies=anomalies,
        summary=summary,
    )
