from __future__ import annotations

from typing import Any, Dict, List


MAX_RETRY_COUNT = 1


def detect_human_review_triggers(
    special_case_items: Dict[str, Any],
    resource_allocation: Dict[str, Any],
    resource_reserve: Dict[str, Dict[str, int]] | None = None,
) -> Dict[str, Any]:
    triggers: Dict[str, Any] = {
        "special_case_items": special_case_items if special_case_items.get("present") else None,
        "weather_escalation_corridors": [],
        "zero_temp_buffer_days": [],
    }

    for day in ["Day0", "Day1"]:
        resource_reserve = resource_reserve or {}
        day_plan = resource_allocation.get(day, {})
        corridors = day_plan.get("corridors", {})
        for corridor_id, stats in corridors.items():
            if int(stats.get("total_units", 0)) > 0 and int(stats.get("weather_risk_score", 0)) >= 3:
                triggers["weather_escalation_corridors"].append({
                    "day": day,
                    "corridor_id": corridor_id,
                    "weather_risk_score": int(stats.get("weather_risk_score", 0)),
                })

        remaining = day_plan.get("remaining_pool", {})
        temp_allocated = sum(int(stats.get("allocated_temp_trucks", 0)) for stats in corridors.values())
        reserved_temp = int(resource_reserve.get(day, {}).get("truck_temp_controlled", 0))

        if (
            temp_allocated > 0
            and int(remaining.get("truck_temp_controlled", 0)) == 0
            and reserved_temp == 0
        ):
            triggers["zero_temp_buffer_days"].append({
                "day": day,
                "allocated_temp_trucks": temp_allocated,
            })

    triggers["review_required"] = bool(
        triggers["special_case_items"]
        or triggers["weather_escalation_corridors"]
        or triggers["zero_temp_buffer_days"]
    )
    return triggers


def collect_human_review_decision(
    triggers: Dict[str, Any],
    retry_count: int,
    prior_approvals: List[str] | None = None,
) -> Dict[str, Any]:
    prior_approvals = prior_approvals or []
    triggers = _filter_previously_approved_triggers(triggers, prior_approvals)

    if not triggers.get("review_required"):
        return {
            "decision": "approved",
            "reason": "No new human-review business triggers detected.",
            "retry_policy": {},
            "triggers": triggers,
            "approvals": [],
        }

    print("\n=== HUMAN DISPATCH REVIEW REQUIRED ===")
    print("Review only the triggered business risks below.")
    retry_policy: Dict[str, Any] = {}
    approvals: List[str] = []

    if triggers.get("special_case_items"):
        items = triggers["special_case_items"].get("items", [])
        print(f"\nSpecial clinical-trial item(s) detected in planning window: {items}")
        choice = _ask_binary(
            f"Special-case item(s) {items}: 0 = approve dispatch, 1 = hold/quarantine and retry"
        )
        if choice == 1:
            retry_policy["hold_item_ids"] = items
        else:
            approvals.append("special_case_items_approved")

    weather_corridors = triggers.get("weather_escalation_corridors", [])
    hold_corridors = set()
    for item in weather_corridors:
        corridor_id = item["corridor_id"]
        score = item["weather_risk_score"]
        print(f"\nSevere weather detected: {item['day']} {corridor_id}, score={score}")
        choice = _ask_binary(
            f"{corridor_id} weather score 3: 0 = approve with escalation, 1 = hold corridor and retry"
        )
        if choice == 1:
            hold_corridors.add(corridor_id)
        else:
            approvals.append(f"weather_escalation_approved:{item['day']}:{corridor_id}")
    if hold_corridors:
        retry_policy["hold_corridors"] = sorted(hold_corridors)

    zero_temp_days = triggers.get("zero_temp_buffer_days", [])
    reserve_by_day: Dict[str, Dict[str, int]] = {}
    if len(zero_temp_days) > 1:
        affected_days = [str(item["day"]) for item in zero_temp_days]
        print(f"\nZero spare temp-controlled trucks detected on: {', '.join(affected_days)}.")
        print("Risk: no backup capacity for clinical-trial or unplanned cold-chain demand.")
        choice = _ask_zero_temp_grouped(affected_days)
        if choice == 0:
            for day in affected_days:
                approvals.append(f"zero_temp_buffer_approved:{day}")
        elif choice == 1:
            for day in affected_days:
                reserve_by_day.setdefault(day, {})["truck_temp_controlled"] = 1
        else:
            _collect_zero_temp_day_by_day(zero_temp_days, reserve_by_day, approvals)
    else:
        _collect_zero_temp_day_by_day(zero_temp_days, reserve_by_day, approvals)
    if reserve_by_day:
        retry_policy["resource_reserve"] = reserve_by_day

    if retry_policy and retry_count >= MAX_RETRY_COUNT:
        print("\nRetry limit reached. Continuing requires approving the current reviewed plan.")
        return {
            "decision": "approved",
            "reason": "Retry limit reached; proceeding after review.",
            "retry_policy": {},
            "triggers": triggers,
            "approvals": approvals,
        }

    return {
        "decision": "retry" if retry_policy else "approved",
        "reason": "Human selected retry controls." if retry_policy else "Human approved triggered risks.",
        "retry_policy": retry_policy,
        "triggers": triggers,
        "approvals": approvals,
    }


def _filter_previously_approved_triggers(
    triggers: Dict[str, Any],
    prior_approvals: List[str],
) -> Dict[str, Any]:
    filtered = {
        "special_case_items": triggers.get("special_case_items"),
        "weather_escalation_corridors": list(triggers.get("weather_escalation_corridors", [])),
        "zero_temp_buffer_days": list(triggers.get("zero_temp_buffer_days", [])),
    }

    if "special_case_items_approved" in prior_approvals:
        filtered["special_case_items"] = None

    filtered["weather_escalation_corridors"] = [
        item for item in filtered["weather_escalation_corridors"]
        if f"weather_escalation_approved:{item.get('day')}:{item.get('corridor_id')}" not in prior_approvals
    ]

    filtered["zero_temp_buffer_days"] = [
        item for item in filtered["zero_temp_buffer_days"]
        if f"zero_temp_buffer_approved:{item.get('day')}" not in prior_approvals
    ]

    filtered["review_required"] = bool(
        filtered["special_case_items"]
        or filtered["weather_escalation_corridors"]
        or filtered["zero_temp_buffer_days"]
    )
    return filtered


def summarize_human_review(
    review: Dict[str, Any],
    previous_summary: str = "",
) -> str:
    triggers = review.get("triggers", {})
    retry_policy = review.get("retry_policy", {})
    approvals = review.get("approvals", [])

    if not triggers.get("review_required"):
        if previous_summary:
            return (
                previous_summary.rstrip()
                + "\n\nPost-retry review:\n"
                + "- No additional human-review business triggers were detected after retry."
            )
        return (
            "Human review required:\n"
            "- No human-review business triggers were detected.\n\n"
            "Human decisions:\n"
            "- No manager approval or retry controls were required.\n\n"
            "Operational effect:\n"
            "- Dispatch planning continued without human-review changes."
        )

    lines: List[str] = ["Human review required:"]
    special = triggers.get("special_case_items")
    if special:
        lines.append(f"- Special-case item(s) detected: {special.get('items', [])}")

    weather = triggers.get("weather_escalation_corridors", [])
    if weather:
        entries = [
            f"{item.get('day')} {item.get('corridor_id')} score={item.get('weather_risk_score')}"
            for item in weather
        ]
        lines.append(f"- Severe weather escalation detected: {', '.join(entries)}")

    zero_temp_days = triggers.get("zero_temp_buffer_days", [])
    if zero_temp_days:
        days = [str(item.get("day")) for item in zero_temp_days]
        lines.append(f"- Zero spare temp-controlled truck buffer detected: {', '.join(days)}")

    lines.append("")
    lines.append("Human decisions:")
    hold_item_ids = retry_policy.get("hold_item_ids", [])
    hold_corridors = retry_policy.get("hold_corridors", [])
    resource_reserve = retry_policy.get("resource_reserve", {})

    if special:
        if hold_item_ids:
            lines.append(f"- Held/quarantined special-case item(s): {hold_item_ids}")
        elif "special_case_items_approved" in approvals:
            lines.append("- Approved special-case item(s) for dispatch.")

    if weather:
        if hold_corridors:
            lines.append(f"- Held severe-weather corridor(s): {hold_corridors}")
        approved_weather = [
            approval.split(":", 1)[1]
            for approval in approvals
            if approval.startswith("weather_escalation_approved:")
        ]
        if approved_weather:
            lines.append(f"- Approved severe-weather dispatch with escalation: {approved_weather}")

    if zero_temp_days:
        approved_days = [
            approval.split(":", 1)[1]
            for approval in approvals
            if approval.startswith("zero_temp_buffer_approved:")
        ]
        for day in approved_days:
            lines.append(f"- Approved zero temp-controlled truck buffer on {day}.")
        for day in sorted(resource_reserve):
            temp_count = int(resource_reserve[day].get("truck_temp_controlled", 0))
            if temp_count:
                lines.append(f"- Reserved {temp_count} temp-controlled truck(s) on {day}.")

    if retry_policy:
        lines.append("- Retry controls were applied and resource allocation was rerun.")
    else:
        lines.append("- No retry controls were applied.")

    lines.append("")
    lines.append("Operational effect:")
    if hold_item_ids:
        lines.append(
            "- Held special-case item(s) were excluded from dispatch demand and are not counted as undelivered."
        )
        lines.append(
            "- Special-case item holds/quarantines are human-review risk controls, separate from missing unique_item_id data-quality issues."
        )
        lines.append(
            "- Do not describe any special-case item as missing unique_item_id unless it is explicitly listed in the missing-ID data."
        )
    if hold_corridors:
        lines.append(
            "- Held corridor demand was excluded from dispatch demand and should be reported as human-held, not resource shortfall."
        )
    if resource_reserve:
        lines.append(
            "- Reserved temp-controlled trucks reduced available dispatch capacity for the affected day(s)."
        )
        lines.append(
            "- Temp-controlled truck reserves are backup-capacity decisions and are not reserved for held/quarantined items."
        )
    lines.append(
        "- Remaining undelivered units should be explained from actual allocation shortfalls: temp truck, standard truck, or driver."
    )

    return "\n".join(lines)


def _collect_zero_temp_day_by_day(
    zero_temp_days: List[Dict[str, Any]],
    reserve_by_day: Dict[str, Dict[str, int]],
    approvals: List[str],
) -> None:
    for item in zero_temp_days:
        day = item["day"]
        print(f"\nZero spare temp-controlled trucks detected on {day}.")
        choice = _ask_binary(
            f"{day} temp buffer: 0 = approve zero buffer, 1 = reserve 1 temp truck and retry"
        )
        if choice == 1:
            reserve_by_day.setdefault(day, {})["truck_temp_controlled"] = 1
        else:
            approvals.append(f"zero_temp_buffer_approved:{day}")


def _ask_binary(prompt: str) -> int:
    while True:
        raw = input(f"{prompt}\nEnter 0 or 1: ").strip()
        if raw in {"0", "1"}:
            return int(raw)
        print("Please enter 0 or 1.")


def _ask_zero_temp_grouped(affected_days: List[str]) -> int:
    day_text = " and ".join(affected_days)
    prompt = (
        "Choose:\n"
        f"0 = approve zero spare temp-controlled buffer for {day_text}\n"
        f"1 = reserve 1 temp-controlled truck on {day_text} and retry allocation\n"
        "2 = decide day by day"
    )
    while True:
        raw = input(f"{prompt}\nEnter 0, 1, or 2: ").strip()
        if raw in {"0", "1", "2"}:
            return int(raw)
        print("Please enter 0, 1, or 2.")
