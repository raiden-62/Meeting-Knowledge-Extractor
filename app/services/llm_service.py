def format_output(raw_data: dict) -> dict:
    # For now just map fields → later replace with real LLM call
    return {
        "decisions": raw_data.get("decisions", []),
        "tasks": raw_data.get("tasks", []),
        "responsible_people": raw_data.get("people", [])
    }