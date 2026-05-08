def extract_entities(text: str) -> dict:
    return {
        "decisions": ["Use FastAPI for backend"],
        "tasks": [
            {"task": "Implement API", "assignee": "Alex"}
        ],
        "people": ["Alex"]
    }