import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any


TEXT_FIELDS = ("description", "title", "text", "name")
TASK_FIELDS = ("assignee", "status", "priority", "due_date")
PUNCT_TRANSLATION = str.maketrans(
    {
        "ё": "е",
        "Ё": "Е",
        "—": " ",
        "–": " ",
        "‑": "-",
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "•": " ",
    }
)
MOJIBAKE_REPLACEMENTS = {
    "вЂ”": " ",
    "вЂ“": " ",
    "вЂў": " ",
    "в„–": " ",
    "В·": " ",
    "В«": '"',
    "В»": '"',
    "В": " ",
}
STATUS_ALIASES = {
    "todo": "todo",
    "to do": "todo",
    "open": "todo",
    "new": "todo",
    "planned": "todo",
    "сделать": "todo",
    "новая": "todo",
    "новый": "todo",
    "запланировано": "todo",
    "in_progress": "in_progress",
    "in progress": "in_progress",
    "progress": "in_progress",
    "started": "in_progress",
    "doing": "in_progress",
    "в работе": "in_progress",
    "в процессе": "in_progress",
    "начата": "in_progress",
    "начато": "in_progress",
    "done": "done",
    "completed": "done",
    "complete": "done",
    "closed": "done",
    "готово": "done",
    "выполнено": "done",
    "выполнена": "done",
    "закрыто": "done",
    "закрыта": "done",
}
PRIORITY_ALIASES = {
    "low": "low",
    "низкая": "low",
    "низкий": "low",
    "низкое": "low",
    "минимальный": "low",
    "medium": "medium",
    "normal": "medium",
    "средняя": "medium",
    "средний": "medium",
    "среднее": "medium",
    "high": "high",
    "critical": "high",
    "urgent": "high",
    "высокая": "high",
    "высокий": "high",
    "высокое": "high",
    "критическая": "high",
    "критический": "high",
    "срочно": "high",
}


def _repair_mojibake(value: str) -> str:
    if not any(marker in value for marker in ("Р", "С", "вЂ", "В")):
        return value
    try:
        repaired = value.encode("cp1251").decode("utf-8")
    except UnicodeError:
        return value
    return repaired if repaired.count("\ufffd") <= value.count("\ufffd") else value


def _normalize_text(value: Any) -> str:
    text = _repair_mojibake(str(value or ""))
    for old, new in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = text.translate(PUNCT_TRANSLATION).casefold()
    text = re.sub(r"[_/\\]+", " ", text)
    text = re.sub(r"[^\w\s-]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _normalize(value: Any) -> str:
    return _normalize_text(value)


def _tokens(value: str) -> set[str]:
    return {token for token in value.split() if len(token) > 1}


def _soft_overlap(left_tokens: set[str], right_tokens: set[str]) -> int:
    unmatched_right = set(right_tokens)
    overlap = 0
    for left_token in sorted(left_tokens, key=len, reverse=True):
        if left_token in unmatched_right:
            unmatched_right.remove(left_token)
            overlap += 1
            continue
        if len(left_token) < 5:
            continue
        best = None
        best_score = 0.0
        for right_token in unmatched_right:
            if len(right_token) < 5:
                continue
            score = SequenceMatcher(None, left_token, right_token).ratio()
            if score > best_score:
                best = right_token
                best_score = score
        if best is not None and best_score >= 0.62:
            unmatched_right.remove(best)
            overlap += 1
    return overlap


def _canonical_status(value: Any) -> str:
    text = _normalize_text(value).replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return STATUS_ALIASES.get(text, text.replace(" ", "_"))


def _canonical_priority(value: Any) -> str:
    text = _normalize_text(value)
    return PRIORITY_ALIASES.get(text, text)


def _canonical_date(value: Any) -> str:
    if value in (None, ""):
        return ""
    raw = _repair_mojibake(str(value or "")).strip()
    for old, new in MOJIBAKE_REPLACEMENTS.items():
        raw = raw.replace(old, new)
    text = raw.translate(PUNCT_TRANSLATION).casefold()
    if not text:
        return ""
    for pattern in ("%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            pass
    match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if match:
        year, month, day = match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    match = re.search(r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})\b", text)
    if match:
        day, month, year = match.groups()
        year_int = int(year) + 2000 if len(year) == 2 else int(year)
        return f"{year_int:04d}-{int(month):02d}-{int(day):02d}"
    return text


def _text_value(item: Any) -> str:
    if isinstance(item, dict):
        for field in TEXT_FIELDS:
            value = item.get(field)
            if value:
                return str(value)
        return ""
    return str(item or "")


def _field_value(item: Any, field: str) -> str:
    if not isinstance(item, dict):
        return ""
    value = item.get(field)
    if field == "status":
        return _canonical_status(value)
    if field == "priority":
        return _canonical_priority(value)
    if field == "due_date":
        return _canonical_date(value)
    return _normalize_text(value)


def _similarity(left: Any, right: Any) -> float:
    left_text = _normalize(_text_value(left))
    right_text = _normalize(_text_value(right))
    if not left_text and not right_text:
        return 1.0
    if not left_text or not right_text:
        return 0.0
    sequence_score = SequenceMatcher(None, left_text, right_text).ratio()
    left_tokens = _tokens(left_text)
    right_tokens = _tokens(right_text)
    if not left_tokens or not right_tokens:
        return sequence_score
    overlap = _soft_overlap(left_tokens, right_tokens)
    token_precision = overlap / len(right_tokens)
    token_recall = overlap / len(left_tokens)
    token_f1 = 2 * token_precision * token_recall / (token_precision + token_recall) if token_precision + token_recall else 0.0
    containment = overlap / min(len(left_tokens), len(right_tokens))
    return max(sequence_score, token_f1, containment * 0.92)


def _f1(matches: int, expected_count: int, actual_count: int) -> dict[str, float | int]:
    precision = matches / actual_count if actual_count else (1.0 if expected_count == 0 else 0.0)
    recall = matches / expected_count if expected_count else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "expected": expected_count,
        "actual": actual_count,
        "matched": matches,
        "missed": max(expected_count - matches, 0),
        "extra": max(actual_count - matches, 0),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _match_items(
    expected_items: list[Any],
    actual_items: list[Any],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[Any], list[Any]]:
    candidates: list[tuple[float, int, int]] = []
    for expected_index, expected in enumerate(expected_items):
        for actual_index, actual in enumerate(actual_items):
            score = _similarity(expected, actual)
            if score >= threshold:
                candidates.append((score, expected_index, actual_index))

    candidates.sort(reverse=True)
    used_expected: set[int] = set()
    used_actual: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, expected_index, actual_index in candidates:
        if expected_index in used_expected or actual_index in used_actual:
            continue
        used_expected.add(expected_index)
        used_actual.add(actual_index)
        matches.append(
            {
                "expected_index": expected_index,
                "actual_index": actual_index,
                "description_similarity": round(score, 4),
                "expected": expected_items[expected_index],
                "actual": actual_items[actual_index],
            }
        )

    missed = [item for index, item in enumerate(expected_items) if index not in used_expected]
    extra = [item for index, item in enumerate(actual_items) if index not in used_actual]
    return matches, missed, extra


def _score_collection(
    expected_items: list[Any],
    actual_items: list[Any],
    threshold: float = 0.55,
) -> dict[str, Any]:
    matches, missed, extra = _match_items(expected_items, actual_items, threshold)
    metrics = _f1(len(matches), len(expected_items), len(actual_items))
    return {
        **metrics,
        "matches": matches,
        "missed_items": missed,
        "extra_items": extra,
    }


def _score_tasks(expected_tasks: list[Any], actual_tasks: list[Any]) -> dict[str, Any]:
    matches, missed, extra = _match_items(expected_tasks, actual_tasks, threshold=0.45)
    field_totals = {field: 0 for field in TASK_FIELDS}
    field_matches = {field: 0 for field in TASK_FIELDS}
    enriched_matches: list[dict[str, Any]] = []

    for match in matches:
        expected = match["expected"]
        actual = match["actual"]
        fields: dict[str, bool] = {}
        for field in TASK_FIELDS:
            expected_value = _field_value(expected, field)
            if expected_value:
                field_totals[field] += 1
                fields[field] = expected_value == _field_value(actual, field)
                if fields[field]:
                    field_matches[field] += 1
        enriched_matches.append({**match, "field_matches": fields})

    field_accuracy = {
        field: round(field_matches[field] / field_totals[field], 4) if field_totals[field] else None
        for field in TASK_FIELDS
    }

    return {
        **_f1(len(matches), len(expected_tasks), len(actual_tasks)),
        "field_accuracy": field_accuracy,
        "matches": enriched_matches,
        "missed_items": missed,
        "extra_items": extra,
    }


def _labels_list(labels: dict[str, Any], key: str) -> list[Any]:
    value = labels.get(key, [])
    return value if isinstance(value, list) else []


def _people_items(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [{"name": name} for name in value.keys()]
    if isinstance(value, list):
        return value
    return []


def score_expected_vs_actual(expected_labels: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    task_score = _score_tasks(_labels_list(expected_labels, "tasks"), _labels_list(actual, "tasks"))
    task_update_score = _score_collection(
        _labels_list(expected_labels, "task_updates"),
        _labels_list(actual, "task_updates"),
        threshold=0.45,
    )
    decision_score = _score_collection(
        _labels_list(expected_labels, "decisions"),
        _labels_list(actual, "decisions"),
        threshold=0.5,
    )
    risk_score = _score_collection(
        _labels_list(expected_labels, "risks"),
        _labels_list(actual, "risks"),
        threshold=0.5,
    )
    people_score = _score_collection(
        _labels_list(expected_labels, "people"),
        _people_items(actual.get("people")),
        threshold=0.75,
    )

    category_scores = {
        "tasks": task_score,
        "task_updates": task_update_score,
        "decisions": decision_score,
        "risks": risk_score,
        "people": people_score,
    }
    weights = {
        "tasks": 0.4,
        "task_updates": 0.15,
        "decisions": 0.2,
        "risks": 0.15,
        "people": 0.1,
    }
    overall = sum(category_scores[key]["f1"] * weight for key, weight in weights.items())

    return {
        "schema_version": 1,
        "overall_score": round(overall, 4),
        "categories": category_scores,
    }
