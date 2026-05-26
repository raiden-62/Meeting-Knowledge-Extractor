import re
from datetime import date, datetime
from typing import Any


DATE_PATTERNS = (
    r"\b([0-2]?\d|3[01])\.(0?[1-9]|1[0-2])\.(20\d{2})\b",
    r"\b([0-2]?\d|3[01])/(0?[1-9]|1[0-2])/(20\d{2})\b",
    r"\b(20\d{2})-(0[1-9]|1[0-2])-([0-2]\d|3[01])\b",
)


def parse_meeting_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    text = str(value).strip()
    if not text:
        return None

    for date_format in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            pass

    return extract_meeting_date(text)


def extract_meeting_date(content: str) -> date | None:
    head = (content or "")[:1000]
    for index, pattern in enumerate(DATE_PATTERNS):
        match = re.search(pattern, head)
        if not match:
            continue
        try:
            if index == 2:
                year, month, day = match.groups()
            else:
                day, month, year = match.groups()
            return date(int(year), int(month), int(day))
        except ValueError:
            return None
    return None


def resolve_meeting_date(value: Any, content: str) -> date:
    return parse_meeting_date(value) or extract_meeting_date(content) or date.today()


def format_meeting_date(value: date) -> str:
    return value.strftime("%d.%m.%Y")


def build_dated_transcript(content: str, meeting_date: date | None) -> str:
    if not meeting_date:
        return content
    return f"Дата встречи: {format_meeting_date(meeting_date)}\n\n{content}"
