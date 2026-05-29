from datetime import datetime, timedelta, timezone

APP_TIMEZONE = timezone(timedelta(hours=3), "UTC+03:00")


def app_now() -> datetime:
    return datetime.now(APP_TIMEZONE)
