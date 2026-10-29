"""Daily observations to monthly values."""

from collections import defaultdict
from datetime import date, timedelta
from statistics import median


def _months(daily: dict[date, float]) -> list[tuple[date, list[float]]]:
    months = defaultdict(list)
    for day, value in daily.items():
        months[day.replace(day=1)].append(value)
    return sorted(months.items())


def monthly_means(daily: dict[date, float]) -> dict[date, float]:
    """The mean of each month's observed days, keyed by the first day of the month."""
    return {month: sum(values) / len(values) for month, values in _months(daily)}


def monthly_medians(daily: dict[date, float]) -> dict[date, float]:
    """The median of each month's observed days, keyed by the first day of the month."""
    return {month: median(values) for month, values in _months(daily)}


def step_means(steps: dict[date, float], start: date) -> dict[date, float]:
    """Monthly means over calendar days of a value in force from each step date until the next.

    Months run from ``start`` to the month of the last step; days before the first step are
    left out.
    """
    return monthly_means(step_days(steps, start))


def step_days(steps: dict[date, float], start: date) -> dict[date, float]:
    """The value in force on each calendar day from ``start`` to the end of the last step's month.

    Days before the first step are left out.
    """
    days = sorted(steps)
    values, current, i = {}, None, 0
    day = max(start, days[0])
    end = days[-1].replace(day=28) + timedelta(days=4)
    end = end - timedelta(days=end.day)
    while day <= end:
        while i < len(days) and days[i] <= day:
            current = steps[days[i]]
            i += 1
        if current is not None:
            values[day] = current
        day += timedelta(days=1)
    return values
