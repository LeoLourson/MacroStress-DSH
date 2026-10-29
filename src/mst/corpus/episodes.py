"""Episode cards: ``<corpus>/episodes/<region>/<ID>.md`` (corpus/README.md, «Эпизод»).

The script writes the header fields and the table «Движения факторов»; the chronology, the
mechanism, the analogy and the sources are written by hand and kept on every rerun.

The window opens 3 months before the first day of ``start`` and closes at the end of the 12th
month after it; ``known_from`` is that last day. For each factor the first series of
``series.toml`` with data on both sides of ``start`` gives:

- base: the last observation before ``start``;
- extreme: the observation from ``start`` to the end of the window farthest from the base;
- change: extreme minus base in p.p., or extreme over base minus one in percent;
- days to extreme: from the first day of ``start`` to the extreme;
- days to recover: from the extreme to the first observation back at the base level or beyond,
  empty if that did not happen inside the window.

Everything is computed the same way for every episode, so cards compare with each other.
"""

import re
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import yaml

from .manifest import Episode
from .series import Cache, Catalog, observations

BEFORE_MONTHS = 3
AFTER_MONTHS = 12
TABLE_MARK = "<!-- Генерируется скриптом из movements. -->"
HAND_SECTIONS = ["Хронология", "Механизм", "Чем полезен как аналогия", "Источники"]
PLACEHOLDERS = {
    "Хронология": "<!-- Не написана. Даты и решения, каждое утверждение со ссылкой на источник "
    "того времени, вышедший в окне эпизода: `[S1/2014 p. 5]` или полная ссылка. -->",
    "Механизм": "<!-- Не написан. Через какие каналы шок дошёл до рынков и балансов, "
    "со ссылками. -->",
    "Чем полезен как аналогия": "<!-- Не написано. Какие условия делают эпизод похожим на "
    "новое событие и где аналогия перестаёт работать. -->",
    "Источники": "<!-- Список источников того времени. -->",
}


def add_months(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


@dataclass(frozen=True)
class Window:
    opens: date
    start: date
    closes: date
    """The last day of the window, also ``known_from``."""


def window(episode: Episode) -> Window:
    start = episode.start_month
    closes = add_months(start, AFTER_MONTHS + 1) - timedelta(days=1)
    return Window(add_months(start, -BEFORE_MONTHS), start, closes)


def resolution(days: list[date]) -> str:
    if len(days) < 2:
        return "day"
    gap = statistics.median((b - a).days for a, b in zip(days, days[1:], strict=False))
    if gap <= 4:
        return "day"
    if gap <= 10:
        return "week"
    if gap <= 45:
        return "month"
    return "quarter"


def change(unit: str, base: float, value: float) -> float:
    return value - base if unit == "pp" else (value / base - 1) * 100


@dataclass(frozen=True)
class Movement:
    factor: str
    series: str
    title: str
    unit: str
    resolution: str
    origin: str
    base: tuple[date, float]
    extreme: tuple[date, float]
    change: float
    days_to_extreme: int
    days_to_recover: int | None

    def header(self) -> dict:
        return {
            "factor": self.factor,
            "series": self.series,
            "unit": "п. п." if self.unit == "pp" else "%",
            "resolution": self.resolution,
            "base": {"date": self.base[0], "value": round(self.base[1], 4)},
            "extreme": {"date": self.extreme[0], "value": round(self.extreme[1], 4)},
            "change": round(self.change, 2),
            "days_to_extreme": self.days_to_extreme,
            "days_to_recover": self.days_to_recover,
        }


def movement(
    factor: str,
    series_id: str,
    catalog: Catalog,
    values: dict[date, float],
    origin: str,
    span: Window,
) -> Movement | None:
    unit = catalog.series[series_id].unit
    before = [day for day in values if span.opens <= day < span.start]
    inside = sorted(day for day in values if span.start <= day <= span.closes)
    if not before or not inside:
        return None
    base_day = max(before)
    base = values[base_day]
    if unit == "pct" and base == 0:
        return None
    extreme_day = max(
        inside, key=lambda day: (abs(change(unit, base, values[day])), -day.toordinal())
    )
    moved = change(unit, base, values[extreme_day])
    recovered = None
    if moved:
        for day in inside:
            if day > extreme_day and change(unit, base, values[day]) * moved <= 0:
                recovered = (day - extreme_day).days
                break
    days = sorted(before + inside)
    return Movement(
        factor,
        series_id,
        catalog.series[series_id].title,
        unit,
        resolution(days),
        origin,
        (base_day, base),
        (extreme_day, values[extreme_day]),
        moved,
        (extreme_day - span.start).days,
        recovered,
    )


@dataclass(frozen=True)
class Measured:
    movements: list[Movement]
    unmeasured: dict[str, str]
    """Factors without a movement, with the reason."""


def measure(episode: Episode, catalog: Catalog, cache: Cache) -> Measured:
    span = window(episode)
    movements, unmeasured = [], {}
    for factor in episode.factors:
        candidates = catalog.candidates(episode.id, episode.country, factor)
        if not candidates:
            unmeasured[factor] = "нет рыночного ряда, описывается в хронологии"
            continue
        found = None
        for series_id in candidates:
            values, origin = observations(catalog, cache, series_id)
            found = movement(factor, series_id, catalog, values, origin, span)
            if found:
                break
        if found:
            movements.append(found)
        else:
            unmeasured[factor] = "нет данных в окне: " + ", ".join(candidates)
    return Measured(movements, unmeasured)


def _number(value: float, unit: str) -> str:
    return f"{value:+.2f}".replace(".", ",") + (" п. п." if unit == "pp" else " %")


def _day(day: date, resolution: str) -> str:
    """A month or quarter value is the period's mean or level: shown as the month, not a day."""
    return f"{day:%m.%Y}" if resolution in ("month", "quarter") else f"{day:%d.%m.%Y}"


def _value(value: float) -> str:
    text = f"{value:.4g}" if abs(value) < 1000 else f"{value:,.0f}".replace(",", " ")
    return text.replace(".", ",")


def table(measured: Measured) -> str:
    lines = [
        "| Фактор | Ряд | База | Экстремум | Изменение | Дней до экстремума "
        "| Дней до восстановления |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in measured.movements:
        note = {"day": "", "week": ", недельный", "month": ", месячный", "quarter": ", квартальный"}
        title = item.title + note[item.resolution]
        lines.append(
            f"| {item.factor} | {title} | {_value(item.base[1])} "
            f"({_day(item.base[0], item.resolution)}) "
            f"| {_value(item.extreme[1])} ({_day(item.extreme[0], item.resolution)}) "
            f"| {_number(item.change, item.unit)} | {item.days_to_extreme} "
            f"| {item.days_to_recover if item.days_to_recover is not None else 'не в окне'} |"
        )
    text = "\n".join(lines) + "\n"
    if measured.unmeasured:
        text += "\nНе измерены рядами:\n\n" + "".join(
            f"- {factor}: {reason}\n" for factor, reason in measured.unmeasured.items()
        )
    if any(item.origin == "store" for item in measured.movements):
        text += (
            "\nЧасть рядов взята из месячного хранилища `series/`: дневные ряды не загружены, "
            "и экстремум внутри месяца не виден.\n"
        )
    return text


def header(episode: Episode, measured: Measured, sources: list) -> str:
    span = window(episode)
    fields = {
        "id": episode.id,
        "kind": "episode",
        "title": episode.title,
        "region": episode.region,
        "country": episode.country,
        "start": episode.start,
        "window": {"before_months": BEFORE_MONTHS, "after_months": AFTER_MONTHS},
        "known_from": span.closes,
        "factors": list(episode.factors),
        "data": list(episode.data),
        "flags": list(episode.flags),
        "sources": sources,
        "movements": [item.header() for item in measured.movements],
        "unmeasured": dict(measured.unmeasured),
    }
    body = yaml.safe_dump(fields, allow_unicode=True, sort_keys=False, width=1000)
    return f"---\n{body}---\n"


def split(text: str) -> tuple[dict, dict[str, str]]:
    """The header fields and the hand-written sections of an existing card."""
    match = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
    if not match:
        return {}, {}
    fields = yaml.safe_load(match.group(1)) or {}
    sections = {}
    for part in re.split(r"(?m)^## ", match.group(2))[1:]:
        title, _, body = part.partition("\n")
        sections[title.strip()] = body.strip()
    return fields, sections


def card(episode: Episode, measured: Measured, existing: str | None) -> str:
    fields, sections = split(existing) if existing else ({}, {})
    sources = fields.get("sources") or []
    parts = [header(episode, measured, sources), f"# {episode.id}. {episode.title}\n"]
    parts.append(
        "## Хронология\n\n" + (sections.get("Хронология") or PLACEHOLDERS["Хронология"]) + "\n"
    )
    parts.append(f"## Движения факторов\n\n{TABLE_MARK}\n\n{table(measured)}")
    for title in HAND_SECTIONS[1:]:
        parts.append(f"## {title}\n\n" + (sections.get(title) or PLACEHOLDERS[title]) + "\n")
    return "\n".join(parts)


def written(text: str) -> bool:
    """Whether the hand-written chronology of a card exists."""
    _, sections = split(text)
    chronology = sections.get("Хронология", "")
    return bool(chronology) and not chronology.startswith("<!-- Не написана")


def write_card(
    episode: Episode, catalog: Catalog, cache: Cache, root: Path
) -> tuple[Path, Measured]:
    path = root / episode.path
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    measured = measure(episode, catalog, cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card(episode, measured, existing), encoding="utf-8")
    return path, measured
