"""The check sets of bench/ stay readable: every event of scenario-from-corpus is an id and a
text, nothing that hints at the answer, and an id names one event."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "bench" / "scenario-from-corpus" / "events.toml"


def problems(events: list[dict]) -> list[str]:
    found, seen = [], set()
    for number, event in enumerate(events, 1):
        if set(event) != {"id", "text"}:
            found.append(f"event {number}: fields {sorted(event)}, expected id and text")
        identifier = event.get("id", "")
        if not re.fullmatch(r"E\d{3}", identifier):
            found.append(f"event {number}: id {identifier!r} is not E<three digits>")
        if identifier in seen:
            found.append(f"event {number}: id {identifier} repeats")
        seen.add(identifier)
        if not str(event.get("text", "")).strip():
            found.append(f"event {number}: empty text")
    return found


def test_the_events_are_ids_and_texts_only():
    events = tomllib.loads(EVENTS.read_text(encoding="utf-8"))["events"]
    assert problems(events) == []


def test_a_repeated_id_or_an_answer_field_is_caught():
    events = [
        {"id": "E001", "text": "ЦБ повышает ставку."},
        {"id": "E001", "text": "ЦБ снижает ставку.", "expected": "ru.key_rate"},
    ]
    assert problems(events) == [
        "event 2: fields ['expected', 'id', 'text'], expected id and text",
        "event 2: id E001 repeats",
    ]
