"""The check sets of bench/ stay readable: every event of scenario-from-corpus is an id and a
text, nothing that hints at the answer, and an id names one event; an episode of the model
backtest that names a corpus card starts in the month the card does."""

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "bench" / "scenario-from-corpus" / "events.toml"
PLAN = ROOT / "bench" / "model-backtest" / "episodes.toml"
MANIFEST = ROOT / "corpus" / "manifest.json"


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


def card_problems(episodes: list[dict], cards: dict[str, str]) -> list[str]:
    """Episodes whose card is missing from the corpus, or starts in another month.

    The agent reading the card and the backtest must speak of one window: a mismatch is an
    error, not a warning.
    """
    found = []
    for episode in episodes:
        card = episode.get("card")
        if card is None:
            continue
        if card not in cards:
            found.append(f"{episode['id']}: no card {card} in the corpus manifest")
        elif cards[card] != episode["start"]:
            found.append(
                f"{episode['id']}: starts {episode['start']}, its card {card} {cards[card]}"
            )
    return found


def test_the_backtest_episodes_start_with_their_corpus_cards():
    episodes = tomllib.loads(PLAN.read_text(encoding="utf-8"))["episodes"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cards = {card["id"]: card["start"] for card in manifest["episodes"]}
    assert card_problems(episodes, cards) == []


def test_a_wrong_start_or_a_missing_card_is_caught():
    episodes = [
        {"id": "R1", "start": "2023-06", "card": "R1"},
        {"id": "R2", "start": "2024-04", "card": "R2"},
        {"id": "R3", "start": "2025-01", "card": "R9"},
        {"id": "R4", "start": "2025-02"},
    ]
    cards = {"R1": "2023-06", "R2": "2024-03"}
    assert card_problems(episodes, cards) == [
        "R2: starts 2024-04, its card R2 2024-03",
        "R3: no card R9 in the corpus manifest",
    ]
