"""The events of one interview, appended as JSON lines and read as a tail.

The journal answers "who did what and when", not "what was answered": a line names the
action, the input key, the agent, the step, the hash and the number of warnings, and never the
numbers of the answer. A line that cannot be read is skipped, not repaired.
"""

import json
import warnings as diagnostic
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

FILE = "_journal.jsonl"
LOCK = RLock()


def path(folder) -> Path:
    return Path(folder) / FILE


def record(folder, action: str, key=None, agent=None, step=None, digest=None,
           warnings=None, reason=None, codes=None, steps=None) -> dict:
    """Append one event; only the fields the action has are written."""
    event = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "action": action}
    for name, value in (
        ("key", key), ("agent", agent), ("step", step),
        ("hash", digest), ("warnings", warnings), ("codes", codes),
        ("reason", reason), ("steps", steps),
    ):
        if value is not None:
            event[name] = value
    with LOCK:
        Path(folder).mkdir(parents=True, exist_ok=True)
        events = tail(folder, limit=0)
        event["seq"] = max((row.get("seq", 0) for row in events), default=0) + 1
        with path(folder).open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            file.flush()
    return event


def tail(folder, key=None, limit: int = 50) -> list[dict]:
    """The last ``limit`` events of one key, or of the whole interview, oldest first."""
    file = path(folder)
    if not file.is_file():
        return []
    events = []
    for line in file.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            diagnostic.warn(
                f"interview journal {file}: skipping an invalid JSON line", stacklevel=2
            )
            continue
        if not isinstance(event, dict):
            continue
        if key is None or event.get("key") == key:
            events.append(event)
    return events[-limit:]
