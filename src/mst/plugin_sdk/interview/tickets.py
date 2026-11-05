"""The tickets a client returns: one short address of an unanswered part.

A ticket names the model, the date, the input key and the step, and nothing else: the client
copies it back unchanged. It is a hash of the address, so the same question always carries the
same ticket and a retried answer is recognised. All addresses live in one map beside the saved
parts and are pruned after a week, so retries work and the folder does not grow one file per
answered part.
"""

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import RLock

from mst import store

TICKET = re.compile(r"[0-9a-f]{16}")
#: How long a ticket stays usable: long enough to retry, short enough not to pile up.
KEEP = timedelta(days=7)
LOCK = RLock()


def folder(root) -> Path:
    """Where the addresses of an interview folder are written."""
    return Path(root) / ".steps"


def _path(root) -> Path:
    return folder(root) / "tickets.json"


def _pruned(known: dict) -> dict:
    """The addresses younger than ``KEEP``; a broken time is dropped, not repaired."""
    cutoff = datetime.now() - KEEP
    kept = {}
    for ticket, address in known.items():
        try:
            at = datetime.fromisoformat(address.get("at", ""))
        except (AttributeError, ValueError):
            continue
        if at >= cutoff:
            kept[ticket] = address
    return kept


def encode(root, model_id: str, day: date, key: str, step: str, generation: int = 0) -> str:
    """The ticket of one part; its address is kept for the retry of an interrupted answer."""
    address = {"model": model_id, "date": day.isoformat(), "key": key, "step": step}
    if generation:
        address["generation"] = generation
    ticket = store.sha256_of(address)[:16]
    with LOCK:
        known = _pruned(store.read_json(_path(root)) or {})
        known[ticket] = {**address, "at": datetime.now().isoformat(timespec="seconds")}
        store.write_json(_path(root), known)
    return ticket


def decode(root, ticket) -> dict:
    """The address a ticket names; a malformed or unknown ticket is refused with a hint."""
    if not isinstance(ticket, str) or not TICKET.fullmatch(ticket):
        raise ValueError("invalid elicitation ticket: copy answer_call from next unchanged")
    known = store.read_json(_path(root)) or {}
    address = {
        key: value for key, value in (known.get(ticket) or {}).items() if key != "at"
    }
    if not address:
        raise ValueError("unknown elicitation ticket; call next")
    return address
