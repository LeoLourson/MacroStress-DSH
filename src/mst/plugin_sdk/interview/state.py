"""Who owns an input and which answers were accepted: one JSON beside the saved parts.

A claim is advisory to the agents and binding inside the interview: it shows in ``progress``
and refuses a second writer, and it expires so a stopped agent does not block its shock
forever. An acceptance names the hash of the answer it approved, so republishing changes the
answer and cancels the acceptance.
"""

from datetime import datetime, timedelta

from mst import store

LOCK_MARK = "_session.json"


def path(folder):
    return folder / LOCK_MARK


def read(folder) -> dict:
    """The state of the interview folder: claims, acceptances and findings."""
    found = store.read_json(path(folder)) or {}
    found.setdefault("claims", {})
    found.setdefault("accepted", {})
    found.setdefault("findings", {})
    found.setdefault("generations", {})
    return found


def write(folder, state) -> None:
    store.write_json(path(folder), state)


def live(state, key):
    """The claim of ``key`` while it has not expired, otherwise ``None``."""
    found = state["claims"].get(key)
    if not found:
        return None
    try:
        expires = datetime.fromisoformat(found["expires"])
    except (KeyError, TypeError, ValueError):
        return None
    return found if expires > datetime.now() else None


def claim(folder, key, agent, minutes: int = 45) -> dict:
    """Give ``key`` to ``agent``; a live claim of another agent is refused with its owner."""
    state = read(folder)
    current = live(state, key)
    if current and current["agent"] != agent:
        raise ValueError(
            f"{key!r} is claimed by {current['agent']} until {current['expires']}: "
            "release it or leave the claim to its owner"
        )
    now = datetime.now()
    state["claims"][key] = {
        "agent": agent,
        "at": now.isoformat(timespec="seconds"),
        "expires": (now + timedelta(minutes=minutes)).isoformat(timespec="seconds"),
    }
    write(folder, state)
    return state["claims"][key]


def release(folder, key, agent) -> dict:
    """Drop ``key`` from the claims of ``agent``; another agent's claim is left alone."""
    state = read(folder)
    current = state["claims"].get(key)
    if current is None:
        return {}
    if current["agent"] != agent:
        raise ValueError(
            f"{key!r} is claimed by {current['agent']}: only its owner releases it"
        )
    del state["claims"][key]
    write(folder, state)
    return current


def accept(folder, key, agent, reason, digest) -> dict:
    """Record that ``agent`` reviewed the answer with ``digest`` and accepted its warnings."""
    state = read(folder)
    state["accepted"][key] = {
        "hash": digest,
        "agent": agent,
        "at": datetime.now().isoformat(timespec="seconds"),
        "reason": reason,
    }
    write(folder, state)
    return state["accepted"][key]


def findings(folder, key, digest, rows) -> None:
    """Keep the summary of a publication's findings: the hash, the count and the codes."""
    state = read(folder)
    if rows:
        state["findings"][key] = {
            "hash": digest,
            "count": len(rows),
            "codes": sorted({row["code"] for row in rows}),
        }
    else:
        state["findings"].pop(key, None)
    write(folder, state)
