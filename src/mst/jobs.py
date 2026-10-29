"""Tasks: a child process with a log, and the lock of the resource it holds.

A job outlives the tool call that started it: the process runs on its own, its
output goes to ``jobs/<id>/log.txt`` and ``jobs/<id>/job.json`` keeps the argv,
the pid and the times. A lock file per resource keeps two jobs away from the
same store; a lock whose process is gone is stale and is taken over.
"""

import ctypes
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from . import store

RESOURCES = ("series", "models", "corpus", "code")
JOB_ID = re.compile(r"^[0-9A-Za-z-]+$")
STILL_ACTIVE = 259
QUERY_LIMITED_INFORMATION = 0x1000


class ResourceBusy(RuntimeError):
    """The resource is held by a job that still runs."""


def start(ws: store.Workspace, resource: str, argv: Sequence[str]) -> str:
    """Run ``python <argv>`` under the lock of ``resource`` and return the job id.

    ``argv`` is the whole argument list after the interpreter: ``["-m", "mst.data",
    "update"]`` for a module and ``["-c", "print(1)"]`` for a one-liner.
    """
    job_id = _new_id()
    lock = _acquire(ws, resource, job_id)
    directory = ws.jobs / job_id
    directory.mkdir(parents=True)
    record = {
        "id": job_id,
        "resource": resource,
        "argv": list(argv),
        "pid": None,
        "started": _now(),
        "finished": None,
    }
    try:
        with (directory / "log.txt").open("wb") as out:
            record["pid"] = subprocess.Popen(
                [sys.executable, *argv],
                cwd=ws.root,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).pid
    except BaseException:
        _release(ws, resource, job_id)
        raise
    store.write_json(directory / "job.json", record)
    store.write_json(lock, {"job": job_id, "pid": record["pid"], "resource": resource})
    return job_id


def status(ws: store.Workspace, job_id: str) -> dict:
    """The job record and whether its process still runs."""
    record = _record(ws, job_id)
    return record | {"running": _alive(record.get("pid"))}


def log(ws: store.Workspace, job_id: str, tail: int = 50) -> str:
    """The last ``tail`` lines of the job's log; an empty string before the first one."""
    _record(ws, job_id)
    path = ws.jobs / job_id / "log.txt"
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-tail:] if tail > 0 else lines)


def cancel(ws: store.Workspace, job_id: str) -> dict:
    """Stop the job's process tree, mark it finished and release its resource."""
    record = _record(ws, job_id)
    pid = record.get("pid")
    if _alive(pid):
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
        _wait_gone(pid)
    record["finished"] = record.get("finished") or _now()
    record["cancelled"] = True
    store.write_json(ws.jobs / job_id / "job.json", record)
    _release(ws, record.get("resource"), job_id)
    return record | {"running": False}


def state(ws: store.Workspace) -> dict:
    """The workspace state, with every job whose process is gone marked finished."""
    _sweep(ws)
    return store.write_state(ws)


def _acquire(ws: store.Workspace, resource: str, job_id: str) -> Path:
    """Create the lock of ``resource``; take over the lock of a job that is gone."""
    if resource not in RESOURCES:
        raise ValueError(f"unknown resource {resource!r}, try one of {', '.join(RESOURCES)}")
    lock = ws.jobs / f"{resource}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = _exclusive(lock)
    except FileExistsError:
        holder = store.read_json(lock) or {}
        if _alive(holder.get("pid")):
            raise ResourceBusy(
                f"resource {resource!r} is busy with job {holder.get('job')}; "
                "wait for it or cancel it"
            ) from None
        lock.unlink(missing_ok=True)
        descriptor = _exclusive(lock)
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        json.dump({"job": job_id, "pid": os.getpid(), "resource": resource}, file)
    return lock


def _exclusive(lock: Path) -> int:
    return os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)


def _release(ws: store.Workspace, resource: str | None, job_id: str) -> None:
    """Remove the lock of ``resource`` if it belongs to ``job_id``."""
    if not resource:
        return
    lock = ws.jobs / f"{resource}.lock"
    holder = store.read_json(lock) or {}
    if holder.get("job") == job_id:
        lock.unlink(missing_ok=True)


def _sweep(ws: store.Workspace) -> None:
    """Mark every job whose process is gone as finished and free its resource."""
    for path in ws.jobs.glob("*/job.json"):
        record = store.read_json(path) or {}
        if record.get("finished") or _alive(record.get("pid")):
            continue
        record["finished"] = _now()
        store.write_json(path, record)
        _release(ws, record.get("resource"), str(record.get("id")))


def _record(ws: store.Workspace, job_id: str) -> dict:
    if not JOB_ID.fullmatch(job_id or ""):
        raise ValueError(f"not a job id: {job_id!r}")
    record = store.read_json(ws.jobs / job_id / "job.json")
    if record is None:
        raise FileNotFoundError(f"no job {job_id!r} in {ws.jobs}")
    return record


def _alive(pid: int | None) -> bool:
    """Whether a process with this pid still runs."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _wait_gone(pid: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return
        time.sleep(0.05)


def _new_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{os.urandom(2).hex()}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
