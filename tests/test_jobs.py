"""A job is a child process with a log, and its resource is locked while it runs."""

import time

import pytest

from mst import jobs, store

SLEEP = ["-c", "import time; time.sleep(30)"]


def wait(ws, job_id, timeout=30.0):
    """Wait until the job's process is gone."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not jobs.status(ws, job_id)["running"]:
            return
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout} s")


def test_a_job_prints_into_its_log(tmp_path):
    ws = store.Workspace(tmp_path)
    job_id = jobs.start(ws, "series", ["-c", "print(1)"])
    wait(ws, job_id)
    assert jobs.log(ws, job_id).strip() == "1"
    record = jobs.status(ws, job_id)
    assert record["argv"] == ["-c", "print(1)"]
    assert record["resource"] == "series"


def test_a_busy_resource_is_refused_with_the_name_of_the_first_job(tmp_path):
    ws = store.Workspace(tmp_path)
    first = jobs.start(ws, "models", SLEEP)
    try:
        assert jobs.state(ws)["jobs"] == 1
        with pytest.raises(jobs.ResourceBusy) as error:
            jobs.start(ws, "models", ["-c", "print(2)"])
        assert first in str(error.value)
    finally:
        jobs.cancel(ws, first)
    assert jobs.state(ws)["jobs"] == 0


def test_a_finished_job_frees_its_resource(tmp_path):
    ws = store.Workspace(tmp_path)
    first = jobs.start(ws, "corpus", ["-c", "print(1)"])
    wait(ws, first)
    second = jobs.start(ws, "corpus", ["-c", "print(2)"])
    wait(ws, second)
    assert jobs.log(ws, second).strip() == "2"


def test_cancel_stops_the_process_and_releases_the_lock(tmp_path):
    ws = store.Workspace(tmp_path)
    job_id = jobs.start(ws, "code", SLEEP)
    cancelled = jobs.cancel(ws, job_id)
    assert cancelled["cancelled"] is True
    assert jobs.status(ws, job_id)["running"] is False
    assert not (ws.jobs / "code.lock").exists()
    assert jobs.state(ws)["jobs"] == 0


def test_an_unknown_resource_and_a_foreign_job_id_are_refused(tmp_path):
    ws = store.Workspace(tmp_path)
    with pytest.raises(ValueError, match="unknown resource"):
        jobs.start(ws, "web", ["-c", "print(1)"])
    with pytest.raises(ValueError, match="not a job id"):
        jobs.status(ws, "../../etc")
