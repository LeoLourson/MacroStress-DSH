"""Execution choices belong to a run, not to its reusable scenario."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from mst import runs, store
from mst.runs import RunRequest
from mst.scenario.spec import Spec

MACRO = "a" * 64
SATELLITES = "b" * 64
POSITION = {"id": "cash", "type": "common_stock", "notional": 1.0, "codes": {"secid": "SBER"}}


def request(**changes):
    return {
        "scenario": "policy-decision",
        "draws": 8,
        "seed": 20261112,
        "positions": [POSITION],
        **changes,
    }


def test_run_request_selects_versions_horizon_and_inner_independently_of_scenario():
    built = RunRequest.model_validate(
        request(macro=MACRO, satellites=SATELLITES, horizon=18, inner=4)
    )
    assert built.macro == MACRO
    assert built.satellites == SATELLITES
    assert built.horizon == 18
    assert built.inner == 4


def test_run_request_defaults_to_twelve_months_and_one_inner_path():
    built = RunRequest.model_validate(request())
    assert built.macro is None and built.satellites is None
    assert built.horizon == 12
    assert built.inner == 1


def test_begin_pins_default_versions_before_background_execution(tmp_path, monkeypatch):
    ws = store.Workspace(tmp_path)
    monkeypatch.setattr(
        store, "state", lambda workspace: {"macro": MACRO, "satellites": SATELLITES}
    )
    checked = runs.begin(ws, "20261112T125232-abcd", request())
    assert checked["macro"] == MACRO and checked["satellites"] == SATELLITES
    assert store.read_json(ws.runs / "20261112T125232-abcd" / "request.json") == checked


@pytest.mark.parametrize("field", ["macro", "satellites"])
def test_run_request_rejects_invalid_version_hash(field):
    with pytest.raises(ValidationError, match=field):
        RunRequest.model_validate(request(**{field: "not-a-version"}))


@pytest.mark.parametrize("changes", [{"horizon": 0}, {"horizon": 361}, {"inner": 0}, {"inner": 65}])
def test_run_request_bounds_execution_counts(changes):
    with pytest.raises(ValidationError):
        RunRequest.model_validate(request(**changes))


def test_value_changes_keep_one_row_per_inner_path(tmp_path):
    """``draws`` counts the groups of the scenario, the run has ``draws * inner`` paths."""
    rows, inner = 16, 2
    first, second = date(2026, 10, 1), date(2026, 11, 1)
    paths = SimpleNamespace(
        months=[first, second],
        ids=["ru.key_rate"],
        macro=np.zeros((rows, 2, 1)),
        drivers=SimpleNamespace(ids=[], paths=np.empty((rows, 2, 0)), flags={}),
        dist=None,
        checks=None,
        diagnostics={},
        seed=5,
        draws=rows // inner,
        shadow=np.zeros((rows, 2, 1)),
    )
    result = SimpleNamespace(
        position_id="cash",
        model_id="common_stock",
        model_version="c" * 64,
        value_change=[float(value) for value in range(rows)],
        model_dump=lambda **_: {"position_id": "cash"},
    )
    batch = SimpleNamespace(results=[result], status="ok")
    request_ = RunRequest.model_validate(request(inner=inner))
    written = runs._write(
        tmp_path,
        "20261112T125232-abcd",
        request_,
        Spec.model_validate({"narrative": "ставка снижена"}),
        {"as_of": "2026-10-01"},
        paths,
        batch,
        None,
        "RUB",
        "2026-11-12T12:52:32+00:00",
        {"targets": []},
    )
    assert written["inner"] == inner and written["draws"] == rows // inner
    changes = np.load(tmp_path / "changes.npz")["changes"]
    assert changes.shape == (rows, 1), "every inner path of a group carries its own change"

