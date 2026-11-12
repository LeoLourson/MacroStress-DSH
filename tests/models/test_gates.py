"""``scripts/satellite_gates.py``: the gates S1-S3 and S6 of a base run, from saved artefacts."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mst import runs
from mst.drivers.registry import path_to_model
from mst.macro.registry import load_registry as load_macro_registry

SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "satellite_gates.py"
SHARE = {
    "id": "sber",
    "type": "common_stock",
    "notional": 10.0,
    "codes": {"secid": "SBER"},
}


@pytest.fixture(scope="module")
def module():
    """The script, imported from its file: it stays a script, not a package."""
    spec = importlib.util.spec_from_file_location("satellite_gates", SCRIPT)
    found = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(found)
    return found


@pytest.fixture()
def base_run(workspace):
    """The base scenario of the test workspace, run once with a single share position."""
    run_id = runs.new_id()
    runs.begin(
        workspace,
        run_id,
        {"scenario": "oil", "draws": 8, "seed": 11, "horizon": 3, "positions": [SHARE]},
    )
    runs.execute(workspace, run_id)
    return run_id


def _share_part(workspace, meta):
    """The recorded part of ``share_price`` of the run's bundle."""
    folder = workspace.models / "satellites" / meta["satellites"]
    return json.loads((folder / "meta.json").read_text(encoding="utf-8"))["quantities"][
        "share_price"
    ]


def test_the_gates_match_the_numbers_of_the_saved_run(module, workspace, base_run):
    report = module.gates(
        workspace,
        run=base_run,
        month=2,
        imoex="ru.usdrub",
        usdrub="gl.brent",
    )
    meta = runs.meta(workspace, base_run)
    part = _share_part(workspace, meta)
    with np.load(runs.folder(workspace, base_run) / "paths.npz") as saved:
        macro, drivers = saved["macro"], saved["drivers"]

    # S1: the median share return of the served shares minus the median of the index.
    registry = load_macro_registry()
    shares = [driver for driver in meta["drivers"] if driver.startswith("share_price:")]
    assert len(shares) >= 1
    j = meta["variables"].index("ru.usdrub")
    column = path_to_model(registry.variable("ru.usdrub").transform, macro[:, :, j])
    index = float(np.median(column[:, 2] - column[:, 0]))
    returns = []
    for driver in shares:
        k = meta["drivers"].index(driver)
        share = path_to_model("log100", drivers[:, :, k])
        returns.append(np.median(share[:, 2] - share[:, 0]))
    assert report["s1"]["value"] == pytest.approx(float(np.median(returns)) - index, abs=1e-9)
    assert report["s1"]["limit"] == 10.0

    # S2: the same difference for every named share.
    for driver in shares:
        k = meta["drivers"].index(driver)
        share = path_to_model("log100", drivers[:, :, k])
        value = float(np.median(share[:, 2] - share[:, 0]) - index)
        assert report["s2"][driver]["value"] == pytest.approx(value, abs=1e-9)
        assert report["s2"][driver]["limit"] == 35.0

    # S3: the share of driver cells whose band is wider than history.
    cells = [row for name, row in meta["checks"]["paths"].items() if name.startswith("driver/")]
    share = float(np.mean([row.get("band_wider_than_history", 0.0) for row in cells]))
    assert report["s3"]["share"] == pytest.approx(share, abs=1e-12)
    assert report["s3"]["limit"] == 0.05

    # The share of paths leaving the training range, from the paths and the bundle's bounds.
    outside = report["training_range"]["share_price"]["drivers"]
    for driver, value in outside.items():
        k = meta["drivers"].index(driver)
        column = path_to_model("log100", drivers[:, :, k])
        node = part["driver"].index(driver)
        low, high = part["train_low"][node], part["train_high"][node]
        left = ((column[:, 1:] < low) | (column[:, 1:] > high)).any(axis=1).mean()
        assert value == pytest.approx(float(left), abs=1e-12)

    # The band of each share at the month: the full band and the residuals-only one.
    for driver in shares:
        k = meta["drivers"].index(driver)
        column = path_to_model("log100", drivers[:, :, k])
        full = float(np.percentile(column[:, 2], 95) - np.percentile(column[:, 2], 5))
        assert report["band"][driver]["full"] == pytest.approx(full, abs=1e-9)
        residuals = report["band"][driver]["residuals_only"]
        assert 0.0 <= residuals <= full + 1e-6


def test_s3_counts_the_horizon_months_as_the_run_does(module, workspace, base_run, monkeypatch):
    # Every band is wider than a history of no width, except month 0, where all paths coincide:
    # the admission rule counts the months after it, so the share is one and not 2/3 of it.
    monkeypatch.setattr("mst.metrics.cells.history_range", lambda history: 1e-9)
    report = module.gates(workspace, run=base_run, month=2, imoex="ru.usdrub", usdrub="gl.brent")
    assert report["s3"]["share"] == 1.0


def test_s1_judges_the_served_shares_whichever_are_named(module, workspace):
    run_id = runs.new_id()
    other = SHARE | {"id": "gazp", "codes": {"secid": "GAZP"}}
    runs.begin(
        workspace,
        run_id,
        {"scenario": "oil", "draws": 8, "seed": 11, "positions": [SHARE, other]},
    )
    runs.execute(workspace, run_id)
    common = {"run": run_id, "month": 2, "imoex": "ru.usdrub", "usdrub": "gl.brent"}
    everything = module.gates(workspace, **common)
    named = module.gates(workspace, shares=["share_price:sber"], **common)
    assert named["s1"]["value"] == everything["s1"]["value"]
    assert list(named["s2"]) == ["share_price:sber"]
    assert len(everything["s2"]) == 2


def test_the_training_range_shares_cover_the_widened_range_too(module, workspace, base_run):
    report = module.gates(
        workspace,
        run=base_run,
        month=2,
        imoex="ru.usdrub",
        usdrub="gl.brent",
    )
    meta = runs.meta(workspace, base_run)
    with np.load(runs.folder(workspace, base_run) / "paths.npz") as saved:
        drivers = saved["drivers"]
    with np.load(
        workspace.models / "satellites" / meta["satellites"] / "arrays.npz"
    ) as stored:
        for quantity, row in report["training_range"].items():
            part = _share_part_of(workspace, meta, quantity)
            d = stored[f"{quantity}.d"]
            transform = _transform_of(workspace, meta, quantity)
            for driver, value in row["widened"].items():
                node = part["driver"].index(driver)
                column = path_to_model(transform, drivers[:, :, meta["drivers"].index(driver)])
                moves = np.abs(d[12:, node] - d[:-12, node])
                move = float(np.nanmax(moves)) if np.isfinite(moves).any() else 0.0
                low = part["train_low"][node] - move
                high = part["train_high"][node] + move
                left = ((column[:, 1:] < low) | (column[:, 1:] > high)).any(axis=1).mean()
                assert value == pytest.approx(float(left), abs=1e-12)
                # Widening the range can only keep paths inside, never add ones outside.
                assert value <= row["drivers"][driver] + 1e-12


def _share_part_of(workspace, meta, quantity):
    folder = workspace.models / "satellites" / meta["satellites"]
    return json.loads((folder / "meta.json").read_text(encoding="utf-8"))["quantities"][
        quantity
    ]


def _transform_of(workspace, meta, quantity):
    return _share_part_of(workspace, meta, quantity)["quantity"]["transform"]


def test_the_gates_name_the_largest_elasticity_to_the_dollar(module, workspace, base_run):
    report = module.gates(
        workspace,
        run=base_run,
        month=2,
        imoex="ru.usdrub",
        usdrub="gl.brent",
    )
    meta = runs.meta(workspace, base_run)
    with np.load(
        workspace.models / "satellites" / meta["satellites"] / "arrays.npz"
    ) as stored:
        gamma = stored["share_price.gamma"]
    part = _share_part(workspace, meta)
    largest = 0.0
    for driver in report["s2"]:
        node = part["driver"].index(driver)
        loading = (gamma[:, node, 2] + gamma[:, node, 3]) / (1 - gamma[:, node, 1])
        largest = max(largest, abs(float(np.median(loading))))
    assert report["s6"]["largest"] == pytest.approx(largest, abs=1e-9)
    assert report["s6"]["limit"] == 2.5
    assert report["s6"]["driver"] is not None


def test_a_run_whose_horizon_is_too_short_is_refused(module, workspace, base_run):
    with pytest.raises(ValueError, match="horizon of 3"):
        module.gates(workspace, run=base_run, month=12, imoex="ru.usdrub", usdrub="gl.brent")


def test_the_script_prints_the_gates_as_markdown(module, workspace, base_run, capsys):
    code = module.main(
        [
            "--workspace",
            str(workspace.root),
            "--run",
            base_run,
            "--month",
            "2",
            "--imoex",
            "ru.usdrub",
            "--usdrub",
            "gl.brent",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "S1" in printed and "S2" in printed and "S3" in printed and "S6" in printed
    assert "share_price:" in printed
