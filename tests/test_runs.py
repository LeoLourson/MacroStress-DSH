"""mst.runs: a run of a scenario writes its paths, its results and its meta in the workspace."""

import asyncio
import time
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from fastmcp import Client

from mst import jobs, runs, store
from mst.mcp import build
from mst.metrics import baselines
from mst.plugin_sdk import FixedPaths
from mst.portfolio import tool as portfolio_tool
from mst.scenario import tool as scenario_tool

POSITIONS = [
    {
        "id": "sber",
        "name": "Сбербанк",
        "type": "common_stock",
        "notional": 10.0,
        "codes": {"secid": "SBER"},
    },
    {
        "id": "corp",
        "type": "bond",
        "notional": 5.0,
        "codes": {"isin": "RU000A0JX0J2"},
        "terms": {"duration": 2.0},
    },
    {
        "id": "mining",
        "type": "loan",
        "notional": 3.0,
        "codes": {"okved": "05"},
        "terms": {"lgd": 0.45},
    },
    {
        "id": "flat",
        "type": "real_estate",
        "notional": 1.0,
        "codes": {"okato": "14000000000"},
        "terms": {"market": "primary"},
    },
]
REQUEST = {"scenario": "oil", "draws": 5, "seed": 11, "positions": POSITIONS}


def start(ws, request=None, run_id=None):
    run_id = run_id or runs.new_id()
    runs.begin(ws, run_id, request or REQUEST)
    return run_id, runs.execute(ws, run_id)


def arrays(ws, run_id):
    meta = runs.meta(ws, run_id)
    with np.load(runs.folder(ws, run_id) / "paths.npz") as stored:
        macro, drivers = stored["macro"], stored["drivers"]
    with np.load(runs.folder(ws, run_id) / "changes.npz") as stored:
        changes = stored["changes"]
    return meta, macro, drivers, changes


def driver(meta, drivers, name):
    return drivers[:, :, meta["drivers"].index(name)]


def test_a_run_writes_meta_paths_results_and_changes(workspace):
    run_id, summary = start(workspace)
    assert summary["status"] == "ok" and summary["positions"] == len(POSITIONS)
    meta, macro, drivers, changes = arrays(workspace, run_id)
    assert meta["status"] == "ok"
    assert meta["admission"]["admitted"] is True
    assert meta["scenario"] == "oil" and meta["seed"] == 11 and meta["draws"] == 5
    assert meta["horizon"] == 3 and len(meta["months"]) == 4
    assert macro.shape == (5, 4, 3) and drivers.shape == (5, 4, 5)
    assert changes.shape == (5, len(POSITIONS))
    stored = runs.result(workspace, run_id)
    assert [item["position_id"] for item in stored["positions"]] == [
        position["id"] for position in POSITIONS
    ]
    assert all(item["status"] == "ok" for item in stored["positions"])
    assert all("value_change" not in item for item in stored["positions"])

    # The conditional forecast of the target indicators: quantiles by month, own units.
    forecast = stored["forecast"]
    assert forecast["targets"] == meta["drivers"]
    assert forecast["months"] == meta["months"][1:]
    for name in meta["drivers"]:
        row = forecast["drivers"][name]
        column = driver(meta, drivers, name)[:, 1:]
        low, mid, high = np.percentile(column, [5, 50, 95], axis=0)
        assert row["p05"] == [round(float(x), 6) for x in low]
        assert row["p50"] == [round(float(x), 6) for x in mid]
        assert row["p95"] == [round(float(x), 6) for x in high]
        assert "unit_months" not in row  # the horizon is the top-level months


def test_named_targets_are_the_forecast_of_the_run(workspace):
    request = dict(REQUEST, targets=["share_price:sber"])
    run_id, summary = start(workspace, request)
    stored = runs.result(workspace, run_id)
    assert stored["forecast"]["targets"] == ["share_price:sber"]
    assert list(stored["forecast"]["drivers"]) == ["share_price:sber"]
    assert stored["run"]["targets"] == ["share_price:sber"]


def test_a_target_the_paths_do_not_carry_is_refused(workspace):
    request = dict(REQUEST, targets=["share_price:nowhere"])
    with pytest.raises(ValueError, match="the paths carry no drivers"):
        start(workspace, request)


def test_a_request_refuses_duplicate_targets(workspace):
    request = dict(REQUEST, targets=["share_price:sber", "share_price:sber"])
    with pytest.raises(ValueError, match="share_price:sber is named twice"):
        runs.begin(workspace, runs.new_id(), request)


def test_the_forecast_refuses_duplicate_targets():
    paths = SimpleNamespace(
        drivers=SimpleNamespace(
            ids=["share_price:sber"],
            paths=np.zeros((2, 3, 1)),
            flags={"share_price:sber": frozenset()},
        ),
        months=[date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
    )
    with pytest.raises(ValueError, match="share_price:sber is named twice"):
        runs.forecast(paths, ["share_price:sber", "share_price:sber"])


def test_every_position_is_the_formula_of_its_model(workspace):
    run_id, _ = start(workspace)
    meta, _, drivers, changes = arrays(workspace, run_id)
    price = driver(meta, drivers, "share_price:sber")
    spread = driver(meta, drivers, "bond_spread:corp")
    overdue = driver(meta, drivers, "overdue_share:mining")
    housing = driver(meta, drivers, "housing_price_primary:belgorod")
    expected = {
        "sber": 10.0 * (price[:, -1] / price[:, 0] - 1),
        "corp": 5.0 * (1 - 2.0 * (spread[:, -1] - spread[:, 0]) / 100 - 1),
        "mining": 3.0 * (1 - 0.45 * (overdue[:, -1] - overdue[:, 0]) - 1),
        "flat": 1.0 * (housing[:, -1] / housing[:, 0] - 1),
    }
    for index, item in enumerate(meta["position_ids"]):
        assert np.allclose(changes[:, index], expected[item], rtol=0, atol=1e-12), item


def test_the_same_request_gives_the_same_numbers(workspace):
    first_id, _ = start(workspace)
    second_id, _ = start(workspace)
    _, _, first_drivers, first = arrays(workspace, first_id)
    _, _, second_drivers, second = arrays(workspace, second_id)
    # The meta differs by id and times; the paths, the changes and the metrics do not.
    assert np.array_equal(first_drivers, second_drivers)
    assert np.array_equal(first, second)
    assert (
        runs.result(workspace, first_id)["positions"]
        == runs.result(workspace, second_id)["positions"]
    )
    other_id, _ = start(workspace, REQUEST | {"seed": 12})
    _, _, _, other = arrays(workspace, other_id)
    assert not np.array_equal(first, other)


def test_a_position_without_a_served_driver_is_invalid_in_the_files(workspace):
    positions = [dict(POSITIONS[0], codes={"secid": "NOWHERE"})]
    run_id, summary = start(workspace, REQUEST | {"positions": positions})
    assert summary["status"] == "invalid"
    stored = runs.result(workspace, run_id)
    item = stored["positions"][0]
    assert item["status"] == "invalid"
    assert [error["code"] for error in item["diagnostics"] if error["severity"] == "error"] == [
        "driver_missing:share_price"
    ]
    assert "secid NOWHERE" in item["diagnostics"][0]["message"]
    _, _, _, changes = arrays(workspace, run_id)
    assert np.all(np.isnan(changes[:, 0]))


def test_a_run_folder_is_not_overwritten(workspace):
    run_id = runs.new_id()
    runs.begin(workspace, run_id, REQUEST)
    with pytest.raises(FileExistsError):
        runs.begin(workspace, run_id, REQUEST)


def test_a_run_refuses_another_base_currency(workspace):
    with pytest.raises(ValueError, match="base currency is RUB"):
        runs.begin(workspace, runs.new_id(), REQUEST | {"base_currency": "USD"})


def test_a_run_that_cannot_start_leaves_its_reason_in_the_meta(workspace):
    run_id = runs.new_id()
    runs.begin(workspace, run_id, REQUEST | {"scenario": "nowhere"})
    with pytest.raises(FileNotFoundError, match="nowhere"):
        runs.execute(workspace, run_id)
    meta = runs.meta(workspace, run_id)
    assert meta["status"] == "failed" and "nowhere" in meta["error"]
    stored = runs.result(workspace, run_id)
    assert stored["run"]["status"] == "failed"
    assert stored["positions"] == [] and stored["summary"] is None


def test_the_catalogue_names_every_run(workspace):
    run_id, _ = start(workspace)
    found = runs.catalogue(workspace)["runs"]
    assert [item["id"] for item in found] == [run_id]
    assert found[0]["status"] == "ok" and found[0]["positions"] == len(POSITIONS)


def test_a_request_needs_a_scenario_its_draws_and_its_seed(workspace):
    for missing in ("scenario", "draws", "seed", "positions"):
        with pytest.raises(ValueError):
            runs.begin(workspace, runs.new_id(), {k: v for k, v in REQUEST.items() if k != missing})


@pytest.fixture
def portfolio(workspace):
    """The saved portfolio ``demo`` of the workspace."""
    portfolio_tool.save(workspace.portfolios, "demo", {"name": "Демо", "positions": POSITIONS})
    return "demo"


@pytest.fixture
def shock_pair(workspace):
    """Three scenarios of horizon 6: no impact, one on months 2..3, one lasting to the end."""
    spec = scenario_tool.get(workspace.scenarios, "oil")["spec"]
    common = {
        "narrative": "перебой",
        "horizon": 6,
        "macro": spec["macro"],
        "satellites": spec["satellites"],
        "approved_by": "И. Иванов",
    }
    scenario_tool.save(workspace.scenarios, "plain", common)
    scenario_tool.save(
        workspace.scenarios,
        "shock",
        common
        | {
            "impacts": [
                {
                    "target": "share_price:sber",
                    "start": 2,
                    "values": [-10.0, -10.0],
                    "reason": "r",
                    "basis": "assumption",
                }
            ]
        },
    )
    scenario_tool.save(
        workspace.scenarios,
        "lasting",
        common
        | {
            "impacts": [
                {
                    "target": "share_price:sber",
                    "start": 1,
                    "values": [-10.0] * 6,
                    "reason": "r",
                    "basis": "assumption",
                }
            ]
        },
    )
    return "plain", "shock", "lasting"


def _report_of(ws, scenario, portfolio, draws: int = 5):
    """The report of one run, its saved monthly losses and its folder."""
    run_id = runs.new_id()
    runs.begin(
        ws, run_id, {"scenario": scenario, "draws": draws, "seed": 11, "portfolio": portfolio}
    )
    runs.execute(ws, run_id)
    stored = runs.result(ws, run_id)
    with np.load(runs.folder(ws, run_id) / "path_losses.npz") as saved:
        return stored["summary"], saved["losses"], runs.folder(ws, run_id)


def test_the_last_month_of_losses_by_month_equals_the_loss(workspace, portfolio, shock_pair):
    report, monthly, _ = _report_of(workspace, "plain", portfolio)
    block = report["losses_by_month"]
    assert monthly.shape == (5, 6)
    assert len(block["by_month"]) == 6
    last = block["by_month"][-1]
    assert last["month"] == report["loss"]["horizon_month"]
    for key in ("p05", "p50", "p95", "es95"):
        assert last[key] == report["loss"][key]
    # The value over the path is the block of the maximum, an ES at least as heavy as any month.
    assert block["path"]["es95"] >= max(row["es95"] for row in block["by_month"])


def test_an_impact_that_ends_before_the_horizon_shows_in_its_months_alone(
    workspace, portfolio, shock_pair
):
    plain, shock = shock_pair[:2]
    plain_report, _, _ = _report_of(workspace, plain, portfolio)
    shock_report, _, _ = _report_of(workspace, shock, portfolio)
    plain_rows = plain_report["losses_by_month"]["by_month"]
    shock_rows = shock_report["losses_by_month"]["by_month"]
    for key in ("p05", "p50", "p95", "es95"):
        # Months 2..3 carry the impact; months 1 and 4..6 do not.
        assert shock_rows[1][key] != plain_rows[1][key]
        assert shock_rows[2][key] != plain_rows[2][key]
        for t in (0, 3, 4, 5):
            assert shock_rows[t][key] == plain_rows[t][key]
        # The end of the horizon sees nothing of an impact that vanished at month 3.
        assert shock_report["loss"][key] == plain_report["loss"][key]


def test_the_comparison_of_a_scenario_of_impacts_only_matches_the_shift(
    workspace, portfolio, shock_pair
):
    plain, _, lasting = shock_pair
    lasting_report, _, lasting_folder = _report_of(workspace, lasting, portfolio, draws=40)
    plain_report, _, plain_folder = _report_of(workspace, plain, portfolio, draws=40)
    comparison = lasting_report["comparison"]
    assert set(comparison) == {"loss", "path"}
    centre = comparison["loss"]["centre"]
    # The impact lowers the share in the log100 scale by 10, that is a factor of exp(-0.1)
    # in own units from month 1 to the end; the other positions read no share price, so the
    # whole shift of the loss is the equity position's notional * (1 - exp(-0.1)) * P(T)/P(0).
    meta = runs.meta(workspace, plain_folder.name)
    k = meta["drivers"].index("share_price:sber")
    with np.load(plain_folder / "paths.npz") as saved:
        price = saved["drivers"][:, :, k]
    shift = 10.0 * (1.0 - np.exp(-0.1)) * price[:, -1] / price[:, 0]
    assert centre["value"] == pytest.approx(float(np.median(shift)), rel=0.05)
    assert centre["interval"][0] > 0.0  # narrow and clear of zero
    assert comparison["path"]["centre"]["value"] > 0.0
    # The comparison of the base with itself is absent, not zero.
    assert plain_report["comparison"] is None


def test_a_scenario_without_conditions_and_impacts_has_no_comparison(
    workspace, portfolio, shock_pair
):
    report, _, _ = _report_of(workspace, "plain", portfolio)
    assert report["comparison"] is None


def test_the_admission_limit_is_strict_and_weighs_every_quantity_equally():
    from mst.metrics.baselines import DOMAIN, WIDTH
    from mst.runs import ADMISSION_LIMIT, _admission

    assert ADMISSION_LIMIT == 0.05
    below = {"paths": {"macro/ru.key_rate": {DOMAIN: 0.0499, WIDTH: 0.0}}}
    at = {"paths": {"macro/ru.key_rate": {DOMAIN: 0.05, WIDTH: 0.0}}}
    assert _admission(below)["admitted"] is True
    assert _admission(at)["admitted"] is False

    mixed = {
        "paths": {
            "macro/a": {DOMAIN: 0.0, WIDTH: 0.2},
            "driver/d:x": {DOMAIN: 0.1, WIDTH: 0.0},
        }
    }
    found = _admission(mixed)
    assert found["shares"] == {DOMAIN: 0.05, WIDTH: 0.1}
    assert found["admitted"] is False
    assert found["broken"] == [
        {"condition": WIDTH, "quantity": "macro/a", "share": 0.2},
        {"condition": DOMAIN, "quantity": "driver/d:x", "share": 0.1},
    ]
    # Without paths to check there is nothing broken, and no refusal either.
    empty = _admission(None)
    assert empty["admitted"] is True and empty["broken"] == []


def test_a_run_whose_bands_leave_the_domain_is_not_admitted_and_keeps_its_numbers(
    workspace, portfolio, shock_pair, monkeypatch
):
    # A domain no band can sit in: the calculation is not admitted, the status says so, the
    # meta names the quantity, and the numbers stay for diagnostics.
    monkeypatch.setattr(baselines, "macro_domain", lambda variable: (1e9, np.inf))
    report, _, folder = _report_of(workspace, "plain", portfolio)
    meta = runs.meta(workspace, folder.name)
    assert report["status"] == "not_admitted"
    assert "not_admitted" in report["flags"]
    assert meta["status"] == "not_admitted"
    assert meta["admission"]["admitted"] is False
    assert meta["admission"]["limit"] == 0.05
    quantities = [row["quantity"] for row in meta["admission"]["broken"]]
    assert "macro/ru.usdrub" in quantities
    assert report["loss"] is not None and report["losses_by_month"] is not None


def test_a_positions_run_without_a_portfolio_carries_the_admission_in_its_meta(
    workspace, shock_pair, monkeypatch
):
    monkeypatch.setattr(baselines, "macro_domain", lambda variable: (1e9, np.inf))
    run_id = runs.new_id()
    runs.begin(
        workspace, run_id, {"scenario": "plain", "draws": 5, "seed": 11, "positions": POSITIONS}
    )
    summary = runs.execute(workspace, run_id)
    meta = runs.meta(workspace, run_id)
    assert meta["admission"]["admitted"] is False
    assert meta["status"] == "not_admitted"
    assert summary["status"] == "not_admitted"


def test_the_contributions_carry_the_interval_of_the_same_function(
    workspace, portfolio, shock_pair
):
    plain, shock, _ = shock_pair
    run_id = runs.new_id()
    runs.begin(
        workspace,
        run_id,
        {
            "scenario": shock,
            "draws": 5,
            "seed": 11,
            "portfolio": portfolio,
            "contributions": True,
        },
    )
    runs.execute(workspace, run_id)
    stored = store.read_json(runs.folder(workspace, run_id) / "contributions.json")
    (row,) = stored["contributions"]
    assert row["element"] == "impact#1"
    assert row["loss"] is not None and row["difference"] is not None
    interval = row["interval"]
    assert set(interval) == {"centre", "width", "p95", "es95"}
    for block in interval.values():
        assert block["interval"][0] <= block["interval"][1]
    # The centre of the same function is the difference of the medians the row reports.
    assert interval["centre"]["value"] == pytest.approx(row["difference"]["p50"])


def test_the_paths_cut_at_a_month_keep_the_start_flags_and_seed():
    paths = FixedPaths(
        macros={"gl.brent": np.arange(14.0).reshape(2, 7)},
        drivers={"price:alpha": np.ones((2, 7))},
        driver_flags={"price:alpha": frozenset({"impact"})},
        start=date(2026, 9, 1),
        seed=11,
    )
    cut = paths.up_to(3)
    assert cut.horizon == 3 and cut.draws == 2
    assert np.array_equal(cut.macro("gl.brent"), paths.macro("gl.brent")[:, :4])
    assert np.array_equal(cut.driver("price:alpha"), np.ones((2, 4)))
    assert cut.flags("price:alpha") == frozenset({"impact"})
    assert cut.start == date(2026, 9, 1) and cut.seed == 11
    with pytest.raises(ValueError, match="run to month 6"):
        paths.up_to(7)


def _macro_version(versions):
    """The engine fixture's macro version loaded back, as a run receives it."""
    from mst.macro import version as macro_version
    from mst.plugins.bvar import MODEL

    return macro_version.load(versions.models, versions.macro, {"bvar.sv": MODEL})


def _history_of(versions, driver) -> tuple[float, float]:
    """``(middle, span)`` of a driver's own history in the model scale."""
    quantity, node = versions.satellite.node(driver)
    history = versions.satellite.arrays[f"{quantity}.d"][:, node]
    finite = history[np.isfinite(history)]
    return float((finite.max() + finite.min()) / 2), float(finite.max() - finite.min())


def _banded_paths(versions, bands: dict[str, tuple[float, float]]) -> SimpleNamespace:
    """Paths whose driver bands are set in the model scale and carried in own units.

    ``bands`` maps a driver to ``(centre, width)`` of its band in the model scale; the band
    is the same in every month, so the share of broken months is exactly 0 or 1. The driver
    paths come back through ``from_model``, as a run's paths hold them in own units — what
    ``_checks`` receives and must map back before it compares with the history.
    """
    from mst.drivers.registry import from_model

    rng = np.random.default_rng(17)
    inside = rng.uniform(-0.5, 0.5, 40)
    ids, columns = [], []
    for driver, (centre, width) in bands.items():
        quantity, _ = versions.satellite.node(driver)
        declared = versions.satellite.registry.quantity(quantity)
        model = centre + width * inside[:, None] * np.ones((1, 13))
        ids.append(driver)
        columns.append(np.asarray(from_model(declared.transform, model), dtype=float))
    return SimpleNamespace(
        ids=list(versions.data.ids),
        macro=np.ones((40, 13, len(versions.data.ids))),
        drivers=SimpleNamespace(
            ids=ids,
            paths=np.stack(columns, axis=-1),
            flags={driver: frozenset() for driver in ids},
        ),
    )


def _bare_spec(versions):
    """A scenario without conditions or impacts, so only the paths are checked."""
    from mst.scenario.spec import Spec

    return Spec.model_validate(
        {
            "narrative": "шкала",
            "horizon": 12,
            "macro": versions.macro,
            "satellites": versions.satellites,
            "approved_by": "И. Иванов",
        }
    )


def _driver_checks(versions, bands) -> dict:
    """The ``driver/...`` rows of ``_checks`` on paths with the given bands."""
    macro = _macro_version(versions)
    paths = _banded_paths(versions, bands)
    return runs._checks(_bare_spec(versions), macro, paths, None, versions.satellite)[
        "paths"
    ]


def test_a_log100_band_narrower_than_the_history_reports_zero_at_a_high_and_low_price(
    versions,
):
    # The width of the band is set in the model scale, where the history lives; the level
    # of the price — a share at 5239 or at 5 — must not enter the verdict. A band of half
    # the history range is admitted whatever the price.
    _, span = _history_of(versions, "share_price:sber")
    rows = _driver_checks(
        versions,
        {
            "share_price:sber": (100.0 * np.log(5239.0), 0.6 * span),
            "share_price:gazp": (100.0 * np.log(5.0), 0.6 * span),
        },
    )
    assert rows["driver/share_price:sber"]["band_wider_than_history"] == 0.0
    assert rows["driver/share_price:gazp"]["band_wider_than_history"] == 0.0
    assert rows["driver/share_price:sber"]["band_outside_domain"] == 0.0


def test_a_log100_band_wider_than_the_history_reports_the_full_share(versions):
    middle, span = _history_of(versions, "share_price:sber")
    rows = _driver_checks(
        versions, {"share_price:sber": (middle + 2.0 * span, 2.5 * span)}
    )
    assert rows["driver/share_price:sber"]["band_wider_than_history"] == 1.0


def test_a_logit_driver_and_an_identity_driver_are_measured_in_their_own_model_scale(
    versions,
):
    # overdue_share is logit and bond_spread is identity: each band is measured in its own
    # quantity's model scale, half the history range admitted and two and a half times it
    # refused, wherever the quantity itself sits.
    for driver, share_of_history, expected in (
        ("overdue_share:mining", 0.6, 0.0),
        ("overdue_share:mining", 2.5, 1.0),
        ("bond_spread:corp", 0.6, 0.0),
        ("bond_spread:corp", 2.5, 1.0),
    ):
        middle, span = _history_of(versions, driver)
        rows = _driver_checks(versions, {driver: (middle, share_of_history * span)})
        assert rows[f"driver/{driver}"]["band_wider_than_history"] == expected, driver


def call(ws, name, arguments):
    """One tool call against a server built for ``ws``."""

    async def run():
        async with Client(build(ws)) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def wait(ws, job_id, timeout=120.0):
    """Wait until the job's process is gone."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not jobs.status(ws, job_id)["running"]:
            return
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish in {timeout} s")


def test_the_run_tool_starts_a_job_and_reports_its_result(workspace):
    started = call(
        workspace,
        "run",
        {"action": "start", "scenario": "oil", "draws": 3, "seed": 5, "positions": POSITIONS},
    ).data
    wait(workspace, started["job"])
    assert jobs.status(workspace, started["job"])["exit_code"] == 0
    status = call(workspace, "run", {"action": "status", "id": started["run"]}).data
    assert status["meta"]["status"] == "ok"
    assert status["job"]["exit_code"] == 0
    stored = call(workspace, "run", {"action": "result", "id": started["run"]}).data
    assert len(stored["positions"]) == len(POSITIONS)
    listed = call(workspace, "run", {"action": "list"}).data["runs"]
    assert [item["id"] for item in listed] == [started["run"]]


def test_the_run_tool_forecasts_the_targets_it_is_given(workspace):
    started = call(
        workspace,
        "run",
        {
            "action": "start",
            "scenario": "oil",
            "draws": 3,
            "seed": 5,
            "positions": POSITIONS,
            "targets": ["share_price:sber"],
        },
    ).data
    wait(workspace, started["job"])
    stored = call(workspace, "run", {"action": "result", "id": started["run"]}).data
    assert stored["run"]["targets"] == ["share_price:sber"]
    assert stored["forecast"]["targets"] == ["share_price:sber"]
    assert list(stored["forecast"]["drivers"]) == ["share_price:sber"]


def test_the_run_tool_names_the_argument_it_lacks(workspace):
    with pytest.raises(Exception, match="seed is required"):
        call(
            workspace,
            "run",
            {"action": "start", "scenario": "oil", "draws": 3, "positions": POSITIONS},
        )


def test_a_run_that_cannot_get_the_resource_is_marked_failed(workspace):
    blocker = jobs.start(workspace, "runs", ["-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(Exception, match="busy"):
            call(
                workspace,
                "run",
                {
                    "action": "start",
                    "scenario": "oil",
                    "draws": 3,
                    "seed": 5,
                    "positions": POSITIONS,
                },
            )
    finally:
        jobs.cancel(workspace, blocker)
    failed = [item for item in runs.catalogue(workspace)["runs"] if item["status"] == "failed"]
    assert len(failed) == 1
    assert "busy" in runs.meta(workspace, failed[0]["id"])["error"]
