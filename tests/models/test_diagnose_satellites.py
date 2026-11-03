"""``diagnose-satellites``: the pull to the long-run level, the elasticities and the checks."""

from datetime import date

import numpy as np
import pytest

from mst.drivers import bundle as satellite_bundle
from mst.model_sdk.contracts import SatelliteEstimateContext
from mst.models.diagnose_satellites import diagnose_satellites
from mst.plugins.hier import MODEL
from mst.plugins.hier.model import HierModel
from mst.plugins.hier.spec import Sampler, Spec
from tests.drivers import simulated

AS_OF = date(2030, 1, 1)


def quick_spec() -> Spec:
    """A short sampler: the medians of a tight posterior do not need a thousand draws."""
    return Spec(
        spread=0.1,
        constant_spread=10.0,
        rho_max=0.99,
        holdout=24,
        reach=12,
        sampler=Sampler(draws=60, burn=30, seed=7),
    )


@pytest.fixture(scope="module")
def estimated(tmp_path_factory):
    """The plugin's estimate on the synthetic data, saved as a bundle; its root and name."""
    panel = simulated.panel(240)
    store = simulated.store(tmp_path_factory.mktemp("series"), panel)
    registry = simulated.registry()
    model = HierModel(spec=quick_spec())
    estimate = model.estimate(
        SatelliteEstimateContext(
            store=store,
            registry=registry,
            panel=panel,
            as_of=AS_OF,
            macro="macro-version",
            quantities=("price",),
        )
    )
    meta, arrays = satellite_bundle.assemble(
        registry, "macro-version", AS_OF, {model.model_id: estimate}
    )
    root = tmp_path_factory.mktemp("models")
    return root, satellite_bundle.save(meta, arrays, root)


def test_the_report_recovers_the_true_rho_and_long_run_elasticities(estimated):
    root, name = estimated
    report = diagnose_satellites(root, name)
    assert report["version"] == name
    assert report["macro"] == "macro-version"
    price = report["quantities"]["price"]
    assert price["model"] == MODEL.model_id
    assert price["differenced"] is False
    sector = price["objects"]["price:sector"]
    assert sector["rows"] > 0
    assert sector["rho"]["p50"] == pytest.approx(simulated.SECTOR[1], abs=0.06)
    truth = (simulated.SECTOR[2] + simulated.SECTOR[3]) / (1 - simulated.SECTOR[1])
    assert sector["long_run"]["gl.brent"]["p50"] == pytest.approx(truth, abs=0.5)
    for name_, gamma in simulated.COMPANIES.items():
        object_ = price["objects"][f"price:{name_}"]
        assert object_["rho"]["p50"] == pytest.approx(gamma[1], abs=0.08)
        truth = (gamma[2] + gamma[3]) / (1 - gamma[1])
        assert object_["long_run"]["gl.brent"]["p50"] == pytest.approx(truth, abs=0.7)
    # The simulated model beats the autoregression everywhere, so every reason is empty;
    # the driver without history is served through the sector above it.
    assert price["checks"] == {"": 5}
    assert price["served"] == 5
    assert price["rmse_ratio"]["p50"] < 1.0


def test_the_reported_pull_is_the_long_run_level_by_twelve_months(estimated):
    root, name = estimated
    report = diagnose_satellites(root, name)
    for object_ in report["quantities"]["price"]["objects"].values():
        if object_["rows"] == 0:
            assert object_["d_star_minus_d0"] is None
            assert object_["pull_12m"] is None
            continue
        rho = object_["rho"]["p50"]
        gap = object_["d_star_minus_d0"]["p50"]
        assert object_["pull_12m"]["p50"] == pytest.approx(gap * (1 - rho**12), rel=0.02)


def _hand_made(tmp_path, *, differences=()):
    """A bundle whose one draw is exactly the truth, so every number has a closed form."""
    registry = simulated.registry()
    months = [date(2010, 1 + t % 12, 1) for t in range(24)]
    gamma = np.stack([simulated.SECTOR, simulated.COMPANIES["alpha"]])[None, :, :]
    x = np.linspace(0.0, 10.0, len(months))[:, None]
    d = np.column_stack([np.full(len(months), 100.0), np.full(len(months), 80.0)])
    part = {
        "model": MODEL.model_id,
        "quantity": registry.quantity("price").model_dump(mode="json", by_alias=True),
        "months": [month.isoformat() for month in months],
        "nodes": ["sector", "alpha"],
        "parent": [-1, 0],
        "classes": ["sector", "company"],
        "driver": ["price:sector", "price:alpha"],
        "rows": [len(months), len(months)],
        "train_low": [90.0, 70.0],
        "train_high": [110.0, 90.0],
        "checks": {
            "0": {
                "satellite": 1.0,
                "baseline": 2.0,
                "forecasts": 12,
                "served": True,
                "reason": "",
            },
            "1": {
                "satellite": 3.0,
                "baseline": 2.0,
                "forecasts": 12,
                "served": False,
                "reason": "not more accurate than without macro",
            },
        },
        "spec": {"differences": list(differences)},
    }
    meta = {
        "macro": "macro-version",
        "as_of": AS_OF.isoformat(),
        "registry": registry.model_dump(mode="json", by_alias=True),
        "quantities": {"price": part},
        "correlated": ["price:sector", "price:alpha"],
        "unserved": [],
    }
    arrays = {
        "price.gamma": gamma,
        "price.sigma": np.ones((1, 2)),
        "price.d": d,
        "price.approximate": np.zeros(d.shape, dtype=bool),
        "price.x": x,
        "correlation": np.eye(2),
    }
    return satellite_bundle.save(meta, arrays, tmp_path), x[-1, 0], d[-1]


def test_a_part_of_known_draws_reports_the_closed_form_numbers(tmp_path):
    name, today, last = _hand_made(tmp_path)
    report = diagnose_satellites(tmp_path, name)
    price = report["quantities"]["price"]
    sector, alpha = price["objects"]["price:sector"], price["objects"]["price:alpha"]

    assert sector["rho"] == {"p05": 0.5, "p50": 0.5, "p95": 0.5}
    assert sector["long_run"]["gl.brent"] == {"p05": 1.4, "p50": 1.4, "p95": 1.4}
    gap = (simulated.SECTOR[0] + (simulated.SECTOR[2] + simulated.SECTOR[3]) * today) / (
        1 - simulated.SECTOR[1]
    ) - last[0]
    assert sector["d_star_minus_d0"]["p50"] == pytest.approx(gap)
    assert sector["pull_12m"]["p50"] == pytest.approx(gap * (1 - 0.5**12))

    truth = (simulated.COMPANIES["alpha"][2] + simulated.COMPANIES["alpha"][3]) / (
        1 - simulated.COMPANIES["alpha"][1]
    )
    assert alpha["long_run"]["gl.brent"]["p50"] == pytest.approx(truth)

    # The counts of the check's reasons and the RMSE ratio of the served nodes.
    assert price["checks"] == {"": 1, "not more accurate than without macro": 1}
    assert price["served"] == 1
    assert price["rmse_ratio"] == {"p25": 0.5, "p50": 0.5, "p75": 0.5}
    assert sector["rmse_ratio"] == 0.5
    assert alpha["rmse_ratio"] == 1.5


def test_a_quantity_in_differences_has_no_long_run_level(tmp_path):
    name, _, _ = _hand_made(tmp_path, differences=("price",))
    report = diagnose_satellites(tmp_path, name)
    price = report["quantities"]["price"]
    assert price["differenced"] is True
    sector = price["objects"]["price:sector"]
    assert sector["d_star_minus_d0"] is None
    assert sector["pull_12m"] is None
    # The elasticity to a permanent shift of x is still the sum of the lags over 1 - rho.
    assert sector["long_run"]["gl.brent"]["p50"] == pytest.approx(1.4)
    assert sector["drift"]["p50"] == pytest.approx(simulated.SECTOR[0])
