"""A macro version and its satellites simulated for the engine and portfolio tests.

The satellites carry exactly the quantities the position models read: a share price, a bond
spread, an overdue share and a primary-market housing price. Every driver is a known
autoregression on the simulated macro panel, so a test can compute what a position must be
worth and read the same number from the run.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mst.data.store import Observation, SeriesStore
from mst.drivers import version as satellite_version
from mst.drivers.objects import series_id
from mst.drivers.registry import Registry, from_model
from mst.drivers.spec import Sampler
from mst.drivers.spec import load_spec as load_drivers_spec
from mst.macro import version as macro_version
from mst.macro.panel import Panel
from mst.plugins.bvar.estimate import Posterior, estimate
from mst.plugins.bvar.identify import identify
from mst.plugins.bvar.model import parts
from tests.drivers import simulated as drivers
from tests.scenario import simulated as scenario

QUANTITIES = (
    ("share_price", "log100", "gl.brent"),
    ("bond_spread", "identity", "ru.key_rate"),
    ("overdue_share", "logit", "ru.key_rate"),
    ("housing_price_primary", "log100", "ru.key_rate"),
)
OBJECTS = [
    {"id": "shares", "class": "market", "label": "Акции"},
    {
        "id": "sber",
        "class": "company",
        "label": "Сбербанк",
        "parent": "shares",
        "codes": {"secid": ["SBER"], "inn": ["7707083893"]},
    },
    {
        "id": "gazp",
        "class": "company",
        "label": "Газпром",
        "parent": "shares",
        "codes": {"secid": ["GAZP"]},
    },
    {"id": "bonds", "class": "market", "label": "Облигации"},
    {
        "id": "corp",
        "class": "issuer",
        "label": "Эмитент",
        "parent": "bonds",
        "codes": {"isin": ["RU000A0JX0J2"]},
    },
    {"id": "activities", "class": "economy", "label": "Отрасли"},
    {
        "id": "mining",
        "class": "activity",
        "label": "Добыча",
        "parent": "activities",
        "codes": {"okved": ["05", "06"]},
    },
    {"id": "regions", "class": "country", "label": "Регионы"},
    {
        "id": "belgorod",
        "class": "region",
        "label": "Белгородская область",
        "parent": "regions",
        "codes": {"okato": ["14000000000"]},
    },
]
#: The autoregression of each quantity in model scale: alpha, rho, beta on x_t and x_{t-1}.
DRIFTS = {
    "share_price": ((4.0, 0.3, 1.0, -0.6), 0.05),
    "bond_spread": ((1.0, 0.2, 0.5, -0.3), 0.01),
    "overdue_share": ((-3.0, 0.2, 0.05, -0.03), 0.002),
    "housing_price_primary": ((5.0, 0.2, 0.5, -0.3), 0.05),
}
_DRIVERS = [
    {"quantity": "share_price", "object": "sber"},
    {"quantity": "share_price", "object": "gazp"},
    {"quantity": "bond_spread", "object": "corp"},
    # A driver without history of its own: the check has nothing above it to take parameters from.
    {"quantity": "bond_spread", "object": "sber", "series": None},
    {"quantity": "overdue_share", "object": "mining"},
    {"quantity": "housing_price_primary", "object": "belgorod"},
]
DRIVERS = [
    {**driver, "series": driver.get("series", series_id(driver["quantity"], driver["object"]))}
    for driver in _DRIVERS
]


def registry() -> Registry:
    """The drivers of the tests: one object per quantity the models read."""
    return Registry.model_validate(
        {
            "quantities": [
                {
                    "id": quantity,
                    "label": quantity,
                    "unit": "ед.",
                    "transform": transform,
                    "regressors": [regressor],
                    "lags": 1,
                }
                for quantity, transform, regressor in QUANTITIES
            ],
            "objects": OBJECTS,
            "drivers": DRIVERS,
        }
    )


def store(root: Path, data: Panel, seed: int = 2) -> SeriesStore:
    """Every driver's history as a known autoregression on its regressor of the panel."""
    rng = np.random.default_rng(seed)
    result = SeriesStore(root)
    columns = {name: index for index, name in enumerate(data.ids)}
    for driver in DRIVERS:
        if driver["series"] is None:
            continue
        quantity = next(item for item in QUANTITIES if item[0] == driver["quantity"])
        gamma, sigma = DRIFTS[driver["quantity"]]
        x = data.y[:, columns[quantity[2]]]
        values = from_model(quantity[1], drivers.path(np.array(gamma), x, rng, sigma=sigma))
        result.write(
            driver["series"],
            [
                Observation(month, float(value), month, False)
                for month, value in zip(data.months, values, strict=True)
            ],
        )
    return result


@dataclass(frozen=True)
class Versions:
    """Two saved model versions of one panel, with the objects they were saved from."""

    models: Path
    macro: str
    satellites: str
    model: Posterior
    satellite: satellite_version.Satellites
    data: Panel
    store: SeriesStore


def build(models: Path, series: Path) -> Versions:
    """Estimate a macro model on a simulated panel and its satellites, and save both."""
    data = scenario.panel()
    data_store = store(series, data)
    spec = scenario.macro.spec(draws=40, burn=0, seed=2, stationary=scenario.IDS)
    model = identify(estimate(data, spec), [scenario.SHOCK], sign_lag=0, tries=500, seed=1)
    estimate_ = parts(model, spec, data, scenario.AS_OF)
    name = macro_version.save("bvar.sv", estimate_.meta, estimate_.arrays, models)
    satellite_spec = load_drivers_spec().model_copy(
        update={"sampler": Sampler(draws=300, burn=100, seed=7)}
    )
    satellite = satellite_version.build(
        registry(), satellite_spec, data, data_store, scenario.AS_OF, name
    )
    return Versions(
        models=models,
        macro=name,
        satellites=satellite_version.save(satellite, models),
        model=model,
        satellite=satellite,
        data=data,
        store=data_store,
    )
