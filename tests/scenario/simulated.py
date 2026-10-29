"""A macro version and its satellites simulated from known models, for the scenario tests."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from mst.data.store import SeriesStore
from mst.drivers import version as satellite_version
from mst.drivers.spec import Sampler
from mst.drivers.spec import load_spec as load_drivers_spec
from mst.macro import version as macro_version
from mst.macro.estimate import Posterior, estimate
from mst.macro.identify import identify
from mst.macro.panel import Panel, months_between, shift
from mst.macro.registry import Shock

from ..drivers import simulated as drivers
from ..macro import simulated as macro

IDS = ["gl.brent", "ru.key_rate", "ru.usdrub"]
MONTHS = 120
AS_OF = date(2010, 1, 1)
SHOCK = Shock(id="gl.oil", label="нефтяной", rises=["gl.brent"])


def panel(months: int = MONTHS, seed: int = 3) -> Panel:
    """A panel whose dynamics are the known VAR of ``tests.macro.simulated``, under real ids."""
    y = macro.simulate(months, np.random.default_rng(seed))
    return Panel(
        ids=IDS,
        n_global=1,
        months=months_between(macro.START, shift(macro.START, months - 1)),
        y=y,
        approximate=np.zeros(y.shape, dtype=bool),
    )


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
    data = panel()
    data_store = drivers.store(series, data)
    spec = macro.spec(draws=40, burn=0, seed=2, stationary=IDS)
    model = identify(estimate(data, spec), [SHOCK], sign_lag=0, tries=500, seed=1)
    name = macro_version.save(model, spec, data, AS_OF, models)
    satellite_spec = load_drivers_spec().model_copy(
        update={"sampler": Sampler(draws=300, burn=100, seed=7)}
    )
    satellite = satellite_version.build(
        drivers.registry(), satellite_spec, data, data_store, AS_OF, name
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
