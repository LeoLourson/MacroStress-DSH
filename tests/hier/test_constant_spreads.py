"""The spread of the drift of one quantity: ``constant_spreads`` overrides ``constant_spread``."""

from datetime import date

import numpy as np

from mst.drivers.history import history
from mst.plugins.hier.estimate import fit
from mst.plugins.hier.spec import Sampler, Spec
from tests.drivers import simulated

AS_OF = date(2030, 1, 1)


def spec(**changes) -> Spec:
    values = dict(
        spread=0.1,
        constant_spread=10.0,
        rho_max=0.99,
        holdout=24,
        reach=12,
        sampler=Sampler(draws=25, burn=10, seed=7),
    )
    values.update(changes)
    return Spec(**values)


def test_a_quantity_named_in_constant_spreads_takes_its_own_spread(tmp_path):
    panel = simulated.panel(240, seed=5)
    store = simulated.store(tmp_path, panel, seed=2)
    hist = history(simulated.registry(), "price", panel, store, AS_OF)
    wide = fit(hist, spec())
    narrow = fit(hist, spec(constant_spread=0.1))
    named = fit(hist, spec(constant_spreads={"price": 0.1}))
    other = fit(hist, spec(constant_spreads={"bond_spread": 0.1}))
    assert np.array_equal(named.gamma, narrow.gamma)
    assert not np.array_equal(named.gamma, wide.gamma)
    assert np.array_equal(other.gamma, wide.gamma)
