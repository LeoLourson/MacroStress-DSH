"""The hierarchical satellite resumes an interrupted estimate and reports its progress."""

from datetime import date

import numpy as np
import pytest

from mst.model_sdk.contracts import SatelliteEstimateContext
from mst.plugins.hier import model as hier_model
from mst.plugins.hier.model import HierModel
from mst.plugins.hier.spec import Sampler, load_spec
from tests.drivers import simulated

AS_OF = date(2030, 1, 1)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    panel = simulated.panel(240)
    store = simulated.store(tmp_path_factory.mktemp("series"), panel)
    return simulated.registry(), panel, store


def quick(seed: int = 1):
    return load_spec().model_copy(update={"sampler": Sampler(draws=40, burn=10, seed=seed)})


def context(data, checkpoints):
    registry, panel, store = data
    return SatelliteEstimateContext(
        store=store,
        registry=registry,
        panel=panel,
        as_of=AS_OF,
        macro="macro-version",
        quantities=("price",),
        checkpoints=checkpoints,
    )


def refuse_to_fit(*args, **kwargs):
    raise AssertionError("the quantity was fitted again")


def test_a_second_estimate_takes_the_finished_quantity_from_its_checkpoint(
    data, tmp_path, monkeypatch
):
    first = HierModel(spec=quick()).estimate(context(data, tmp_path))
    monkeypatch.setattr(hier_model, "fit", refuse_to_fit)
    monkeypatch.setattr(hier_model, "check", refuse_to_fit)
    again = HierModel(spec=quick()).estimate(context(data, tmp_path))
    assert again.parts["price"].meta == first.parts["price"].meta
    for name, array in first.parts["price"].arrays.items():
        np.testing.assert_array_equal(again.parts["price"].arrays[name], array)
    assert set(again.residuals) == set(first.residuals)
    for driver, series in first.residuals.items():
        np.testing.assert_array_equal(again.residuals[driver], series)


def test_a_changed_spec_does_not_take_the_old_checkpoint(data, tmp_path, monkeypatch):
    HierModel(spec=quick(seed=1)).estimate(context(data, tmp_path))
    monkeypatch.setattr(hier_model, "fit", refuse_to_fit)
    with pytest.raises(AssertionError, match="fitted again"):
        HierModel(spec=quick(seed=2)).estimate(context(data, tmp_path))


def test_without_a_checkpoint_folder_nothing_is_written(data, tmp_path):
    HierModel(spec=quick()).estimate(context(data, None))
    assert list(tmp_path.iterdir()) == []


def test_the_estimate_reports_each_quantity_and_the_sampler_steps(data, capsys):
    HierModel(spec=quick()).estimate(context(data, None))
    err = capsys.readouterr().err
    assert "price: fit 100%" in err
    assert "price: check 100%" in err
    assert "price: done" in err
