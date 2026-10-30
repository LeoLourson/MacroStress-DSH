"""mst.drivers.registry: quantities, objects, drivers and how positions match them."""

import json

import pytest

from mst.drivers.registry import Registry, from_model, load_registry, to_model
from mst.macro.registry import load_registry as load_macro_registry
from tests.drivers import simulated


def test_the_registry_declares_a_catalog_per_quantity_and_known_macro_variables():
    registry = load_registry(catalog=None)
    registry.check(load_macro_registry())
    assert all(q.catalog is not None for q in registry.quantities)


def test_a_catalog_file_adds_objects_and_drivers_and_positions_match_by_code(tmp_path):
    objects = [
        {"id": "economy", "class": "activity_0", "label": "ВСЕГО", "codes": {"okved": ["4"]}},
        {"id": "construction", "class": "activity_1", "label": "строительство",
         "parent": "economy", "codes": {"okved": ["41", "42"]}},
        {"id": "building", "class": "activity_2", "label": "здания", "parent": "construction",
         "codes": {"okved": ["41"]}},
    ]  # fmt: skip
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps({"quantities": {"overdue_share": {"objects": objects,
                    "drivers": ["economy", "construction", "building"]}}}),  # fmt: skip
        encoding="utf-8",
    )
    registry = load_registry(catalog=catalog)
    assert registry.match("overdue_share", "okved", "41.20") == "overdue_share:building"
    assert registry.match("overdue_share", "okved", "42.11") == "overdue_share:construction"
    assert registry.match("overdue_share", "okved", "47.11") == "overdue_share:economy"
    assert registry.match("overdue_share", "okved", "41.20", among={"overdue_share:economy"}) == (
        "overdue_share:economy"
    )
    assert registry.match("overdue_share", "okved", "01.11") is None
    assert next(d for d in registry.drivers if d.object == "building").series == (
        "drv.overdue_share.building"
    )


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda o: o[1].update(parent="nowhere"), "unknown parent"),
        (lambda o: o[0].update(parent="alpha"), "cycle"),
    ],
)
def test_the_registry_refuses_broken_trees(change, message):
    raw = simulated.registry().model_dump(by_alias=True)
    change(raw["objects"])
    with pytest.raises(ValueError, match=message):
        Registry.model_validate(raw)


def test_transforms_invert():
    for transform, value in (("identity", -3.0), ("log100", 7.5), ("logit", 0.07)):
        assert from_model(transform, to_model(transform, value)) == pytest.approx(value)
    with pytest.raises(ValueError):
        to_model("logit", 1.0)


def test_an_unknown_quantity_or_object_is_a_key_error():
    registry = load_registry()
    with pytest.raises(KeyError):
        registry.quantity("no_such_quantity")
    with pytest.raises(KeyError):
        registry.object("no_such_object")
