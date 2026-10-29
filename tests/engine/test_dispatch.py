"""mst.engine.dispatch: what a position gets, and why it is not calculated."""

from datetime import date

import pytest

from mst.engine import run
from mst.engine.dispatch import prepare
from mst.engine.model_catalog import Catalog
from mst.plugin_sdk import (
    CalculationModel,
    CalculationResult,
    DataRequirements,
    DriverRequirement,
    FixedPaths,
    Position,
    RunOptions,
)
from mst.plugin_sdk.results import money
from tests.engine.fixed import paths
from tests.engine.simulated import registry

OPTIONS = RunOptions(as_of=date(2026, 9, 1), horizon=2)


def errors(result):
    """The codes of the diagnostics that stopped a position."""
    return [item.code for item in result.diagnostics if item.severity == "error"]


def equity(**changes) -> Position:
    values = {
        "id": "sber",
        "name": "Сбербанк",
        "type": "common_stock",
        "notional": 10.0,
        "codes": {"secid": "SBER"},
    }
    values.update(changes)
    return Position.model_validate(values)


def bond(**changes) -> Position:
    values = {
        "id": "corp",
        "type": "bond",
        "notional": 5.0,
        "codes": {"isin": "RU000A0JX0J2"},
        "terms": {"duration": 2.0},
    }
    values.update(changes)
    return Position.model_validate(values)


def test_a_served_driver_is_resolved_by_the_position_codes():
    prepared, problems = prepare(equity(), OPTIONS, paths(), registry(), {"share_price:sber"})
    assert prepared is not None
    assert prepared.drivers == {"price": "share_price:sber"}
    assert prepared.value_current == 10.0
    assert problems == []


def test_the_inn_resolves_when_the_ticker_is_not_served():
    prepared, _ = prepare(
        equity(codes={"inn": "7707083893"}), OPTIONS, paths(), registry(), {"share_price:sber"}
    )
    assert prepared.drivers == {"price": "share_price:sber"}


def test_a_position_without_a_served_driver_is_not_calculated_and_names_the_code():
    batch = run([equity()], paths(), drivers=registry(), served={"share_price:gazp"})
    result = batch.results[0]
    assert result.status == "invalid"
    assert errors(result) == ["driver_missing:share_price"]
    assert result.value_change == [] and result.metrics == {}
    message = "; ".join(item.message for item in result.diagnostics)
    assert "share_price" in message and "secid SBER" in message


def test_a_driver_the_catalog_does_not_know_is_missing_the_same_way():
    batch = run([equity(codes={"secid": "NOWHERE"})], paths(), drivers=registry())
    assert errors(batch.results[0]) == ["driver_missing:share_price"]


def test_an_unknown_instrument_type_is_refused_with_its_name():
    batch = run(
        [equity(type="option")], paths(), drivers=registry(), served={"share_price:sber"}
    )
    result = batch.results[0]
    assert result.status == "invalid"
    assert errors(result) == ["no_model"]
    assert "option" in result.diagnostics[0].message


def test_terms_are_checked_against_the_model_schema():
    missing = run([bond(terms={})], paths(), drivers=registry(), served={"bond_spread:corp"})
    result = missing.results[0]
    assert result.status == "invalid"
    assert errors(result) == ["terms_invalid"]
    assert "duration" in result.diagnostics[0].message

    given = run([bond()], paths(), drivers=registry(), served={"bond_spread:corp"})
    assert given.results[0].status == "ok"


def test_a_currency_without_a_macro_path_is_refused():
    batch = run([equity(currency="GBP")], paths(), drivers=registry(), served={"share_price:sber"})
    assert errors(batch.results[0]) == ["fx_missing:GBP"]


def test_a_declared_macro_variable_the_paths_lack_is_named():
    paths_ = paths()
    without_rate = FixedPaths(
        macros={}, drivers=paths_.drivers, start=paths_.start, seed=paths_.seed
    )
    batch = run(
        [equity(currency="USD")], without_rate, drivers=registry(), served={"share_price:sber"}
    )
    assert errors(batch.results[0]) == ["macro_missing:ru.usdrub"]
    assert "ru.usdrub" in batch.results[0].diagnostics[0].message


class Broken(CalculationModel):
    """A model that returns a result of the wrong shape."""

    model_id = "broken.v1"
    version = "0"
    label = "broken"
    instrument_types = ("broken",)
    metrics = ("broken.pnl",)

    def requirements(self, position, options):
        return DataRequirements()

    def validate(self, prepared):
        return []

    def calculate(self, prepared, context):
        return CalculationResult(
            position_id=prepared.position.id,
            instrument_type=prepared.position.type,
            model_id=self.model_id,
            model_version=self.version,
            status="ok",
            base_currency=prepared.base_currency,
            value_current=prepared.value_current,
            value_change=[0.0],
            metrics={"broken.pnl": {"unit": money(prepared.base_currency), "value": 0.0}},
        )


def test_a_result_of_the_wrong_shape_is_invalid():
    catalog = Catalog((Broken(),))
    position = equity(id="broken", type="broken")
    batch = run([position], paths(), drivers=registry(), catalog=catalog)
    result = batch.results[0]
    assert result.status == "invalid"
    assert errors(result) == ["output:value_change"]
    assert "1 realisations" in result.diagnostics[0].message


def test_a_duplicate_position_id_is_refused():
    with pytest.raises(ValueError, match="share one id"):
        run([equity(), equity()], paths(), drivers=registry())


class TakesNoTerms(CalculationModel):
    """A model that declares no terms schema."""

    model_id = "bare.v1"
    version = "0"
    label = "bare"
    instrument_types = ("bare",)
    terms_schema = None

    def requirements(self, position, options):
        return DataRequirements()

    def validate(self, prepared):
        return []

    def calculate(self, prepared, context):
        raise AssertionError("a position with terms is not calculated")


class RepeatedKeys(CalculationModel):
    """A model whose requirements name one key twice."""

    model_id = "repeated.v1"
    version = "0"
    label = "repeated"
    instrument_types = ("repeated",)

    def requirements(self, position, options):
        return DataRequirements(
            drivers=[
                DriverRequirement(key="price", quantity="share_price"),
                DriverRequirement(key="price", quantity="bond_spread"),
            ]
        )

    def validate(self, prepared):
        return []

    def calculate(self, prepared, context):
        raise AssertionError("a position with broken requirements is not calculated")


def test_terms_of_a_model_that_takes_none_are_refused():
    catalog = Catalog((TakesNoTerms(),))
    batch = run(
        [equity(id="bare", type="bare", terms={"x": 1})],
        paths(),
        drivers=registry(),
        catalog=catalog,
    )
    result = batch.results[0]
    assert errors(result) == ["terms_invalid"]
    assert "takes no terms" in result.diagnostics[0].message


def test_two_requirements_with_one_key_are_refused():
    catalog = Catalog((RepeatedKeys(),))
    batch = run(
        [equity(id="rep", type="repeated")], paths(), drivers=registry(), catalog=catalog
    )
    result = batch.results[0]
    assert errors(result) == ["requirements_failed"]
    assert "share the key price" in result.diagnostics[0].message
