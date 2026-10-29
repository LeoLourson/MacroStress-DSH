"""mst.engine.run: the batch of a run, without any input or output."""

from mst.engine import run
from mst.plugin_sdk import Position
from tests.engine.fixed import paths
from tests.engine.simulated import registry


def position(**changes) -> Position:
    values = {
        "id": "sber",
        "type": "common_stock",
        "notional": 10.0,
        "codes": {"secid": "SBER"},
    }
    values.update(changes)
    return Position.model_validate(values)


def test_the_batch_status_is_the_worst_status_of_its_positions():
    batch = run(
        [position(), position(id="option", type="option")],
        paths(),
        drivers=registry(),
        served={"share_price:sber"},
    )
    assert [result.status for result in batch.results] == ["ok", "invalid"]
    assert batch.status == "invalid"
    assert [result.position_id for result in batch.calculated] == ["sber"]


def test_a_position_without_a_model_keeps_its_exposure():
    batch = run([position(id="option", type="option", notional=7.0)], paths(), drivers=registry())
    result = batch.results[0]
    assert result.value_current == 7.0
    assert result.model_id == "none"


def test_the_results_keep_the_order_of_the_positions():
    batch = run(
        [position(id="second"), position(id="first")],
        paths(),
        drivers=registry(),
        served={"share_price:sber"},
    )
    assert [result.position_id for result in batch.results] == ["second", "first"]
