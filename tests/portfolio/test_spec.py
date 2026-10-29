"""mst.portfolio.spec: a portfolio is its positions and the currency they are valued in."""

import pytest
from pydantic import ValidationError

from mst.portfolio.spec import Portfolio


def position(**changes):
    values = {
        "id": "sber",
        "type": "common_stock",
        "notional": 10.0,
        "codes": {"secid": "SBER"},
    }
    values.update(changes)
    return values


def test_a_portfolio_needs_at_least_one_position():
    with pytest.raises(ValidationError, match="at least 1"):
        Portfolio.model_validate({"positions": []})


def test_two_positions_cannot_share_an_id():
    with pytest.raises(ValidationError, match="share one id"):
        Portfolio.model_validate({"positions": [position(), position()]})


def test_the_base_currency_is_an_upper_case_code():
    with pytest.raises(ValidationError):
        Portfolio.model_validate({"base_currency": "rub", "positions": [position()]})
    assert Portfolio.model_validate({"positions": [position()]}).base_currency == "RUB"


def test_a_position_names_its_type_and_a_positive_notional():
    with pytest.raises(ValidationError):
        Portfolio.model_validate({"positions": [position(notional=0.0)]})
    with pytest.raises(ValidationError):
        Portfolio.model_validate({"positions": [position(type=None)]})
