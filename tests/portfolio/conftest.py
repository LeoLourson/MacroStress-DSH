"""Fixtures of the portfolio tests: the positions a saved portfolio holds."""

import copy

import pytest

from mst.portfolio import tool as portfolio_tool

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


@pytest.fixture
def positions():
    """A fresh copy of the positions, so a test can change one of them."""
    return copy.deepcopy(POSITIONS)


@pytest.fixture
def portfolio(workspace, positions):
    """The saved portfolio ``demo`` of the workspace."""
    portfolio_tool.save(workspace.portfolios, "demo", {"name": "Демо", "positions": positions})
    return "demo"
