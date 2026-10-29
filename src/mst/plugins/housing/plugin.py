"""The housing plugin: its declaration and the demonstration case of the housing-price model."""

from datetime import date

import numpy as np

from ...plugin_sdk import DemoCase, FixedPaths, Plugin, Position
from .model import MODEL

DEMO = FixedPaths(
    drivers={
        "housing_price_primary:belgorod": np.array(
            [[100000.0, 105000.0, 110000.0], [100000.0, 95000.0, 90000.0]]
        )
    },
    start=date(2026, 9, 1),
)

plugin = Plugin(
    plugin_id="housing",
    version="1.0.0",
    sdk=">=1.0,<2.0",
    label="Недвижимость",
    description="Квартира на путях цены квадратного метра региона.",
    models=(MODEL,),
    demo=(
        DemoCase(
            name="belgorod",
            position=Position(
                id="flat",
                name="Квартира в Белгородской области",
                type="real_estate",
                notional=5.0,
                codes={"okato": "14000000000"},
                terms={"market": "primary"},
            ),
            paths=DEMO,
            drivers={"housing_price_primary": "housing_price_primary:belgorod"},
        ),
    ),
)
