"""The bond plugin: its declaration and the demonstration case of the spread model."""

from datetime import date

import numpy as np

from ...plugin_sdk import DemoCase, FixedPaths, Plugin, Position
from .model import MODEL

DEMO = FixedPaths(
    drivers={"bond_spread:corp": np.array([[5.0, 6.0, 7.0], [5.0, 4.0, 3.0]])},
    start=date(2026, 9, 1),
)

plugin = Plugin(
    plugin_id="bond",
    version="1.0.0",
    sdk=">=1.0,<2.0",
    label="Облигация",
    description="Купонная облигация на путях спреда к кривой ОФЗ.",
    models=(MODEL,),
    demo=(
        DemoCase(
            name="corp",
            position=Position(
                id="corp",
                name="Корпоративная облигация",
                type="bond",
                notional=5.0,
                codes={"isin": "RU000A0JX0J2"},
                terms={"duration": 2.0},
            ),
            paths=DEMO,
            drivers={"bond_spread": "bond_spread:corp"},
        ),
    ),
)
