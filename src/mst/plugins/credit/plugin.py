"""The credit plugin: its declaration and the demonstration case of the overdue-share model."""

from datetime import date

import numpy as np

from ...plugin_sdk import DemoCase, FixedPaths, Plugin, Position
from .model import MODEL

DEMO = FixedPaths(
    drivers={"overdue_share:mining": np.array([[0.02, 0.03, 0.04], [0.02, 0.01, 0.005]])},
    start=date(2026, 9, 1),
)

plugin = Plugin(
    plugin_id="credit",
    version="1.0.0",
    sdk=">=1.0,<2.0",
    label="Кредит или депозит",
    description="Кредит или депозит на путях доли просроченной задолженности отрасли.",
    models=(MODEL,),
    demo=(
        DemoCase(
            name="mining",
            position=Position(
                id="mining",
                name="Кредит добывающей отрасли",
                type="loan",
                notional=10.0,
                codes={"okved": "05"},
                terms={"lgd": 0.5},
            ),
            paths=DEMO,
            drivers={"overdue_share": "overdue_share:mining"},
        ),
    ),
)
