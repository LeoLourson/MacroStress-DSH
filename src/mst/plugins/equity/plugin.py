"""The equity plugin: its declaration and the demonstration case of the share model."""

from datetime import date

import numpy as np

from ...plugin_sdk import DemoCase, FixedPaths, Plugin, Position
from .model import MODEL

DEMO = FixedPaths(
    drivers={"share_price:sber": np.array([[100.0, 105.0, 110.0], [100.0, 95.0, 90.0]])},
    start=date(2026, 9, 1),
)

plugin = Plugin(
    plugin_id="equity",
    version="1.0.0",
    sdk=">=1.0,<2.0",
    label="Акция",
    description="Обыкновенная акция на путях драйвера её цены.",
    models=(MODEL,),
    demo=(
        DemoCase(
            name="sber",
            position=Position(
                id="sber",
                name="Сбербанк",
                type="common_stock",
                notional=10.0,
                codes={"secid": "SBER"},
            ),
            paths=DEMO,
            drivers={"share_price": "share_price:sber"},
        ),
    ),
)
