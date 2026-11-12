"""Translation data from the exact coordinates saved in model versions."""

import numpy as np

from ..macro.registry import load_registry
from .translate import QuantityData, TranslationData


def context_from_versions(macro, satellites=None):
    registry = load_registry()
    quantities = {v.id: QuantityData(v.id, v.transform) for v in registry.variables}
    history = {name: np.asarray(macro.panel.y[:, k]) for k, name in enumerate(macro.panel.ids)}
    if satellites is not None:
        for quantity, part in satellites.parts.items():
            transform = satellites.registry.quantity(quantity).transform
            values = satellites.arrays.get(f"{quantity}.d")
            for k, driver in enumerate(part["driver"]):
                if driver is None:
                    continue
                quantities[driver] = QuantityData(driver, transform)
                if values is not None:
                    history[driver] = np.asarray(values[:, k])
    last = {}
    for name, values in history.items():
        finite = values[np.isfinite(values)]
        if finite.size:
            last[name] = float(finite[-1])
    origin = macro.panel.months[-1].strftime("%Y-%m")
    return TranslationData(origin, quantities, last, history)
