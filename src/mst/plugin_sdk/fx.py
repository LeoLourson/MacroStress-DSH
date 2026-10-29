"""The rate of a position's currency to the run's base currency, read from the macro paths.

A run values every position in the base currency of its portfolio. The rouble is the base the
macro model prices everything against: a dollar or a yuan position reads ``ru.usdrub`` or
``ru.cnyrub`` from the paths (``mst.macro.registry``). A pair the paths do not carry is refused
by name rather than valued at an invented rate.
"""

import numpy as np

from .model import Position
from .paths import Paths

#: The macro variables of the currencies the model prices against the rouble.
VARIABLES = {"USD": "ru.usdrub", "CNY": "ru.cnyrub"}


def exposure(position: Position, paths: Paths, base: str = "RUB") -> float:
    """The position's value at ``t = 0`` in millions of the base currency."""
    return position.notional * spot(paths, position.currency, base)


def pair(currency: str, base: str = "RUB") -> str | None:
    """The macro variable of the pair, or None: the same currency, or a pair with no path."""
    if currency == base:
        return None
    return VARIABLES.get(currency) if base == "RUB" else None


def variables(currency: str, base: str = "RUB") -> list[str]:
    """The macro variables a model declares so the platform checks the pair is in the paths."""
    name = pair(currency, base)
    return [name] if name else []


def variable(currency: str, base: str = "RUB") -> str | None:
    """The macro variable of ``currency`` to ``base``; a pair the model cannot price is refused."""
    name = pair(currency, base)
    if name is None and currency != base:
        raise ValueError(f"no macro path from {currency} to {base}: the model prices against RUB")
    return name


def spot(paths: Paths, currency: str, base: str = "RUB") -> float:
    """Roubles per unit of ``currency`` in the calculation month ``t = 0``."""
    name = variable(currency, base)
    return 1.0 if name is None else float(_path(paths, name)[0, 0])


def ratio(paths: Paths, currency: str, base: str = "RUB") -> np.ndarray:
    """``FX_H / FX_0`` of each repetition: how the currency moved by the horizon."""
    name = variable(currency, base)
    if name is None:
        return np.ones(paths.draws)
    path = _path(paths, name)
    return path[:, -1] / path[:, 0]


def _path(paths: Paths, name: str) -> np.ndarray:
    """The macro path of a currency pair, or a refusal that names it."""
    try:
        return np.asarray(paths.macro(name), dtype=float)
    except KeyError:
        raise ValueError(f"the paths carry no {name}") from None
