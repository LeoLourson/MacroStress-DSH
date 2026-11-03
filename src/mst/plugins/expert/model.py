"""The expert model as a macro plugin: the adapter to the model SDK.

The economics lives in ``answers``, ``laws``, ``stream``, ``estimate``, ``forecast`` and
``subset``; this module declares the two variants — ``expert.elicited.rw`` and
``expert.elicited.market``, which differ only in the base path — and hands the SDK's
neutral structures to them. The estimate reads the agent's answers from the elicitation
folder of the workspace; the questions and the form of the answers come from
``answers.py``, so the agent cannot swap in its own model: it fills numbers into a fixed
form.
"""

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from mst.model_sdk.contracts import (
    ConstraintLaw,
    MacroEstimate,
    MacroEstimateContext,
    MacroModel,
    MacroSimulation,
    ScenarioPlan,
)

SHOCKS_PATH = Path(__file__).with_name("shocks.toml")
#: The name of the residual among the causes, the ``J`` set the platform checks against.
RESIDUAL = "residual"
#: The key of the market answer, the input the ``market`` variant adds.
MARKET = "market"


@dataclass(frozen=True)
class Shock:
    """One of the 18 shocks: its id, what it means and what one unit of its size is."""

    id: str
    meaning: str
    size_unit: str


def load_shocks(path: Path = SHOCKS_PATH) -> tuple[Shock, ...]:
    """The fixed set of the shocks, the same for every calculation date."""
    with path.open("rb") as file:
        rows = tomllib.load(file)["shocks"]
    return tuple(Shock(**row) for row in rows)


class ExpertModel(MacroModel):
    """The law of the macro variables from the expert's answers, in one of two variants.

    ``rw`` takes a random walk with drift for the base path of every variable; ``market``
    takes the market expectations the same elicitation writes where they exist. Everything
    else — the shock stream, the responses, the copula, the residual — is shared.
    """

    version = "0.1.0"
    label = "Макромодель по опросу эксперта"
    variables: tuple[str, ...] = ()

    def __init__(self, base: Literal["rw", "market"]) -> None:
        self.base = base
        self.model_id = f"expert.elicited.{base}"
        self.label = {
            "rw": "Опрос эксперта: случайное блуждание с дрейфом",
            "market": "Опрос эксперта: рыночные ожидания",
        }[base]
        self.description = (
            "Уровень 0: 18 шоков с частотами, законами размера и откликами из ответов "
            "агента, t-связка между откликами и гауссов остаток — разность ковариации "
            "приростов истории и вклада шоков. База — случайное блуждание с дрейфом"
            + (" и рыночными ожиданиями, где они есть." if base == "market" else ".")
        )

    def input_keys(self, as_of: date) -> tuple[str, ...]:
        keys = tuple(shock.id for shock in load_shocks())
        return (*keys, MARKET) if self.base == "market" else keys

    def estimate(self, context: MacroEstimateContext) -> MacroEstimate:
        from .estimate import estimate

        return estimate(context, base=self.base)

    def load(self, meta: dict, arrays: dict):
        from .estimate import load as load_model

        return load_model(meta, arrays)

    def simulate(
        self, model, plan: ScenarioPlan, *, draws: int, seed: int
    ) -> MacroSimulation:
        from .forecast import simulate

        return simulate(model, plan, draws=draws, seed=seed)

    def constraint_law(
        self, model, plan: ScenarioPlan, *, draws: int, seed: int
    ) -> ConstraintLaw | None:
        from .forecast import constraint_law

        return constraint_law(model, plan, draws=draws, seed=seed)

    def question(self, context: MacroEstimateContext, key: str) -> dict:
        from .answers import question

        if key not in self.input_keys(context.as_of):
            raise KeyError(f"the model declares no input {key!r}")
        return question(context, key)

    def check_input(self, context: MacroEstimateContext, key: str, payload: dict) -> dict:
        from .answers import check

        if key not in self.input_keys(context.as_of):
            raise KeyError(f"the model declares no input {key!r}")
        return check(context, key, payload)
