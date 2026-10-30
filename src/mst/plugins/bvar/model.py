"""The BVAR as a macro model plugin: estimate, load, simulate, severity and the episode check.

The economics lives in ``estimate``, ``sv``, ``var``, ``identify`` and ``forecast``; this
module is the adapter to the model SDK. ``estimate`` builds the panel known on the calculation
date, draws the posterior, identifies the shocks of the sign table and returns the metadata
and arrays of the version. ``load`` rebuilds the posterior of a saved version. ``simulate``
and ``severity`` convert the platform's neutral plan into the scenario contract the forecast
module works with.
"""

from datetime import date
from pathlib import Path

from mst.macro.panel import build_panel
from mst.model_sdk.contracts import (
    MacroEstimate,
    MacroEstimateContext,
    MacroModel,
    MacroSimulation,
    ScenarioPlan,
    Severity,
)
from mst.scenario.contracts import Scenario

from .estimate import Posterior, estimate
from .forecast import severity as severity_bvar
from .forecast import simulate as simulate_bvar
from .identify import identify
from .signs import load_signs
from .spec import Spec, load_spec

#: The arrays of a version, in the order the posterior holds them.
ARRAYS = ("const", "lags", "dummy", "chol", "logvar", "volvol", "dof", "history", "rotation")


def parts(model: Posterior, spec: Spec, panel, as_of: date) -> MacroEstimate:
    """The metadata and arrays of the version of an identified posterior."""
    if model.rotation is None:
        raise ValueError("only an identified posterior is a model version")
    meta = {
        "as_of": as_of.isoformat(),
        "ids": model.ids,
        "n_global": model.n_global,
        "months": [m.isoformat() for m in model.months],
        "shocks": model.shocks,
        "diagnostics": model.diagnostics,
        "spec": spec.model_dump(mode="json"),
    }
    arrays = {name: getattr(model, name) for name in ARRAYS}
    arrays |= {"panel_y": panel.y, "panel_approximate": panel.approximate}
    return MacroEstimate(meta=meta, arrays=arrays)


def to_scenario(plan: ScenarioPlan) -> Scenario:
    """The platform's neutral plan as the scenario contract the forecast module reads."""
    return Scenario.model_validate(
        {
            "narrative": "plan",
            "horizon": plan.horizon,
            "constraints": [
                {
                    "terms": [
                        {"variable": term.variable, "month": term.month, "weight": term.weight}
                        for term in constraint.terms
                    ],
                    "low": constraint.low,
                    "high": constraint.high,
                }
                for constraint in plan.constraints
            ],
            "driving_shocks": list(plan.driving_shocks),
        }
    )


class BvarModel(MacroModel):
    """The BVAR with stochastic volatility, sign restrictions and Student tails."""

    model_id = "bvar.sv"
    version = "0.1.0"
    label = "BVAR со стохастической волатильностью"
    variables: tuple[str, ...] = ()
    description = (
        "Векторная авторегрессия с Minnesota-приором, блочной экзогенностью, стохастической "
        "волатильностью и стьюдентовскими хвостами; шоки идентифицированы знаками signs.toml."
    )

    def estimate(self, context: MacroEstimateContext) -> MacroEstimate:
        spec = load_spec()
        panel = build_panel(
            context.store,
            context.registry,
            start=spec.start,
            as_of=context.as_of,
            seasonal=spec.seasonal,
        )
        posterior = estimate(panel, spec)
        model = identify(
            posterior,
            load_signs().shocks,
            sign_lag=spec.sign_lag,
            tries=spec.rotation_tries,
            seed=spec.sampler.seed,
        )
        return parts(model, spec, panel, context.as_of)

    def load(self, meta: dict, arrays: dict) -> Posterior:
        spec = Spec.model_validate(meta["spec"])
        return Posterior(
            ids=meta["ids"],
            n_global=meta["n_global"],
            months=[date.fromisoformat(m) for m in meta["months"]],
            lags_count=spec.lags,
            dummies=list(spec.dummies),
            diagnostics=meta["diagnostics"],
            shocks=meta["shocks"],
            **{key: arrays[key] for key in ARRAYS},
        )

    def simulate(self, model, plan: ScenarioPlan, *, draws: int, seed: int) -> MacroSimulation:
        return simulate_bvar(model, to_scenario(plan), draws=draws, seed=seed)

    def severity(self, model, plan: ScenarioPlan, *, draws: int, seed: int) -> Severity | None:
        return severity_bvar(model, to_scenario(plan), draws=draws, seed=seed)

    def backtest(
        self, *, store, models: Path, corpus: Path, as_of: date, only=(), details=None
    ) -> int:
        """The episode check of the BVAR: coverage of the unconstrained variables by episode."""
        from .backtest import run

        return run(
            self, store=store, models=models, corpus=corpus, as_of=as_of,
            only=only, details=details,
        )
