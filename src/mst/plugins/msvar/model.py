"""The MS-VAR as a macro model plugin: estimate, load, simulate and the constraint law.

The economics lives in ``estimate``, ``identify`` and ``forecast``; this module is the
adapter to the model SDK. ``estimate`` builds the panel known on the calculation date,
draws the posterior, identifies the shocks of the sign table in every regime and returns
the metadata and arrays of the version. ``load`` rebuilds the posterior of a saved version.
``simulate`` and ``constraint_law`` read the platform's neutral ``ScenarioPlan`` directly:
the regime element of the event arrives as a driving-shock name the same way a structural
shock does, so the plugin never sees the scenario contracts.
"""

from datetime import date

import numpy as np

from mst.macro.panel import build_panel
from mst.model_sdk.contracts import (
    ConstraintLaw,
    MacroEstimate,
    MacroEstimateContext,
    MacroModel,
    MacroSimulation,
    ScenarioPlan,
)

from .estimate import Posterior, estimate
from .forecast import constraint_law as constraint_law_msvar
from .forecast import simulate as simulate_msvar
from .identify import identify
from .signs import load_signs
from .spec import Spec, load_spec

#: The arrays of a version, in the order the posterior holds them.
ARRAYS = ("const", "lags", "chol", "logvar", "transition", "states", "history", "rotation")


def parts(model: Posterior, spec: Spec, panel, as_of: date) -> MacroEstimate:
    """The metadata and arrays of the version of an identified posterior.

    The ``shocks`` the platform checks a scenario's driving shocks against are the economic
    shocks and the regime elements together — the ``J`` set of the amended scheme — while
    ``regimes`` names the regime elements alone, so ``load`` can tell them apart.
    """
    if model.rotation is None:
        raise ValueError("only an identified posterior is a model version")
    meta = {
        "as_of": as_of.isoformat(),
        "ids": model.ids,
        "n_global": model.n_global,
        "months": [m.isoformat() for m in model.months],
        "shocks": model.shocks + model.regime_ids,
        "regimes": model.regime_ids,
        "diagnostics": model.diagnostics,
        "spec": spec.model_dump(mode="json"),
    }
    arrays = {name: getattr(model, name) for name in ARRAYS}
    arrays |= {"panel_y": panel.y, "panel_approximate": panel.approximate}
    return MacroEstimate(meta=meta, arrays=arrays)


class MSVarModel(MacroModel):
    """The Markov-switching VAR with sign restrictions inside every regime."""

    model_id = "msvar.markov"
    version = "0.1.0"
    label = "MS-VAR с марковскими переключениями"
    variables: tuple[str, ...] = ()
    latents = ("regime_path",)
    description = (
        "Векторная авторегрессия, у которой константа, лаговые коэффициенты и ковариация "
        "ошибок переключаются по марковскому режиму; шоки идентифицированы знаками "
        "signs.toml внутри каждого режима, и режимные элементы события адресуются "
        "сценарием наравне с шоками."
    )

    def estimate(self, context: MacroEstimateContext) -> MacroEstimate:
        spec = load_spec()
        signs = load_signs()
        seasonal = [v.id for v in context.registry.variables if v.seasonal]
        panel = build_panel(
            context.store,
            context.registry,
            start=spec.start,
            as_of=context.as_of,
            seasonal=seasonal,
        )
        posterior = estimate(panel, spec, len(signs.regimes))
        model = identify(
            posterior,
            signs,
            sign_lag=spec.sign_lag,
            tries=spec.rotation_tries,
            seed=spec.sampler.seed,
        )
        return parts(model, spec, panel, context.as_of)

    def load(self, meta: dict, arrays: dict) -> Posterior:
        spec = Spec.model_validate(meta["spec"])
        regimes = list(meta["regimes"])
        # The J set holds the economic shocks and the regime elements; only the regime
        # elements are regimes, the rest are the columns of B.
        shocks = [s for s in meta["shocks"] if s not in set(regimes)]
        return Posterior(
            ids=meta["ids"],
            n_global=meta["n_global"],
            months=[date.fromisoformat(m) for m in meta["months"]],
            lags_count=spec.lags,
            n_regimes=len(regimes),
            shocks=shocks,
            regime_ids=regimes,
            diagnostics=meta["diagnostics"],
            **{key: arrays[key] for key in ARRAYS},
        )

    def simulate(self, model, plan: ScenarioPlan, *, draws: int, seed: int) -> MacroSimulation:
        return simulate_msvar(model, plan, draws=draws, seed=seed)

    def constraint_law(
        self, model, plan: ScenarioPlan, *, draws: int, seed: int
    ) -> ConstraintLaw | None:
        return constraint_law_msvar(model, plan, draws=draws, seed=seed)
