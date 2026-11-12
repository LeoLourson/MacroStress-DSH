"""The BVAR as a macro model plugin: estimate, load, simulate and the constraint law.

The economics lives in ``estimate``, ``sv``, ``var``, ``identify`` and ``forecast``; this
module is the adapter to the model SDK. ``estimate`` builds the panel known on the calculation
date, draws the posterior, identifies the shocks of the sign table and returns the metadata
and arrays of the version. ``load`` rebuilds the posterior of a saved version. ``simulate``
and ``constraint_law`` hand the platform's neutral ``ScenarioPlan`` to the forecast module
as it is.
"""

from datetime import date

from mst.macro.panel import build_panel
from mst.model_sdk.contracts import (
    ConstraintLaw,
    MacroEstimate,
    MacroEstimateContext,
    MacroModel,
    MacroSimulation,
    ScenarioPlan,
)

from .estimate import Posterior, dummy_columns, estimate
from .forecast import constraint_law as constraint_law_bvar
from .forecast import simulate as simulate_bvar
from .identify import identify
from .signs import load_signs
from .spec import Spec, load_spec

#: The arrays of a version, in the order the posterior holds them.
ARRAYS = ("const", "lags", "dummy", "chol", "logvar", "volvol", "dof", "history", "rotation")
JUMPS = ("jump_prob", "jump_scale")
"""The arrays of the jump mixture; a version estimated before it has none and does not jump."""


def _dummies(spec: Spec, meta: dict, arrays: dict) -> list:
    """The dummy regressors a saved version was estimated with.

    With differences a version also reads the changes of the spec's dummies; a version
    estimated before they were added holds a coefficient for the spec's own dummies only.
    """
    columns = dummy_columns(spec, date.fromisoformat(meta["months"][-1]))
    if arrays["dummy"].shape[-1] != len(columns):
        return list(spec.dummies)
    return columns


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
    arrays = {name: getattr(model, name) for name in (*ARRAYS, *JUMPS)}
    arrays |= {"panel_y": panel.y, "panel_approximate": panel.approximate}
    return MacroEstimate(meta=meta, arrays=arrays)


class BvarModel(MacroModel):
    """The BVAR with stochastic volatility, sign restrictions and Student tails."""

    model_id = "bvar.sv"
    version = "0.1.0"
    latents = ("volatility_path",)
    label = "BVAR со стохастической волатильностью"
    variables: tuple[str, ...] = ()
    description = (
        "Векторная авторегрессия с Minnesota-приором, блочной экзогенностью, стохастической "
        "волатильностью и стьюдентовскими хвостами; шоки идентифицированы знаками signs.toml."
    )

    def estimate(self, context: MacroEstimateContext) -> MacroEstimate:
        spec = load_spec()
        seasonal = [v.id for v in context.registry.variables if v.seasonal]
        panel = build_panel(
            context.store,
            context.registry,
            start=spec.start,
            as_of=context.as_of,
            seasonal=seasonal,
        )
        # A model estimated on the data known at a date cannot know a regime change after
        # the last month of its history: the dummies of later months are left out.
        spec = spec.model_copy(
            update={"dummies": [d for d in spec.dummies if d.month <= panel.months[-1]]}
        )
        posterior = estimate(panel, spec)
        model = identify(
            posterior,
            load_signs().shocks,
            sign_lag=spec.sign_lag,
            tries=spec.rotation_tries,
            seed=spec.sampler.seed,
            rho_max=spec.rho_max,
        )
        return parts(model, spec, panel, context.as_of)

    def load(self, meta: dict, arrays: dict) -> Posterior:
        # A pre-plugin version recorded the seasonal variables in its spec; the panel build owns
        # them now, so the stored key is dropped instead of being re-added to Spec.
        stored = dict(meta["spec"])
        stored.pop("seasonal", None)
        spec = Spec.model_validate(stored)
        return Posterior(
            ids=meta["ids"],
            n_global=meta["n_global"],
            months=[date.fromisoformat(m) for m in meta["months"]],
            # A version with differences keeps its levels form, one lag longer than the spec.
            lags_count=arrays["lags"].shape[1],
            dummies=_dummies(spec, meta, arrays),
            diagnostics=meta["diagnostics"],
            shocks=meta["shocks"],
            **{key: arrays[key] for key in ARRAYS},
            **{key: arrays[key] for key in JUMPS if key in arrays},
        )

    def simulate(self, model, plan: ScenarioPlan, *, draws: int, seed: int) -> MacroSimulation:
        return simulate_bvar(model, plan, draws=draws, seed=seed)

    def constraint_law(
        self, model, plan: ScenarioPlan, *, draws: int, seed: int
    ) -> ConstraintLaw | None:
        return constraint_law_bvar(model, plan, draws=draws, seed=seed)
