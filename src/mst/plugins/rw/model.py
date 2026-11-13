"""Adapter of the joint Student walk to the macro model SDK."""

from dataclasses import dataclass
from datetime import date

import numpy as np

from mst.macro.panel import build_panel
from mst.model_sdk import (
    ConstraintLaw,
    MacroEstimate,
    MacroEstimateContext,
    MacroModel,
    MacroSimulation,
    ScenarioPlan,
    StructuralForm,
)

from .estimate import estimate
from .forecast import constraint_law, simulate
from .spec import load_spec

@dataclass(frozen=True)
class JointWalk:
    ids: list[str]
    n_global: int
    months: list[date]
    nu: float
    drift: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    start_mean: np.ndarray
    start_sd: np.ndarray
    persistence: np.ndarray
    intercept: np.ndarray
    scale_persistence: float
    scale_volatility: float
    diagnostics: dict


class RwModel(MacroModel):
    """No lags: joint Student monthly changes with a common crisis multiplier."""

    model_id = "rw.joint"
    version = "0.2.0"
    latents = ("scale_path",)
    label = "Joint random walk with heavy tails"
    description = (
        "Joint Student monthly changes with zero drift for rates and spreads; the scale is the "
        "sample covariance of the changes, the common multiplier may follow a persistent path, "
        "and interventions propagate through the full conditional law, without causal claims."
    )

    def estimate(self, context: MacroEstimateContext) -> MacroEstimate:
        spec = load_spec()
        spec.validate_ids([variable.id for variable in context.registry.variables])
        panel = build_panel(
            context.store,
            context.registry,
            start=spec.start,
            as_of=context.as_of,
            seasonal=[v.id for v in context.registry.variables if v.seasonal],
        )
        diagnostics, arrays, nu = estimate(panel, spec)
        return MacroEstimate(
            meta={
                "as_of": context.as_of.isoformat(),
                "ids": panel.ids,
                "n_global": panel.n_global,
                "months": [m.isoformat() for m in panel.months],
                "shocks": panel.ids,
                "nu": nu,
                "diagnostics": diagnostics,
                "spec": spec.model_dump(mode="json"),
                "structural_form": {"order_within_month": panel.ids, "time_recursion": True},
            },
            arrays=arrays | {"panel_y": panel.y, "panel_approximate": panel.approximate},
        )

    def load(self, meta: dict, arrays: dict) -> JointWalk:
        n = len(meta["ids"])
        scale_path = meta.get("diagnostics", {}).get("scale_path", {})
        return JointWalk(
            ids=meta["ids"],
            n_global=meta["n_global"],
            months=[date.fromisoformat(m) for m in meta["months"]],
            nu=meta["nu"],
            diagnostics=meta["diagnostics"],
            persistence=np.asarray(arrays.get("persistence", np.ones(n)), dtype=float),
            intercept=np.asarray(arrays.get("intercept", arrays["drift"]), dtype=float),
            scale_persistence=float(scale_path.get("persistence", 0.0)),
            scale_volatility=float(scale_path.get("volatility", 0.0)),
            drift=arrays["drift"],
            scale=arrays["scale"],
            weights=arrays["weights"],
            start_mean=arrays["start_mean"],
            start_sd=arrays["start_sd"],
        )

    def structural_form(self, model: JointWalk) -> StructuralForm:
        # Registry order is the base order; sets are moved first in each intervened month.
        return StructuralForm(tuple(model.ids), time_recursion=True)

    def simulate(self, model, plan: ScenarioPlan, *, draws: int, seed: int) -> MacroSimulation:
        return simulate(model, plan, draws=draws, seed=seed)

    def constraint_law(
        self, model, plan: ScenarioPlan, *, draws: int, seed: int
    ) -> ConstraintLaw | None:
        return constraint_law(model, plan, draws=draws, seed=seed)
