"""Adapter of the joint Student walk to the macro model SDK."""

import hashlib
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
from .spec import Spec, load_spec


@dataclass(frozen=True)
class JointWalk:
    ids: list[str]
    n_global: int
    months: list[date]
    nu: np.ndarray
    """The ``nu`` of every block's multiplier."""
    blocks: np.ndarray
    """The block of every variable's multiplier."""
    drift: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    start_mean: np.ndarray
    start_sd: np.ndarray
    persistence: np.ndarray
    intercept: np.ndarray
    momentum: np.ndarray
    previous: np.ndarray
    scale_persistence: np.ndarray
    scale_volatility: np.ndarray
    scale_state: np.ndarray
    """The deviation of every block's log scale from its mean in the last month of the panel."""
    diagnostics: dict
    tail: np.ndarray
    """The last months of the panel, from the latest one at which every variable was known."""


class RwModel(MacroModel):
    """No lags: joint Student monthly changes with a crisis multiplier for every block."""

    model_id = "rw.joint"
    version = "0.5.0"
    latents = ("scale_path",)
    label = "Joint random walk with heavy tails"
    description = (
        "Joint Student monthly changes with zero drift for rates and spreads; the scale is the "
        "sample covariance of the changes, every block of variables has its own multiplier, "
        "and interventions propagate through the full conditional law, without causal claims."
    )

    def __init__(self, *, spec: Spec | None = None):
        self._spec = spec.model_copy(deep=True) if spec is not None else None
        if self._spec is not None:
            digest = hashlib.sha256(self._spec.model_dump_json().encode()).hexdigest()[:16]
            self.version = f"{type(self).version}+spec.{digest}"

    def estimate(self, context: MacroEstimateContext) -> MacroEstimate:
        spec = self._spec if self._spec is not None else load_spec()
        spec.validate_ids(
            [*(variable.id for variable in context.registry.variables), *context.left_out]
        )
        panel = build_panel(
            context.store,
            context.registry,
            start=spec.start,
            as_of=context.as_of,
            seasonal=[v.id for v in context.registry.variables if v.seasonal],
        )
        diagnostics, arrays, nu = estimate(panel, spec, left_out=list(context.left_out))
        return MacroEstimate(
            meta={
                "as_of": context.as_of.isoformat(),
                "ids": panel.ids,
                "n_global": panel.n_global,
                "months": [m.isoformat() for m in panel.months],
                "shocks": panel.ids,
                "nu": nu.tolist(),
                "diagnostics": diagnostics,
                "spec": spec.model_dump(mode="json"),
                "structural_form": {"order_within_month": panel.ids, "time_recursion": True},
            },
            arrays=arrays | {"panel_y": panel.y, "panel_approximate": panel.approximate},
        )

    def load(self, meta: dict, arrays: dict) -> JointWalk:
        n = len(meta["ids"])
        nu = np.atleast_1d(np.asarray(meta["nu"], dtype=float))
        scale_path = meta.get("diagnostics", {}).get("scale_path") or [{}] * len(nu)
        return JointWalk(
            ids=meta["ids"],
            n_global=meta["n_global"],
            months=[date.fromisoformat(m) for m in meta["months"]],
            nu=nu,
            blocks=np.asarray(arrays["blocks"], dtype=int),
            diagnostics=meta["diagnostics"],
            persistence=np.asarray(arrays.get("persistence", np.ones(n)), dtype=float),
            intercept=np.asarray(arrays.get("intercept", arrays["drift"]), dtype=float),
            momentum=np.asarray(arrays.get("momentum", np.zeros(n)), dtype=float),
            previous=np.asarray(arrays.get("previous", np.full(n, np.nan)), dtype=float),
            scale_persistence=np.array([item.get("persistence", 0.0) for item in scale_path]),
            scale_volatility=np.array([item.get("volatility", 0.0) for item in scale_path]),
            scale_state=np.array([item.get("state", 0.0) for item in scale_path]),
            drift=arrays["drift"],
            scale=arrays["scale"],
            weights=arrays["weights"],
            start_mean=arrays["start_mean"],
            start_sd=arrays["start_sd"],
            tail=arrays["tail"],
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
