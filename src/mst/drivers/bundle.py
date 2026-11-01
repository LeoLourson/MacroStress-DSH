"""The satellite bundle: one version, many models, one per quantity.

A satellite version used to be the estimate of one model over every quantity. Now it is a
bundle: each quantity is estimated by the satellite model the platform assigned to it — the
one that declares it, or the one an explicit assignment names — and the version records which
model estimated which quantity. ``load`` groups the parts by model and hands each plugin its
own, so only the plugin that estimated a part reads it back.

The platform's part in a bundle is small: it assembles the parts (``assemble``), correlates
the residuals every model returned, saves and loads the version, answers which drivers are
served (``served``, from the checks the models recorded) and simulates the drivers by
dispatching to each model with its own drivers, its own impacts and its slice of the
correlation matrix. A driver whose simulated path leaves the range of the history it was
estimated on — the convention is an array named ``<quantity>.d`` — is marked
``outside_training_range`` when at least one percent of the paths do. The range is the one the
model recorded for the node in the part (``train_low`` and ``train_high``: the observed values
inside the node's own estimation window); a node without a range is not checked.

Every part names the model that estimated it, and ``load`` hands each plugin its own, so only
the plugin that estimated a part reads it back. The correlation matrix used to be imposed
inside every model: when a bundle's quantities are served by several models, the residuals of
drivers of different models were drawn independently, so a cross-model correlation measured by
``assemble`` was not applied to their joint paths. The platform draws the shocks itself (see
``simulate``), so that measured correlation reaches the joint paths.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from mst import versions
from mst.model_sdk.contracts import (
    BUNDLE_META_REQUIRED,
    PART_META_REQUIRED,
    AdditiveImpact,
    Check,
    DriverPaths,
    MacroSimulation,
    SatelliteEstimate,
    SatelliteModel,
    SatellitePart,
)

from .history import correlation
from .registry import Registry

OUTSIDE_TRAINING = "outside_training_range"
"""At least one percent of the paths leave the range of the driver's own history."""
IMPACT = "impact"
"""A direct impact of the scenario moves the path."""


@dataclass(frozen=True)
class SatelliteBundle:
    """A satellite version loaded back, with the model of every quantity in it."""

    macro: str
    as_of: date
    registry: Registry
    parts: dict[str, dict]
    """The recorded meta of every quantity's part."""
    quantity_model: dict[str, str]
    """The satellite model id that estimated each quantity."""
    models: dict[str, object]
    """What each plugin's ``load`` returned, by model id."""
    plugins: dict[str, SatelliteModel]
    checks: dict[str, dict[int, Check]]
    """Per quantity, the check of every driver."""
    correlated: list[str]
    correlation: np.ndarray
    arrays: dict[str, np.ndarray]

    def node(self, driver: str) -> tuple[str, int]:
        """``(quantity, index)`` of a driver among its quantity's nodes."""
        quantity = driver.split(":", 1)[0]
        part = self.parts.get(quantity)
        if part is None or driver not in part["driver"]:
            raise KeyError(driver)
        return quantity, part["driver"].index(driver)

    def quantity_of(self, driver: str) -> str:
        """The quantity a driver belongs to."""
        return self.node(driver)[0]

    def served(self) -> list[str]:
        """Every driver whose check passed, in a stable order."""
        return [
            part["driver"][node]
            for quantity in self.registry.quantities
            if (part := self.parts.get(quantity.id))
            for node, item in sorted(self.checks.get(quantity.id, {}).items())
            if item.served and part["driver"][node] is not None
        ]

    def part_months_end(self) -> date | None:
        """The last month of the first quantity's part; None in an empty bundle."""
        for part in self.parts.values():
            return date.fromisoformat(part["months"][-1])
        return None

    def simulate(
        self,
        macro: MacroSimulation,
        drivers: list[str],
        impacts: Iterable[AdditiveImpact] = (),
        *,
        seed: int,
    ) -> DriverPaths:
        """The paths of ``drivers`` on ``macro``, in the model scale, by their own models.

        The platform draws the joint shocks of every model itself: one standard normal draw
        per repetition from the stream ``[seed, 0, r]`` over every correlated driver of the
        request, multiplied by the Cholesky factor of the full matrix ``assemble`` measured,
        so the cross-model correlation reaches the joint paths. Each model receives its own
        slice of those shocks, in the order of the drivers it was asked for. The platform
        draws ``horizon + max(len(part["months"])) - 1`` rows: a model that starts from its
        last observed month ``last >= 0`` needs ``horizon + len(its months) - 1 - last``
        steps, which does not exceed that count, and reads the first rows it needs.
        """
        impacts = list(impacts)
        served = set(self.served())
        for driver in drivers:
            if driver not in served:
                quantity, node = self.node(driver)
                raise ValueError(
                    f"{driver} is not served: {self.checks[quantity][node].reason}"
                )
        for impact in impacts:
            if impact.target not in drivers:
                raise ValueError(
                    f"an impact on {impact.target}, which the run does not simulate"
                )
        by_model: dict[str, list[str]] = {}
        for driver in drivers:
            by_model.setdefault(self.quantity_model[self.node(driver)[0]], []).append(driver)
        draws, horizon = macro.paths.shape[0], macro.paths.shape[1] - 1
        platform = self._platform_shocks(macro, drivers, by_model, horizon, seed)
        paths = np.empty((draws, horizon + 1, len(drivers)))
        flags: dict[str, set[str]] = {driver: set() for driver in drivers}
        # Streams start at 1: the platform's own shock stream is 0, so a model's stream never
        # collides with it.
        for stream, model_id in enumerate(sorted(by_model), start=1):
            found = self.plugins[model_id].simulate(
                self.models[model_id],
                macro,
                by_model[model_id],
                [item for item in impacts if item.target in by_model[model_id]],
                correlation=(self.correlated, self.correlation),
                shocks=platform[model_id],
                seed=seed,
                stream=stream,
            )
            if sorted(found.ids) != sorted(by_model[model_id]):
                raise ValueError(
                    f"{model_id} returned the drivers {', '.join(found.ids) or 'none'}; "
                    f"the run asked for {', '.join(by_model[model_id])}"
                )
            for k, driver in enumerate(found.ids):
                column = drivers.index(driver)
                paths[:, :, column] = found.paths[:, :, k]
                flags[driver] |= set(found.flags[driver])
        for column, driver in enumerate(drivers):
            flags[driver] |= _training_range_flags(self, driver, paths[:, :, column])
        return DriverPaths(
            ids=list(drivers),
            paths=paths,
            flags={driver: frozenset(flags[driver]) for driver in drivers},
        )

    def _platform_shocks(
        self,
        macro: MacroSimulation,
        drivers: list[str],
        by_model: dict[str, list[str]],
        horizon: int,
        seed: int,
    ) -> dict[str, np.ndarray]:
        """The standardized residuals of every model's correlated drivers, drawn once.

        ``(R, rows, k)`` per model, in the order of that model's ``drivers`` argument, where
        ``rows`` is the number the platform supplies and ``k`` counts only the drivers with a
        history of their own: the matrix was measured on those, and a driver without history
        stays on the model's own independent noise. The full matrix is positive definite by
        construction; a matrix edited outside ``assemble`` that is not is refused here.
        """
        positions = {driver: column for column, driver in enumerate(self.correlated)}
        requested = [driver for driver in drivers if driver in positions]
        if requested:
            picked = [positions[driver] for driver in requested]
            matrix = self.correlation[np.ix_(picked, picked)]
            try:
                cholesky = np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    "the correlation matrix of the bundle is not positive definite: the "
                    "platform cannot draw the joint shocks of its models"
                ) from error
        else:
            cholesky = np.zeros((0, 0))
        longest = max(
            (len(part.get("months", ())) for part in self.parts.values()), default=0
        )
        rows = horizon + max(longest - 1, 0)
        draws = macro.paths.shape[0]
        shocks = np.empty((draws, rows, len(requested)))
        for repetition in range(draws):
            rng = np.random.default_rng([seed, 0, repetition])
            shocks[repetition] = rng.standard_normal((rows, len(requested))) @ cholesky.T
        column = {driver: k for k, driver in enumerate(requested)}
        return {
            model_id: shocks[:, :, [column[driver] for driver in own if driver in column]]
            for model_id, own in by_model.items()
        }


def assemble(
    registry: Registry,
    macro: str,
    as_of: date,
    estimates: Mapping[str, SatelliteEstimate],
    *,
    unserved: Iterable[str] = (),
) -> tuple[dict, dict[str, np.ndarray]]:
    """The meta and arrays of a bundle from the estimates of its models.

    The residuals every model returned are correlated in the order of the registry's
    quantities and the node order inside each part, so the same estimates give the same
    matrix whatever model estimated what. The platform writes ``meta["model"]`` of every part
    itself: a part that carries a different model id is refused, so provenance cannot be
    spoofed.
    """
    parts: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    series: dict[str, np.ndarray] = {}
    owners: dict[str, SatelliteEstimate] = {}
    for model_id, estimate in estimates.items():
        for quantity, part in estimate.parts.items():
            absent = [
                key for key in PART_META_REQUIRED if key != "model" and key not in part.meta
            ]
            if absent:
                raise ValueError(_missing_keys(quantity, absent))
            named = part.meta.get("model")
            if named is not None and named != model_id:
                raise ValueError(
                    f"the part of {quantity} names model {named!r}, but {model_id!r} "
                    "estimated it: the platform writes the model id itself"
                )
            parts[quantity] = {**part.meta, "model": model_id}
            owners[quantity] = estimate
            for name, value in part.arrays.items():
                arrays[f"{quantity}.{name}"] = value
    correlated: list[str] = []
    for quantity in registry.quantities:
        part = parts.get(quantity.id)
        if part is None:
            continue
        estimate = owners[quantity.id]
        for node, driver in enumerate(part["driver"]):
            if driver is None or part["rows"][node] <= 0:
                continue
            correlated.append(driver)
            series[driver] = estimate.residuals[driver]
    meta = {
        "macro": macro,
        "as_of": as_of.isoformat(),
        "registry": registry.model_dump(mode="json", by_alias=True),
        "quantities": parts,
        "correlated": correlated,
        "unserved": sorted(unserved),
    }
    arrays["correlation"] = correlation(list(series.values())) if series else np.zeros((0, 0))
    return meta, arrays


def save(meta: dict, arrays: dict, root: Path) -> str:
    """Write a bundle version unless it exists; returns its hash."""
    missing = [key for key in BUNDLE_META_REQUIRED if key not in meta]
    if missing:
        raise ValueError(
            f"a satellite version must record {', '.join(BUNDLE_META_REQUIRED)}: "
            f"no {', '.join(missing)}"
        )
    for quantity, part in meta["quantities"].items():
        absent = [key for key in PART_META_REQUIRED if key not in part]
        if absent:
            raise ValueError(_missing_keys(quantity, absent))
    if "correlation" not in arrays:
        raise ValueError("a satellite version must carry the correlation matrix")
    return versions.save(Path(root) / "satellites", meta, arrays)


def _missing_keys(quantity: str, absent: list[str]) -> str:
    """The message of a part that does not hold every key the platform reads."""
    return (
        f"the part of {quantity} must record {', '.join(PART_META_REQUIRED)}: "
        f"no {', '.join(absent)}"
    )


def load(
    root: Path, name: str, models: Mapping[str, SatelliteModel]
) -> SatelliteBundle:
    """The bundle ``name`` loaded by the plugins that estimated its quantities.

    A part that names no model is refused: every part names the model that estimated it.
    """
    meta, arrays = versions.read(Path(root) / "satellites", name, "satellite version")
    registry = Registry.model_validate(meta["registry"])
    quantity_model = {}
    for quantity, part in meta["quantities"].items():
        recorded = part.get("model")
        if recorded is None:
            raise KeyError(
                f"satellite version {name}: the part of {quantity} names no model, so no "
                "installed model can read it; re-estimate the bundle"
            )
        quantity_model[quantity] = recorded
    missing = sorted(
        {model_id for model_id in quantity_model.values() if model_id not in models}
    )
    if missing:
        raise KeyError(
            f"satellite version {name} was estimated by {', '.join(missing)}, "
            "which is not installed"
        )
    parts_by_model: dict[str, dict[str, SatellitePart]] = {}
    for quantity, part in meta["quantities"].items():
        own = {
            key[len(quantity) + 1 :]: value
            for key, value in arrays.items()
            if key.startswith(quantity + ".")
        }
        parts_by_model.setdefault(quantity_model[quantity], {})[quantity] = SatellitePart(
            meta=_part_meta(part), arrays=own
        )
    loaded = {
        model_id: models[model_id].load(parts)
        for model_id, parts in parts_by_model.items()
    }
    checks = {
        quantity: {int(node): Check(**row) for node, row in part["checks"].items()}
        for quantity, part in meta["quantities"].items()
    }
    return SatelliteBundle(
        macro=meta["macro"],
        as_of=date.fromisoformat(meta["as_of"]),
        registry=registry,
        parts=meta["quantities"],
        quantity_model=quantity_model,
        models=loaded,
        plugins={model_id: models[model_id] for model_id in parts_by_model},
        checks=checks,
        correlated=meta["correlated"],
        correlation=arrays["correlation"],
        arrays=arrays,
    )


def _part_meta(part: dict) -> dict:
    """The part meta as the platform hands it to the model that estimated it."""
    return part


def catalogue(root: Path) -> list[dict]:
    """Every saved satellite version, newest first, as its meta records it."""
    return [_summary(path) for path in versions.folders(Path(root) / "satellites")]


def describe(root: Path, name: str) -> dict:
    """One version: the drivers its checks served, and those they rejected with the reason."""
    meta, _ = versions.read(Path(root) / "satellites", name, "satellite version")
    served, rejected = [], []
    for part in meta["quantities"].values():
        for node, row in part["checks"].items():
            driver = part["driver"][int(node)]
            if row["served"]:
                served.append(driver)
            else:
                rejected.append(
                    {
                        "driver": driver,
                        "reason": row["reason"],
                        "satellite": row["satellite"],
                        "baseline": row["baseline"],
                        "forecasts": row["forecasts"],
                    }
                )
    return {
        "version": name,
        "macro": meta["macro"],
        "as_of": meta["as_of"],
        "models": {q: part["model"] for q, part in meta["quantities"].items()},
        "served": served,
        "rejected": rejected,
    }


def _summary(folder: Path) -> dict:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    checks = [
        row for part in meta["quantities"].values() for row in part["checks"].values()
    ]
    return {
        "version": folder.name,
        "macro": meta["macro"],
        "as_of": meta["as_of"],
        "models": sorted({part["model"] for part in meta["quantities"].values()}),
        "quantities": sorted(meta["quantities"]),
        "served": sum(1 for row in checks if row["served"]),
        "checked": len(checks),
    }


def _recorded_bounds(part: dict, node: int) -> tuple[float | None, float | None]:
    """The training bounds a part records for a node; ``(None, None)`` when it records none."""
    low, high = part.get("train_low"), part.get("train_high")
    if low is None or high is None or node >= len(low) or node >= len(high):
        return None, None
    return low[node], high[node]


def _training_range_flags(
    bundle: SatelliteBundle, driver: str, paths: np.ndarray
) -> set[str]:
    """``outside_training_range`` when one percent of the paths leave the trained range.

    The range is the one the part records for the node: the lowest and highest value the node
    was observed at inside its own estimation window. A node without a range — a part that
    records ``None`` for it — is not checked.
    """
    quantity, node = bundle.node(driver)
    key = f"{quantity}.d"
    if key not in bundle.arrays:
        return set()
    low, high = _recorded_bounds(bundle.parts.get(quantity, {}), node)
    if low is None or high is None:
        return set()
    outside = ((paths[:, 1:] < low) | (paths[:, 1:] > high)).any(axis=1)
    if outside.mean() >= 0.01:
        return {OUTSIDE_TRAINING}
    return set()
