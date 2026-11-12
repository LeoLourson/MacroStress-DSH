"""Bitwise scenario baselines from stored BVAR and MS-VAR versions, not fitted fixtures."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from mst import plugins
from mst.drivers.bundle import SatelliteBundle
from mst.drivers.registry import load_registry
from mst.macro.version import load
from mst.scenario.build import build
from mst.scenario.context import context_from_versions
from mst.scenario.legacy import read_legacy
from mst.scenario.spec import Spec

GOLDEN = Path(__file__).with_name("golden")
REPOSITORY = Path(__file__).resolve().parents[2]
DRAWS, SEED = 256, 20261112
VERSIONS = {
    "bvar.sv": "51ead9204774f7ef0b4657c4e1a3997c77c584a902ebb6d306c68439977658f8",
    "msvar.markov": "ecaa07d09611885aa4f98de116ea4b870fc31caa34788dafea3998e2e9ed57ab",
}
CASES = ("empty", "shifts-only", "constraints-no-causes")


def models_root() -> Path:
    """Use external stored versions; an explicitly selected workspace must be complete."""
    explicit = os.environ.get("MACROSTRESS_WORKSPACE")
    root = (Path(explicit) if explicit else REPOSITORY / "workspace") / "models"
    missing = [name for name in VERSIONS.values() if not (root / "macro" / name).is_dir()]
    if missing:
        message = f"Golden baselines need stored macro versions in {root}: {', '.join(missing)}"
        if explicit:
            pytest.fail(message)
        pytest.skip(message)
    return root


def scenario_of(raw: dict, macro, satellites: SatelliteBundle) -> Spec:
    """Translate frozen legacy inputs using the version's exact panel coordinates."""
    return read_legacy(raw, context=context_from_versions(macro, satellites))


def empty_satellites(macro) -> SatelliteBundle:
    """Exercise the production builder without unrelated satellite models or drivers."""
    return SatelliteBundle(
        macro=macro.name,
        as_of=macro.as_of,
        registry=load_registry(),
        parts={},
        quantity_model={},
        models={},
        plugins={},
        checks={},
        correlated=[],
        correlation=np.empty((0, 0)),
        arrays={},
    )


def paths_of(case: str, macro):
    raw = json.loads((GOLDEN / f"{case}.json").read_text(encoding="utf-8"))
    satellites = empty_satellites(macro)
    return build(
        scenario_of(raw, macro, satellites),
        macro,
        satellites,
        draws=DRAWS,
        seed=SEED,
        horizon=raw["horizon"],
        inner=raw["inner"],
    )


def metadata_of(paths) -> str:
    """Canonical JSON preserves each float's round-trip value, including signed zero."""
    return json.dumps(
        {
            "ids": paths.ids,
            "months": [month.isoformat() for month in paths.months],
            "draws": paths.draws,
            "seed": paths.seed,
            "dist": None if paths.dist is None else asdict(paths.dist),
        },
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def baseline_stem(case: str, version: str) -> Path:
    return GOLDEN / f"{version[:8]}-{case}"


@pytest.fixture(scope="module", params=VERSIONS.items(), ids=VERSIONS)
def stored_macro(request):
    model_id, version = request.param
    macro = load(models_root(), version, plugins.registry().macro)
    assert macro.meta["model"] == model_id
    return macro


@pytest.mark.parametrize("case", CASES)
def test_stored_macro_paths_and_price_match_bitwise(stored_macro, case):
    stem = baseline_stem(case, stored_macro.name)
    paths = paths_of(case, stored_macro)
    assert paths.macro.shape == (DRAWS, 13, 23)
    assert paths.shadow is not None
    assert np.isfinite(paths.macro).all()
    assert np.isfinite(paths.shadow).all()
    assert paths.drivers.ids == []
    assert paths.drivers.paths.shape == (DRAWS, 13, 0)
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as expected:
        for name, actual in (("macro", paths.macro), ("shadow", paths.shadow)):
            saved = expected[name]
            assert actual.shape == saved.shape
            assert actual.dtype == saved.dtype
            assert actual.tobytes() == saved.tobytes(), f"{stem.name}: {name} differs bitwise"
    assert metadata_of(paths) == stem.with_suffix(".json").read_text(encoding="utf-8")
