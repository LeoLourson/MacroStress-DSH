"""Capture stage 0 once; refuse to replace any existing baseline."""

import platform
import sys

import numpy as np
import scipy

from mst import plugins
from mst.macro.version import load
from tests.scenario.test_golden import (
    CASES,
    VERSIONS,
    baseline_stem,
    metadata_of,
    models_root,
    paths_of,
)


def capture() -> None:
    targets = [
        baseline_stem(case, version).with_suffix(suffix)
        for version in VERSIONS.values()
        for case in CASES
        for suffix in (".npz", ".json")
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to replace golden baselines: {', '.join(existing)}")
    root = models_root()
    for model_id, version in VERSIONS.items():
        macro = load(root, version, plugins.registry().macro)
        assert macro.meta["model"] == model_id
        for case in CASES:
            paths = paths_of(case, macro)
            assert paths.shadow is not None
            assert np.isfinite(paths.macro).all() and np.isfinite(paths.shadow).all()
            metadata = metadata_of(paths)
            stem = baseline_stem(case, version)
            with stem.with_suffix(".npz").open("xb") as destination:
                np.savez_compressed(destination, macro=paths.macro, shadow=paths.shadow)
            with stem.with_suffix(".json").open("x", encoding="utf-8", newline="\n") as destination:
                destination.write(metadata)
            price = None if paths.dist is None else paths.dist.distance
            print(f"{model_id} {case}: {paths.macro.shape}, price={price!r}", flush=True)
    print(
        f"Captured on {platform.platform()}; Python {sys.version.split()[0]}; "
        f"NumPy {np.__version__}; SciPy {scipy.__version__}",
        flush=True,
    )


if __name__ == "__main__":
    capture()
