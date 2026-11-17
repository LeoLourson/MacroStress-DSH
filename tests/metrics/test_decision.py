"""mst.metrics.decision: the replacement rule of a candidate against the incumbent."""

import numpy as np
import pytest

from mst.metrics.decision import compare


def rows(scale, *, units=12, months=3, seed=0):
    """Episode rows whose model CRPS is ``scale`` times a noisy common level."""
    rng = np.random.default_rng(seed)
    found = []
    for u in range(units):
        for m in range(1, months + 1):
            level = rng.uniform(0.5, 1.5)
            found.append(
                {
                    "episode": f"E{u:02d}",
                    "month": m,
                    "quantity": "q",
                    "level": "macro",
                    "half": "validation",
                    "model": scale * level * rng.uniform(0.95, 1.05),
                }
            )
    return found


def decide(mine, theirs, *, trusted=True, incumbent_trusted=True):
    return compare(
        {"episodes": (mine, theirs, "episode", 1)},
        trusted=trusted,
        incumbent_trusted=incumbent_trusted,
        level="macro",
        seed=1,
    )


def test_an_untrusted_candidate_never_replaces():
    found = decide(rows(0.5), rows(1.0), trusted=False)
    assert found["checks"]["episodes"]["better"]
    assert not found["replaces"]


def test_a_trusted_candidate_replaces_an_untrusted_incumbent_whatever_the_scores():
    assert decide(rows(1.0, seed=2), rows(1.0), incumbent_trusted=False)["replaces"]


def test_between_trusted_versions_only_a_proven_gain_replaces():
    assert decide(rows(0.5), rows(1.0))["replaces"]
    same = decide(rows(1.0), rows(1.0))
    assert not same["checks"]["episodes"]["better"] and not same["replaces"]
    worse = decide(rows(2.0), rows(1.0))
    assert worse["checks"]["episodes"]["worse"] and not worse["replaces"]


def test_a_gain_on_one_check_with_a_loss_on_another_keeps_the_incumbent():
    found = compare(
        {
            "episodes": (rows(0.5), rows(1.0), "episode", 1),
            "grid": (rows(2.0, seed=3), rows(1.0, seed=3), "episode", 1),
        },
        trusted=True,
        incumbent_trusted=True,
        level="macro",
        seed=1,
    )
    assert found["checks"]["episodes"]["better"] and found["checks"]["grid"]["worse"]
    assert not found["replaces"]


def test_two_versions_on_different_cells_are_an_error():
    with pytest.raises(ValueError, match="do not share"):
        decide(rows(1.0, units=12), rows(1.0, units=11))
