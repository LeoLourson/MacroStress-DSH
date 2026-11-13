from datetime import date

import pytest
from pydantic import ValidationError

from mst.plugins.rw.spec import Break, Spec, load_spec


def test_defaults_match_bundled_spec():
    spec = load_spec()
    assert spec == Spec()
    assert spec.start == date(2012, 1, 1)
    assert spec.nu_bounds == (2.5, 60.0)
    assert spec.em_tol == 1e-8
    assert spec.em_iterations == 500
    assert spec.min_changes == 3
    assert spec.reversion == Spec().reversion
    assert spec.scale_start == date(2015, 1, 1)
    assert spec.breaks == [
        Break(month=date(2021, 1, 1), variables=["ru.spread_aa", "ru.spread_a", "ru.spread_bbb"])
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unknown": 1},
        {"nu_bounds": (2, 60)},
        {"nu_bounds": (10, 5)},
        {"em_tol": 0},
        {"em_iterations": 0},
        {"em_iterations": 2.0},
        {"em_iterations": True},
        {"min_changes": "3"},
        {"seed": -1},
        {"seed": 2.0},
        {"seed": True},
        {"start": date(2012, 1, 2)},
        {"eig_floor": 0},
        {"eig_floor": 1},
        {"min_changes": 0},
        {"breaks": [{"month": "2021-01-02", "variables": ["x"]}]},
        {"drifts": ["x", "x"]},
        {"reversion": ["x", "x"]},
        {"scale_start": date(2015, 1, 2)},
    ],
)
def test_spec_rejects_invalid_settings(kwargs):
    with pytest.raises(ValidationError):
        Spec(**kwargs)


def test_validate_ids_is_explicit_and_names_unknown_variables():
    spec = Spec(drifts=["unknown"], breaks=[])
    with pytest.raises(ValueError, match="unknown"):
        spec.validate_ids(["ru.cpi"])
    spec.validate_ids(["unknown"])
    with pytest.raises(ValueError, match="unknown"):
        Spec(reversion=["unknown"], breaks=[]).validate_ids(["ru.cpi"])
    with pytest.raises(ValueError, match="typo"):
        Spec(drifts=[], breaks=[Break(month=date(2020, 1, 1), variables=["typo"])]).validate_ids(
            ["x"]
        )


def test_load_custom_spec_without_temporary_directory():
    # The repository file exercises TOML parsing without sandbox-sensitive tmp_path.
    assert load_spec("src/mst/plugins/rw/spec.toml").seed == 20261029
