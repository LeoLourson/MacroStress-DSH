"""The diagnose command and the macro gates on saved artefacts, nothing estimated."""

import json
import shutil
from datetime import date

import numpy as np
import pytest

from mst.macro import version as macro_version
from mst.models import diagnose as diagnose_version
from mst.models import gates as gates_version
from mst.plugins.bvar.spec import Spec as BvarSpec
from mst.plugins.msvar.spec import Spec as MsvarSpec

IDS = ["gl.a", "ru.b", "ru.c"]
MONTHS = [date(2000 + m // 12, m % 12 + 1, 1) for m in range(36)]


def panel_y(seed: int) -> np.ndarray:
    y = np.cumsum(np.random.default_rng(seed).standard_normal((36, 3)), axis=0) + 10.0
    y[:3, 2] = np.nan  # a late series, so the panel has something to interpolate
    return y


def bvar_spec() -> dict:
    return BvarSpec.model_validate(
        {
            "start": date(2000, 1, 1),
            "lags": 1,
            "tightness": 0.2,
            "volatility": "constant",
            "volvol_mean": 0.03,
            "volvol_df": 5.0,
            "stationary": [],
            "sign_lag": 2,
            "rotation_tries": 100,
            "sampler": {"draws": 10, "burn": 1, "thin": 1, "seed": 1},
        }
    ).model_dump(mode="json")


def msvar_spec() -> dict:
    return MsvarSpec.model_validate(
        {
            "start": date(2000, 1, 1),
            "lags": 1,
            "tightness": 0.2,
            "stationary": [],
            "sign_lag": 2,
            "rotation_tries": 100,
            "transition_stay": 4.0,
            "transition_switch": 0.5,
            "sampler": {"draws": 10, "burn": 1, "thin": 1, "seed": 1},
        }
    ).model_dump(mode="json")


def shared_meta() -> dict:
    return {
        "ids": IDS,
        "n_global": 1,
        "months": [m.isoformat() for m in MONTHS],
        "as_of": "2002-12-01",
        "diagnostics": {"estimated": 10, "stable": 6, "identified": 5},
    }


@pytest.fixture(scope="module")
def versions(tmp_path_factory):
    """A bvar and an msvar version of known draws, saved the way the platform saves them."""
    root = tmp_path_factory.mktemp("models")
    y = panel_y(0)
    panel = {"panel_y": y, "panel_approximate": np.zeros(y.shape, dtype=bool)}
    # Half the draws explosive (rho = 1.1), half stable (rho = 0.9); two dof columns at the
    # floor and one Gaussian.
    explosive = np.tile(np.diag([1.1, 0.5, 0.5])[None], (5, 1, 1, 1)).reshape(5, 1, 3, 3)
    stable = np.tile(np.diag([0.9, 0.5, 0.5])[None], (5, 1, 1, 1)).reshape(5, 1, 3, 3)
    lags = np.concatenate([explosive, stable])
    bvar = macro_version.save(
        "bvar.sv",
        {"shocks": ["ru.t"], "spec": bvar_spec(), **shared_meta()},
        {"lags": lags, "dof": np.tile([3.0, 4.0, np.inf], (10, 1)), **panel},
        root,
    )
    # The msvar: the calm regime stable in every draw, the stress one explosive in half.
    calm = np.tile(np.diag([0.8, 0.5, 0.5])[None, None], (10, 1, 1, 1, 1))
    per_regime = np.concatenate([calm, lags[:, None]], axis=1)
    states = np.zeros((10, 35), dtype=int)
    states[:, 30:] = 1  # the last months of history sit in the stress regime, every draw
    msvar = macro_version.save(
        "msvar.markov",
        {
            "shocks": ["ru.t", "regime.calm", "regime.stress"],
            "regimes": ["regime.calm", "regime.stress"],
            "spec": msvar_spec(),
            **shared_meta(),
        },
        {"lags": per_regime, "states": states, **panel},
        root,
    )
    return root, bvar, msvar


def test_bvar_rho_draws_and_dof(versions):
    root, bvar, _ = versions
    found = diagnose_version.diagnose(root, bvar)
    assert found["model"] == "bvar.sv"
    assert found["draws"] == {"estimated": 10, "stable": 6, "identified": 5}
    model = found["rho"]["model"]
    for limit in ("1", "1.005", "1.01", "1.02"):
        assert model["above"][limit] == pytest.approx(0.5)
    assert model["median"] == pytest.approx(1.0)  # five draws at 1.1, five at 0.9
    assert model["max"] == pytest.approx(1.1)
    assert found["dof"] == {"gl.a": 3.0, "ru.b": 4.0, "ru.c": None}


def test_bvar_prior_weight_grows_as_the_prior_tightens(versions, tmp_path):
    root, bvar, _ = versions
    loose = diagnose_version.diagnose(root, bvar)["prior_weight"]
    assert set(loose) == set(IDS)
    # The audit's measure: a loose prior on 36 months of levels weighs little, and a tight
    # one weighs more — the same panel, the same draws, a smaller lambda.
    assert all(0.0 < weight < 1.0 for weight in loose.values())
    spec = bvar_spec() | {"tightness": 0.01}
    tighter = macro_version.save(
        "bvar.sv",
        {"shocks": ["ru.t"], "spec": spec, **shared_meta()},
        {
            "lags": np.tile(np.diag([0.9, 0.5, 0.5])[None], (10, 1, 1, 1)).reshape(10, 1, 3, 3),
            "panel_y": panel_y(0),
            "panel_approximate": np.zeros((36, 3), dtype=bool),
        },
        tmp_path,
    )
    tight = diagnose_version.diagnose(tmp_path, tighter)["prior_weight"]
    assert all(tight[id_] > loose[id_] for id_ in IDS)


def test_msvar_rho_is_per_regime_and_the_regime_shares_are_read(versions):
    root, _, msvar = versions
    found = diagnose_version.diagnose(root, msvar)
    assert found["model"] == "msvar.markov"
    calm, stress = found["rho"]["regime.calm"], found["rho"]["regime.stress"]
    assert calm["median"] == pytest.approx(0.8)
    assert calm["above"]["1"] == 0.0
    assert stress["median"] == pytest.approx(1.0)
    assert stress["above"]["1"] == pytest.approx(0.5)
    # The last months of history sit in the stress regime in every draw.
    assert found["stress_share_last_month"] == 1.0
    assert found["regime_share_last_month"]["regime.calm"] == 0.0
    by_year = found["regime_share_by_year"]
    assert set(by_year) == {"2000", "2001", "2002"}
    # The equation months run 2000-02 .. 2002-12; the stress months are the five last months
    # of 2002, and 2002 holds twelve equation months.
    assert by_year["2002"]["regime.stress"] == pytest.approx(5 / 12)
    assert by_year["2000"]["regime.stress"] == 0.0
    for year in by_year.values():
        assert sum(year.values()) == pytest.approx(1.0)
    assert "dof" not in found


def test_a_version_of_an_unknown_model_has_no_prior_weight(versions, tmp_path):
    root, bvar, _ = versions
    meta, arrays = macro_version.read(root, bvar)
    foreign = macro_version.save("other.model", meta | {"model": "other.model"}, arrays, tmp_path)
    found = diagnose_version.diagnose(tmp_path, foreign)
    assert found["model"] == "other.model"
    assert found["prior_weight"] == {}
    assert found["rho"]["model"]["median"] == pytest.approx(1.0)


def test_a_changed_version_is_refused(versions, tmp_path):
    root, bvar, _ = versions
    shutil.copytree(root / "macro" / bvar, tmp_path / "macro" / bvar)
    meta_file = tmp_path / "macro" / bvar / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["changed"] = "by hand"
    meta_file.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its hash"):
        diagnose_version.diagnose(tmp_path, bvar)


def write_run(
    tmp_path, run_id, *, variables, checks, changes, values, paths
) -> None:
    """A finished run folder with the files the gates read."""
    folder = tmp_path / "runs" / run_id
    folder.mkdir(parents=True)
    (folder / "meta.json").write_text(
        json.dumps({"scenario": "test", "variables": variables, "checks": {"paths": checks}}),
        encoding="utf-8",
    )
    np.savez(folder / "changes.npz", changes=changes)
    np.savez(folder / "paths.npz", macro=paths)
    (folder / "results.json").write_text(
        json.dumps(
            {
                "positions": [
                    {"position_id": f"p{k}", "value_current": v}
                    for k, v in enumerate(values)
                ]
            }
        ),
        encoding="utf-8",
    )


def test_macro_gates_read_a_finished_run(tmp_path):
    variables = ["ru.usdrub", "ru.key_rate", "gl.fed_rate", "ru.spread_bbb", "ru.cpi"]
    checks = {
        "macro/ru.usdrub": {"band_outside_domain": 0.0, "band_wider_than_history": 0.1},
        "macro/ru.key_rate": {"band_outside_domain": 0.09, "band_wider_than_history": 0.2},
        "driver/share_price:sibn": {"band_outside_domain": 0.9, "band_wider_than_history": 0.9},
    }
    changes = np.array([[1.0, 2.0], [-3.0, 5.0], [0.0, 0.0], [7.0, -1.0]])  # best path gains 6
    paths = np.zeros((4, 5, len(variables)))
    paths[:, 1:, 0] = 400.0  # USD/RUB above 300 in every path
    paths[0, 2, 1] = -1.0  # one path dips the key rate below zero
    write_run(
        tmp_path,
        "20260101T000000-0001",
        variables=variables,
        checks=checks,
        changes=changes,
        values=[5000.0, 5000.0],
        paths=paths,
    )
    found = gates_version.gates(tmp_path, ["20260101T000000-0001"])
    (row,) = found["runs"]
    assert row["scenario"] == "test"
    # The driver cells do not count: the shares pool the macro cells only.
    assert row["macro_band_outside_domain"] == pytest.approx(0.045)
    assert row["macro_band_wider_than_history"] == pytest.approx(0.15)
    assert row["largest_gain_of_notional"] == pytest.approx(6.0 / 10000.0)
    assert row["absurd"] == {
        "usdrub_above_300": 1.0,
        "key_rate_below_0": 0.25,
        "fed_rate_below_0": 0.0,
        "spread_bbb_below_0": 0.0,
    }
    # The key rate and the Fed rate carry the registry floor of zero; the fake run's paths
    # sit at zero for every horizon month, so every path touches the floor. The other
    # variables carry none.
    assert row["bound_touch"] == {"ru.key_rate": 1.0, "gl.fed_rate": 1.0}
    assert row["gates"] == {
        "band_outside_domain_below_five_percent": True,
        "band_wider_than_history_below_five_percent": False,
        "largest_gain_below_ten_notional": True,
    }


def test_macro_gates_name_a_broken_run(tmp_path):
    found = gates_version.gates(tmp_path, ["20260101T000000-0002"])
    (row,) = found["runs"]
    assert row["run"] == "20260101T000000-0002"
    assert "no meta.json" in row["broken"]
