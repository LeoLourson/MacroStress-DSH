"""``python -m mst.metrics``: the command line names the episode plan it checks on."""

import pytest

from mst.metrics import cli
from mst.metrics.backtest import _file_digest

BASE = ["--as-of", "2026-10-01", "--store", "s", "--models", "m"]


@pytest.mark.parametrize("command", [["backtest", "--corpus", "c.json"], ["rolling"]])
def test_backtest_needs_the_episode_plan(command, capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main([*command, *BASE])

    assert stopped.value.code == 2
    assert "--episodes" in capsys.readouterr().err


def test_the_plan_hash_does_not_depend_on_where_the_file_lies(tmp_path):
    old = tmp_path / "src" / "episodes.toml"
    new = tmp_path / "bench" / "episodes.toml"
    for file in (old, new):
        file.parent.mkdir()
        file.write_text("draws = 40\nseed = 3\n", encoding="utf-8")

    assert _file_digest(old) == _file_digest(new) is not None
