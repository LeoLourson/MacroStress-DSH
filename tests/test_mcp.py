"""The server answers a call in memory, so the MCP layer is checked without a transport."""

import asyncio

import pytest
from fastmcp import Client

from mst import jobs, mcp, store
from mst.mcp import build


def call(ws, name, arguments):
    """One tool call against a server built for ``ws``."""

    async def run():
        async with Client(build(ws)) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


def test_jobs_state_returns_the_workspace_summary(tmp_path):
    ws = store.Workspace(tmp_path)
    result = call(ws, "jobs", {"action": "state"})
    assert result.data == {
        "workspace": str(tmp_path.resolve()),
        "macro": None,
        "satellites": None,
        "corpus_index": None,
        "portfolios": 0,
        "scenarios": 0,
        "runs": 0,
        "jobs": 0,
    }


def test_jobs_state_refreshes_the_file_the_prompt_section_reads(tmp_path):
    ws = store.Workspace(tmp_path)
    (ws.portfolios / "demo").mkdir(parents=True)
    assert store.read_json(ws.state_file) is None
    call(ws, "jobs", {"action": "state"})
    assert store.read_json(ws.state_file)["portfolios"] == 1


def test_jobs_wait_returns_a_finished_job_in_one_call(tmp_path):
    ws = store.Workspace(tmp_path)
    job_id = jobs.start(ws, "code", ["-c", "print(1)"])
    found = call(ws, "jobs", {"action": "wait", "id": job_id, "timeout_sec": 60})
    assert found.data["running"] is False
    assert found.data["exit_code"] == 0
    assert "1" in call(ws, "jobs", {"action": "log", "id": job_id}).data["log"]


def test_jobs_wait_returns_the_running_job_when_the_time_is_out(tmp_path):
    ws = store.Workspace(tmp_path)
    job_id = jobs.start(ws, "code", ["-c", "import time; time.sleep(30)"])
    try:
        found = call(ws, "jobs", {"action": "wait", "id": job_id, "timeout_sec": 0.5})
    finally:
        call(ws, "jobs", {"action": "cancel", "id": job_id})
    assert found.data["running"] is True
    assert found.data["exit_code"] is None


def test_jobs_rejects_an_unknown_action(tmp_path):
    ws = store.Workspace(tmp_path)
    try:
        call(ws, "jobs", {"action": "start"})
    except Exception as error:  # fastmcp raises ToolError with the message of the tool
        assert "unknown action" in str(error)
    else:
        raise AssertionError("an unknown action must not be accepted")


def test_jobs_status_log_and_cancel_go_through_the_tool(tmp_path):
    ws = store.Workspace(tmp_path)
    job_id = jobs.start(ws, "code", ["-c", "import time; time.sleep(30)"])
    try:
        assert call(ws, "jobs", {"action": "status", "id": job_id}).data["running"] is True
        assert call(ws, "jobs", {"action": "log", "id": job_id}).data == {"id": job_id, "log": ""}
    finally:
        cancelled = call(ws, "jobs", {"action": "cancel", "id": job_id}).data
    assert cancelled["cancelled"] is True
    assert cancelled["running"] is False


def test_the_run_tool_keeps_an_explicit_empty_targets_list(workspace):
    # An explicit empty list is not "all drivers": RunRequest refuses it, and the tool must
    # pass it through instead of dropping the argument.
    positions = [
        {"id": "sber", "type": "common_stock", "notional": 1.0, "codes": {"secid": "SBER"}}
    ]
    try:
        call(
            workspace,
            "run",
            {
                "action": "start",
                "scenario": "oil",
                "draws": 3,
                "seed": 5,
                "positions": positions,
                "targets": [],
            },
        )
    except Exception as error:
        assert "targets name the drivers of the forecast" in str(error)
    else:
        raise AssertionError("an explicit empty targets list must not mean all drivers")


def test_plugin_output_does_not_reach_the_servers_stdout(tmp_path, monkeypatch, capsys):
    # scenario, portfolio and model run plugin code in the server's process; a stray
    # print() of a plugin must land on stderr, where the client logs it, not on the
    # stdout the stdio transport owns.
    from mst import plugins

    def noisy():
        print("plugin debug noise")
        raise KeyError("nowhere")

    monkeypatch.setattr(plugins, "registry", noisy)
    ws = store.Workspace(tmp_path)
    try:
        call(ws, "model", {"kind": "catalog", "action": "list"})
    except Exception as error:  # fastmcp raises ToolError with the message of the tool
        assert "nowhere" in str(error)
    else:
        raise AssertionError("a broken registry must not pass silently")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "plugin debug noise" in captured.err


def test_jobs_without_an_id_is_refused(tmp_path):
    ws = store.Workspace(tmp_path)
    try:
        call(ws, "jobs", {"action": "status"})
    except Exception as error:
        assert "id is required" in str(error)
    else:
        raise AssertionError("status without an id must not be accepted")


def test_the_scenario_tool_takes_the_arguments_of_the_reverse_search(workspace):
    try:
        call(
            workspace,
            "scenario",
            {
                "action": "reverse", "id": "oil", "portfolio": "nowhere",
                "kappa": 1.0, "draws": 2, "seed": 1,
            },
        )
    except Exception as error:
        assert "nowhere" in str(error)
    else:
        raise AssertionError("the portfolio of the search does not exist")


def test_the_backtest_is_started_on_the_plan_of_the_repository(tmp_path):
    argv = mcp._backtest_argv(store.Workspace(tmp_path), "2026-10-01", "bvar.sv")

    plan = argv[argv.index("--episodes") + 1]
    assert plan.replace("\\", "/").endswith("bench/model-backtest/episodes.toml")


def test_a_backtest_without_the_plan_names_the_command_to_run(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp, "EPISODE_PLAN", tmp_path / "absent.toml")

    with pytest.raises(ValueError, match="--episodes FILE"):
        mcp._backtest_argv(store.Workspace(tmp_path), "2026-10-01", "bvar.sv")
def _one_input_model(monkeypatch):
    from mst import plugins
    from tests.plugins import fakes

    registry = plugins.build([fakes.model_plugin(macro=(fakes.InputMacroModel(),))])
    monkeypatch.setattr(plugins, "registry", lambda: registry)


def test_the_inputs_tool_lists_questions_and_refuses_an_unknown_action(tmp_path, monkeypatch):
    _one_input_model(monkeypatch)
    ws = store.Workspace(tmp_path)
    listed = call(ws, "model", {"kind": "inputs", "action": "list", "as_of": "2026-11-01"}).data
    assert listed["model"] == "gmodel.inputs.v1"
    assert [row["key"] for row in listed["keys"]] == ["q1", "q2"]
    assert all(not row["written"] for row in listed["keys"])
    try:
        call(ws, "model", {"kind": "inputs", "action": "nope", "as_of": "2026-11-01"})
    except Exception as error:
        assert "unknown action" in str(error)


def test_the_inputs_tool_gives_the_question_and_writes_the_answer(tmp_path, monkeypatch):
    _one_input_model(monkeypatch)
    ws = store.Workspace(tmp_path)
    question = call(
        ws,
        "model",
        {"kind": "inputs", "action": "question", "as_of": "2026-11-01", "key": "q1"},
    ).data
    assert question["key"] == "q1" and "text" in question and "schema" in question

    written = call(
        ws,
        "model",
        {
            "kind": "inputs",
            "action": "write",
            "as_of": "2026-11-01",
            "key": "q1",
            "payload": {"value": 3.0},
        },
    ).data
    assert written["key"] == "q1" and len(written["hash"]) == 64
    assert written["missing"] == ["q2"]

    found = call(
        ws, "model", {"kind": "inputs", "action": "status", "as_of": "2026-11-01"}
    ).data
    assert found["written"] == ["q1"] and found["missing"] == ["q2"]
    assert (ws.elicitation / "2026-11-01" / "q1.json").is_file()


def test_a_refused_answer_writes_nothing_and_names_the_field(tmp_path, monkeypatch):
    _one_input_model(monkeypatch)
    ws = store.Workspace(tmp_path)
    try:
        call(
            ws,
            "model",
            {
                "kind": "inputs",
                "action": "write",
                "as_of": "2026-11-01",
                "key": "q1",
                "payload": {"nope": 1.0},
            },
        )
    except Exception as error:
        assert "value" in str(error)
    assert not (ws.elicitation / "2026-11-01").exists()
