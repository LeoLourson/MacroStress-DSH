"""The DSH plugin belongs to no machine: no absolute path, one version with the Python package."""

import json
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from mst import guides, mcp

ROOT = Path(__file__).resolve().parent.parent
ABSOLUTE = re.compile(r"[A-Za-z]:[\\/][A-Za-z]|/home/[a-z]|/Users/[A-Za-z]")


def test_no_tracked_text_names_a_machine():
    # The owner's account name in any tracked file means a path that works on one machine only.
    found = subprocess.run(
        ["git", "grep", "-nIE", r"[A-Za-z]:\\Users\\|C:/Users/|/home/[a-z]+/", "--", ".",
         ":!tests/test_dsh_package.py", ":!uv.lock", ":!corpus/manifest.json"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
    ).stdout
    assert not found, found


def test_the_patch_holds_no_absolute_path():
    assert not ABSOLUTE.search((ROOT / "dsh" / "cordis.patch.yml").read_text(encoding="utf-8"))
    assert ABSOLUTE.search(r"command: 'C:\Users\someone\python.exe'")


def test_the_plugin_is_public_and_shares_the_version_of_the_package():
    plugin = json.loads((ROOT / "dsh" / "package.json").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert plugin.get("private") is not True
    assert plugin["version"] == project["version"]
    assert plugin["dsh"]["bundle"]["patch"] == "./cordis.patch.yml"
    assert "dsh-plugin" in plugin["keywords"]
    patch = (ROOT / "dsh" / "cordis.patch.yml").read_text(encoding="utf-8")
    assert f"macrostress>={project['version']}" in patch


def test_the_skills_are_the_guides_of_the_server():
    found = sorted((ROOT / "dsh" / "skills").glob("*.md"))
    skills = [path for path in found if not path.name.startswith("README")]
    assert [path.stem for path in skills] == [item["name"] for item in guides.catalogue()]
    for path in skills:
        own = guides.source(path.stem).read_text(encoding="utf-8")
        assert path.read_text(encoding="utf-8").replace("\r\n", "\n") == own.replace(
            "\r\n", "\n"
        )


def test_the_default_workspace_follows_the_rule_of_the_plugin(monkeypatch, tmp_path):
    monkeypatch.setenv("MACROSTRESS_WORKSPACE", str(tmp_path))
    assert mcp.default_workspace() == tmp_path
    monkeypatch.delenv("MACROSTRESS_WORKSPACE")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mcp.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert mcp.default_workspace() == tmp_path / "local" / "macrostress" / "workspace"
    monkeypatch.setattr(mcp.sys, "platform", "darwin")
    assert mcp.default_workspace() == (
        tmp_path / "Library" / "Application Support" / "macrostress" / "workspace"
    )
    monkeypatch.setattr(mcp.sys, "platform", "linux")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert mcp.default_workspace() == tmp_path / ".local" / "share" / "macrostress" / "workspace"
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert mcp.default_workspace() == tmp_path / "xdg" / "macrostress" / "workspace"


def test_an_existing_workspace_in_the_home_directory_is_kept(monkeypatch, tmp_path):
    monkeypatch.delenv("MACROSTRESS_WORKSPACE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".macrostress" / "workspace").mkdir(parents=True)
    assert mcp.default_workspace() == tmp_path / ".macrostress" / "workspace"


def test_an_existing_project_workspace_is_found(monkeypatch, tmp_path):
    monkeypatch.delenv("MACROSTRESS_WORKSPACE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "workspace").mkdir()
    assert mcp.default_workspace().resolve() == tmp_path / "workspace"


def test_without_rights_outside_the_project_the_server_uses_the_project_workspace(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("MACROSTRESS_WORKSPACE", raising=False)
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("LOCALAPPDATA", str(home / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "local"))
    monkeypatch.chdir(tmp_path)
    mkdir = Path.mkdir

    def refuse_outside(self, *args, **kwargs):
        if home in self.resolve().parents:
            raise PermissionError(13, "Access is denied", str(self))
        return mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", refuse_outside)
    monkeypatch.setattr(mcp, "build", lambda ws: type("Server", (), {"run": lambda self: None})())
    mcp.main([])
    assert (tmp_path / "workspace" / "workspace.json").is_file()
    assert not home.exists()


def test_a_workspace_that_cannot_be_created_stops_the_server_with_a_hint(monkeypatch, tmp_path):
    def refuse(self, *args, **kwargs):
        raise PermissionError(13, "Access is denied", str(self))

    monkeypatch.setattr(Path, "mkdir", refuse)
    with pytest.raises(SystemExit) as stop:
        mcp.main(["--workspace", str(tmp_path / "ws")])
    # Python prints a string exit code to stderr and exits with status 1.
    message = stop.value.code
    assert str(tmp_path / "ws") in message and "MACROSTRESS_WORKSPACE" in message
