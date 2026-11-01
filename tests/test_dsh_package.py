"""The DSH plugin belongs to no machine: no absolute path, one version with the Python package."""

import json
import re
import subprocess
import tomllib
from pathlib import Path

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
        own = (guides.ROOT / path.name).read_text(encoding="utf-8")
        assert path.read_text(encoding="utf-8").replace("\r\n", "\n") == own.replace("\r\n", "\n")


def test_the_default_workspace_follows_the_rule_of_the_plugin(monkeypatch, tmp_path):
    monkeypatch.setenv("MACROSTRESS_WORKSPACE", str(tmp_path))
    assert mcp.default_workspace() == tmp_path
    monkeypatch.delenv("MACROSTRESS_WORKSPACE")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert mcp.default_workspace() == tmp_path / ".macrostress" / "workspace"
