"""The plugin SDK: the guides a plugin hands to every client."""

import pytest

from mst import guides, plugins
from tests.plugins import fakes


def guide_file(tmp_path, name, text=None):
    """A Markdown guide whose frontmatter name is ``name``."""
    path = tmp_path / f"{name}.md"
    path.write_text(
        text
        or f"---\nname: {name}\ndescription: Test guide\nwhenToUse: never\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return path


def test_a_plugin_guide_is_served_beside_the_server_guides(tmp_path, monkeypatch):
    path = guide_file(tmp_path, "gmodel-guide")
    registry = plugins.build([fakes.model_plugin(guides=(path,))])
    monkeypatch.setattr(plugins, "registry", lambda: registry)
    assert "gmodel-guide" in {item["name"] for item in guides.catalogue()}
    assert guides.get("gmodel-guide")["content"] == "# gmodel-guide"
    assert guides.source("gmodel-guide") == path


def test_a_guide_file_must_exist(tmp_path):
    with pytest.raises(plugins.PluginError, match="guide 'nowhere.md' does not exist"):
        plugins.build([fakes.model_plugin(guides=(tmp_path / "nowhere.md",))])


def test_a_guide_name_must_be_its_file_stem(tmp_path):
    path = guide_file(
        tmp_path,
        "other",
        text="---\nname: different\ndescription: d\nwhenToUse: n\n---\n\n# x\n",
    )
    with pytest.raises(plugins.PluginError, match="declares the name 'different'"):
        plugins.build([fakes.model_plugin(guides=(path,))])


def test_two_plugins_cannot_declare_one_guide(tmp_path):
    path = guide_file(tmp_path, "shared")
    with pytest.raises(plugins.PluginError, match="declared by gmodel and gmodel2"):
        plugins.build(
            [
                fakes.model_plugin(guides=(path,)),
                fakes.model_plugin(plugin_id="gmodel2", guides=(path,)),
            ]
        )
