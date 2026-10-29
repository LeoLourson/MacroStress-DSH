"""mst.versions: the folders of model versions, named by their hash."""

import os

import numpy as np
import pytest

from mst import versions

META = {"as_of": "2020-01-01", "ids": ["gl.a", "ru.b"], "note": "сдвиг"}
ARRAYS = {"x": np.arange(6.0).reshape(2, 3), "flag": np.array([True, False])}


def test_the_hash_stays_the_one_the_saved_versions_are_named_by():
    # Computed by the hash of mst.macro.version before it moved here.
    assert versions.digest(META, ARRAYS) == (
        "27ed5e4f4e2c3eca9dc3e394ac2da5ee70cc6140544b0c2a0e2f7bad3bb63a19"
    )


def test_a_saved_version_reads_back_and_a_changed_one_is_refused(tmp_path):
    name = versions.save(tmp_path, META, ARRAYS)
    meta, arrays = versions.read(tmp_path, name, "model version")
    assert meta == META and np.array_equal(arrays["x"], ARRAYS["x"])

    (tmp_path / name / "meta.json").write_text('{"as_of": "2021-01-01"}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its hash"):
        versions.read(tmp_path, name, "model version")


def test_a_name_that_is_not_a_hash_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a model version"):
        versions.read(tmp_path, "../../etc", "model version")


def test_folders_leave_out_drafts_and_put_the_newest_first(tmp_path):
    older = versions.save(tmp_path, META, ARRAYS)
    newer = versions.save(tmp_path, {**META, "note": "другой"}, ARRAYS)
    draft = tmp_path / ".draft-x"
    draft.mkdir()
    (draft / "meta.json").write_text("{}", encoding="utf-8")
    stamp = (tmp_path / older).stat().st_mtime
    os.utime(tmp_path / newer, (stamp + 60, stamp + 60))
    os.utime(draft, (stamp + 120, stamp + 120))

    assert [path.name for path in versions.folders(tmp_path)] == [newer, older]
