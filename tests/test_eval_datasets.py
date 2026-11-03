"""The published release preserves upstream histories and detects changed data."""
from copy import deepcopy
import json

import pytest

from evaluation.core import sha256
from evaluation.datasets import DEFAULT_DIRECTORY, load_collection, verify


def test_four_collection_release_preserves_official_non_user_endings_and_tool_ids():
    result = verify()
    assert result["collections"] == {
        "model-spec-original": 587, "model-spec-generic": 194,
        "constitution-original": 71, "constitution-generic": 86,
    }
    assert result["total_rows"] == 938 and result["custom_rows"] == 351
    official = load_collection("model-spec-original")
    assert sum(row["input"][-1]["role"] == "tool" for row in official) == 4
    assert sum(row["input"][-1]["role"] == "system" for row in official) == 3
    tools = [message for row in official for message in row["input"] if message["role"] == "tool"]
    call_ids = {call["id"] for row in official for message in row["input"] for call in message.get("tool_calls", [])}
    assert all(message["tool_call_id"] in call_ids for message in tools)
    for name in result["collections"]:
        for row in load_collection(name):
            assert "origin_run" not in row["metadata"] and "original_run" not in row["metadata"]


@pytest.mark.parametrize("update_file_hash", [False, True])
def test_modified_rubric_requires_new_content_binding(tmp_path, update_file_hash):
    name = "model-spec-generic"
    manifest = json.loads((DEFAULT_DIRECTORY / "manifest.json").read_text())
    info = manifest["collections"][name]
    rows = deepcopy(load_collection(name))
    rows[0]["target"] = "Changed rubric without the original source review."
    path = tmp_path / info["path"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if update_file_hash:
        info["sha256"] = sha256(path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inputs or rubrics changed" if update_file_hash else "checksum changed"):
        load_collection(name, tmp_path)
