"""Published research artifacts detect corruption even after a file is rehashed."""
import csv
import json
import shutil

import pytest

from evaluation.core import sha256
from evaluation.published_results import RESULT_DIRECTORY, ROOT, main, verify


@pytest.fixture
def publication(tmp_path):
    directory = tmp_path / "published"
    shutil.copytree(ROOT / RESULT_DIRECTORY, directory)
    return directory


def edit_metrics(directory, edit, rehash=True):
    path = directory / "metrics.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    edit(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)
    if rehash:
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["artifacts"]["metrics.csv"]["sha256"] = sha256(path)
        (directory / "manifest.json").write_text(json.dumps(manifest))


def test_published_matrix_integrity_does_not_certify_judge_accuracy():
    result = verify()
    assert result == {"integrity_verified": True, "model_count": 13, "benchmark_count": 4,
                      "metric_rows": 52, "model_item_results": 12194,
                      "review_status": "automatic_scores_with_known_judge_errors", "score_validity_established": False}


def test_changed_file_fails_before_metric_checks(publication):
    edit_metrics(publication, lambda rows: rows[0].update(passed="0"), rehash=False)
    with pytest.raises(ValueError, match="Artifact checksum changed"):
        verify(publication)


def test_rehashing_incorrect_arithmetic_does_not_hide_changed_pass_counts(publication):
    edit_metrics(publication, lambda rows: rows[0].update(passed="0"))
    with pytest.raises(ValueError, match="Pass rate differs from counts"):
        verify(publication)


@pytest.mark.parametrize("change", ["drop", "duplicate"])
def test_every_model_dataset_cell_is_required_once(publication, change):
    def edit(rows):
        rows.pop() if change == "drop" else rows.append(rows[0].copy())
    edit_metrics(publication, edit)
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        verify(publication)


def test_omitted_legacy_identifier_diagnostics_are_not_silently_zero(publication):
    def edit(rows):
        for row in rows:
            if row["benchmark"] == "model-spec-original":
                row["judge_section_mismatch"] = "0"
    edit_metrics(publication, edit)
    with pytest.raises(ValueError, match="Aggregate diagnostics differ"):
        verify(publication)


def test_public_result_cli_reports_integrity_and_known_limits(capsys):
    assert main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["integrity_verified"] is True and result["score_validity_established"] is False


def test_cli_reports_missing_checkout_without_a_traceback(tmp_path, capsys):
    assert main(["--root", str(tmp_path)]) == 2
    assert "Published result error:" in capsys.readouterr().err
