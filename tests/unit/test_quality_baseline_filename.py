from __future__ import annotations

from pathlib import Path

from evaluation.quality_baseline import next_baseline_filename


def test_empty_results_dir_starts_sequence_at_001(tmp_path: Path):
    name = next_baseline_filename(tmp_path, "claude-sonnet-5", 0.2)

    assert name == "retrieval_baseline_001_claude-sonnet-5_0.2.json"

def test_missing_results_dir_starts_sequence_at_001(tmp_path: Path):
    missing = tmp_path / "does-not-exist-yet"

    name = next_baseline_filename(missing, "claude-sonnet-5", 0.2)

    assert name == "retrieval_baseline_001_claude-sonnet-5_0.2.json"

def test_sequence_increments_past_highest_existing_number(tmp_path: Path):
    (tmp_path / "retrieval_baseline_007_claude-sonnet-5_0.2.json").write_text("{}")
    (tmp_path / "retrieval_baseline_003_claude-sonnet-5_0.2.json").write_text("{}")

    name = next_baseline_filename(tmp_path, "claude-sonnet-5", 0.2)

    assert name.startswith("retrieval_baseline_008_")

def test_model_id_is_sanitized_for_filesystem_safety(tmp_path: Path):
    name = next_baseline_filename(tmp_path, "us.anthropic/claude-sonnet-5:1m", 0.2)

    assert name == "retrieval_baseline_001_us.anthropic-claude-sonnet-5-1m_0.2.json"
    assert "/" not in name and ":" not in name
