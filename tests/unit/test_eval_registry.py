"""Eval registry writer (docs/test-plan.md §14, S6.1; SPEC §11.2).

The registry is observed through ``evaluation.registry`` only: what ``write_run`` returns, raises
and leaves on disk under the registry root, and the values ``prompt_version`` and
``run_snapshot_hash`` compute from their inputs.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

import pytest
from evaluation.registry import (
    RegistryError,
    new_run_id,
    prompt_version,
    run_snapshot_hash,
    write_run,
)

from app.agents import graph
from app.services import llm

STRING_FIELDS = (
    "run_id",
    "git_sha",
    "branch",
    "model",
    "prompt_version",
    "evidence_release",
    "snapshot_hash",
    "fixture_version",
    "classification",
)
REQUIRED_FIELDS = (*STRING_FIELDS[:-1], "temperature", "metrics", "classification")
HEX16 = re.compile(r"^[0-9a-f]{16}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RELEASE = "alpha-evidence:0.1.0"


def record(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "run_id": "20261117T101500Z-60ee35c-ab12",
        "git_sha": "60ee35c",
        "branch": "task/s6-1",
        "model": "deepseek.v3.2",
        "temperature": 0.3,
        "prompt_version": "0123456789abcdef",
        "evidence_release": RELEASE,
        "snapshot_hash": "a" * 64,
        "fixture_version": "fedcba9876543210",
        "metrics": {"grounded_claim_rate": 0.93, "citation_coverage": 0.92},
        "classification": "candidate",
    }
    base.update(overrides)
    return base


def assert_nothing_written(root: Path) -> None:
    assert not root.exists() or not list(root.rglob("*")), "a refused record left files behind"


# --- Lineage required ------------------------------------------------------------------------


def test_lineage_complete_record_is_written_under_its_run_id(tmp_path: Path) -> None:
    """§14 Lineage required: a record with every §11.2 field is accepted and persisted."""
    root = tmp_path / "runs"
    entry = record()

    written = write_run(entry, root=root)

    assert written == root / f"{entry['run_id']}.json"
    stored = json.loads(written.read_text())
    for key, value in entry.items():
        assert stored[key] == value


def test_lineage_optional_result_path_is_stored(tmp_path: Path) -> None:
    """§14 Lineage required: result_path is the one optional field and is kept when given."""
    written = write_run(record(result_path="evaluation/quality_res/x.json"), root=tmp_path / "runs")

    assert json.loads(written.read_text())["result_path"] == "evaluation/quality_res/x.json"


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_lineage_missing_field_is_refused_and_writes_no_file(tmp_path: Path, field: str) -> None:
    """§14 Lineage required: any missing field is refused; a refused record writes no file."""
    root = tmp_path / "runs"
    entry = record()
    del entry[field]

    with pytest.raises(RegistryError):
        write_run(entry, root=root)

    assert_nothing_written(root)


@pytest.mark.parametrize("field", STRING_FIELDS)
def test_lineage_empty_string_field_is_refused_and_writes_no_file(
    tmp_path: Path, field: str
) -> None:
    """§14 Lineage required: any string field that is empty is refused; no file is written."""
    root = tmp_path / "runs"

    with pytest.raises(RegistryError):
        write_run(record(**{field: ""}), root=root)

    assert_nothing_written(root)


def test_lineage_extra_field_is_refused_and_writes_no_file(tmp_path: Path) -> None:
    """§14 Lineage required: an extra field is refused; no file is written."""
    root = tmp_path / "runs"

    with pytest.raises(RegistryError):
        write_run(record(operator_note="hand edited"), root=root)

    assert_nothing_written(root)


@pytest.mark.parametrize(
    "bad",
    [
        {"classification": "golden"},
        {"temperature": "warm"},
        {"metrics": {"grounded_claim_rate": "high"}},
        {"metrics": ["grounded_claim_rate"]},
    ],
    ids=[
        "classification-not-in-set",
        "temperature-not-a-number",
        "metric-not-a-number",
        "metrics-not-a-mapping",
    ],
)
def test_lineage_wrongly_typed_field_is_refused_and_writes_no_file(
    tmp_path: Path, bad: dict[str, Any]
) -> None:
    """§14 Lineage required: SPEC §11.2 types (classification in {candidate, approved}, numeric
    temperature and metrics) are enforced by the writer."""
    root = tmp_path / "runs"

    with pytest.raises(RegistryError):
        write_run(record(**bad), root=root)

    assert_nothing_written(root)


def test_lineage_new_run_id_has_the_documented_format_and_is_writable(tmp_path: Path) -> None:
    """Brief: run_id = <UTC yyyymmddThhmmssZ>-<sha7>-<4 hex>; a generated id is accepted."""
    sha = "60ee35c9d1f4b7a2c3e5f6a7b8c9d0e1f2a3b4c5"

    run_id = new_run_id(sha)

    assert re.fullmatch(r"\d{8}T\d{6}Z-60ee35c-[0-9a-f]{4}", run_id)
    assert len({new_run_id(sha) for _ in range(40)}) > 1
    write_run(record(run_id=run_id, git_sha=sha), root=tmp_path / "runs")


# --- Append-only -----------------------------------------------------------------------------


def test_append_only_existing_run_id_is_refused_and_file_is_byte_identical(
    tmp_path: Path,
) -> None:
    """§14 Append-only: a second record with an existing run_id is refused; the file is unchanged."""
    root = tmp_path / "runs"
    first = write_run(record(), root=root)
    before = first.read_bytes()

    with pytest.raises(RegistryError):
        write_run(record(metrics={"grounded_claim_rate": 0.1}, branch="other"), root=root)

    assert first.read_bytes() == before
    assert list(root.iterdir()) == [first]


def test_append_only_different_run_ids_coexist(tmp_path: Path) -> None:
    """§14 Append-only: only a repeated run_id is refused."""
    root = tmp_path / "runs"

    one = write_run(record(run_id="20261117T101500Z-60ee35c-0001"), root=root)
    two = write_run(record(run_id="20261117T101500Z-60ee35c-0002"), root=root)

    assert one != two
    assert sorted(path.name for path in root.iterdir()) == sorted([one.name, two.name])


# --- Classification ceiling ------------------------------------------------------------------


def test_classification_ceiling_live_evidence_cannot_be_approved(tmp_path: Path) -> None:
    """§14 Classification ceiling (SPEC §11.1): live + approved is refused and writes no file."""
    root = tmp_path / "runs"

    with pytest.raises(RegistryError):
        write_run(record(evidence_release="live", classification="approved"), root=root)

    assert_nothing_written(root)


@pytest.mark.parametrize(
    ("evidence_release", "classification"),
    [("live", "candidate"), (RELEASE, "approved"), (RELEASE, "candidate")],
)
def test_classification_ceiling_allows_everything_else(
    tmp_path: Path, evidence_release: str, classification: str
) -> None:
    """§14 Classification ceiling: live+candidate and frozen-release+approved are accepted."""
    written = write_run(
        record(evidence_release=evidence_release, classification=classification),
        root=tmp_path / "runs",
    )

    stored = json.loads(written.read_text())
    assert (stored["evidence_release"], stored["classification"]) == (
        evidence_release,
        classification,
    )


# --- prompt_version --------------------------------------------------------------------------


def test_prompt_version_is_a_short_hex_and_stable_for_the_same_prompt_text() -> None:
    """§14 prompt_version: a 16-hex sha256 prefix; the same prompt text gives the same value."""
    first = prompt_version()

    assert HEX16.fullmatch(first)
    assert prompt_version() == first


def test_prompt_version_changes_when_the_system_prompt_is_edited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§14 prompt_version: an edit to the analyst system prompt gives a different value."""
    original = prompt_version()

    monkeypatch.setattr(llm, "ANALYST_SYSTEM_PROMPT", llm.ANALYST_SYSTEM_PROMPT + " Be terse.")
    edited = prompt_version()

    assert HEX16.fullmatch(edited)
    assert edited != original
    monkeypatch.undo()
    assert prompt_version() == original


def test_prompt_version_changes_when_the_draft_template_is_edited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§14 prompt_version: an edit to the draft-memo template gives a different value."""
    original = prompt_version()

    template = vars(graph)["DRAFT_PROMPT_TEMPLATE"]
    monkeypatch.setattr(graph, "DRAFT_PROMPT_TEMPLATE", template + "\nBe terse.", raising=False)
    edited = prompt_version()

    assert HEX16.fullmatch(edited)
    assert edited != original
    monkeypatch.undo()
    assert prompt_version() == original


# --- Run snapshot hash -----------------------------------------------------------------------

HASHES = [f"{n:064x}" for n in (3, 1, 2, 7)]


def test_run_snapshot_hash_is_sha256_hex() -> None:
    """§14 Run snapshot hash: a 64-hex digest."""
    assert HEX64.fullmatch(run_snapshot_hash(RELEASE, HASHES))


def test_run_snapshot_hash_ignores_the_order_of_the_hashes() -> None:
    """§14 Run snapshot hash: depends on the set of hashes, not their order (any iterable)."""
    expected = run_snapshot_hash(RELEASE, HASHES)

    for permutation in itertools.permutations(HASHES):
        assert run_snapshot_hash(RELEASE, list(permutation)) == expected
    assert run_snapshot_hash(RELEASE, iter(reversed(HASHES))) == expected


def test_run_snapshot_hash_depends_on_the_release_id() -> None:
    """§14 Run snapshot hash: the same hashes under another release id hash differently."""
    assert run_snapshot_hash("alpha-evidence:0.2.0", HASHES) != run_snapshot_hash(RELEASE, HASHES)
    assert run_snapshot_hash("live", HASHES) != run_snapshot_hash(RELEASE, HASHES)


def test_run_snapshot_hash_depends_on_the_members_of_the_set() -> None:
    """§14 Run snapshot hash: a changed or missing member changes the hash."""
    base = run_snapshot_hash(RELEASE, HASHES)

    assert run_snapshot_hash(RELEASE, [*HASHES[:-1], f"{99:064x}"]) != base
    assert run_snapshot_hash(RELEASE, HASHES[:-1]) != base
    assert run_snapshot_hash(RELEASE, [*HASHES, f"{99:064x}"]) != base
