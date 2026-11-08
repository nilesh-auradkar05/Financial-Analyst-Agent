"""Building an embedding arm's index from the baseline chunks (test-plan §21; CM-5a, CM-5b, layer I).

The baseline is a 2560-d index stamped with the default identifier; the arm is a 1024-d Bedrock
Cohere index. Both live on the local OpenSearch test container, each filled through the store with a
hashed fake embedder of its dimension. The build runs under the arm's settings, as the CLI does, so
reading the baseline must not be refused because its vector size differs from the configured one.

CM-5b (test-plan §21, amended 2026-11-08): the build also waits for the arm index to become visible
(on OpenSearch Serverless a write is searchable up to ~6 s later) and checks every fixture gold
anchor. The corpus is the Gate-shaped one of `gate_support`, so an anchored fixture matches it.
Write lag is simulated at the store boundary: the target store defers part of its writes.
"""

from __future__ import annotations

import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from app.components.retrieval.opensearch_store import OpenSearchVectorStore
from app.components.retrieval.vector_store import IndexDocument, SearchFilters
from evaluation.validate_retrieval_fixture import AnchoredFixture
from tests.embedding_arms import BEDROCK_COHERE, OLLAMA, use_embedding_settings
from tests.rag.gate_support import SECTIONS, build_gate_chunks, load_fixture_from
from tests.rag.helpers import SizedFakeEmbedder, raw_vector

NO_ENDPOINT = "http://127.0.0.1:1"
SOURCE_EMBEDDER = SizedFakeEmbedder(OLLAMA.dimension, model="qwen-like")
TARGET_EMBEDDER = SizedFakeEmbedder(BEDROCK_COHERE.dimension, model="cohere-like")
TIMEOUT_S = 0.6
POLL_S = 0.05


def _store(url: str, index: str, embedder: SizedFakeEmbedder) -> OpenSearchVectorStore:
    return OpenSearchVectorStore(index_name=index, url=url, timeout_s=10, embeddings=embedder)


class DeferringTargetStore(OpenSearchVectorStore):
    """The arm's store as Serverless shows it: chunks chosen by `defer` become searchable
    `release_after_s` seconds late (never, when None); the rest are written at once. Everything
    else is the real store, re-embedding included."""

    def __init__(
        self,
        *,
        defer: Callable[[IndexDocument], bool],
        release_after_s: float | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._defer = defer
        self._release_after_s = release_after_s
        self.late_failures: list[BaseException] = []
        self._timers: list[threading.Timer] = []

    def add_documents(self, documents: list[IndexDocument]) -> int:
        later = [document for document in documents if self._defer(document)]
        written = super().add_documents([d for d in documents if not self._defer(d)])
        if later and self._release_after_s is not None:
            timer = threading.Timer(self._release_after_s, self._write_late, args=(later,))
            self._timers.append(timer)
            timer.start()
        return written

    def _write_late(self, documents: list[IndexDocument]) -> None:
        try:
            super().add_documents(documents)
        except BaseException as error:  # surfaced from the test thread
            self.late_failures.append(error)

    def finish_late_writes(self) -> None:
        for timer in self._timers:
            timer.join()


ArmPair = Callable[..., tuple[OpenSearchVectorStore, DeferringTargetStore]]


@pytest.fixture
def corpus() -> list[IndexDocument]:
    """The Gate-shaped corpus: one chunk per gold anchor of `anchored`, two tickers."""
    return build_gate_chunks()


@pytest.fixture
def anchored(tmp_path: Path) -> AnchoredFixture:
    directory = tmp_path / "anchored"
    directory.mkdir()
    return load_fixture_from(directory)


@pytest.fixture
def arm_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opensearch_url: str,
    new_index: Callable[[], str],
) -> Iterator[ArmPair]:
    """Factory: a baseline filled with the given chunks and an empty arm index, settings on the arm."""
    targets: list[DeferringTargetStore] = []

    def build(
        documents: list[IndexDocument],
        *,
        defer: Callable[[IndexDocument], bool] = lambda document: False,
        release_after_s: float | None = None,
    ) -> tuple[OpenSearchVectorStore, DeferringTargetStore]:
        use_embedding_settings(monkeypatch, tmp_path, OLLAMA.routed_to(NO_ENDPOINT))
        source = _store(opensearch_url, new_index(), SOURCE_EMBEDDER)
        source.add_documents(documents)
        use_embedding_settings(monkeypatch, tmp_path, BEDROCK_COHERE.routed_to(NO_ENDPOINT))
        target = DeferringTargetStore(
            defer=defer,
            release_after_s=release_after_s,
            index_name=new_index(),
            url=opensearch_url,
            timeout_s=10,
            embeddings=TARGET_EMBEDDER,
        )
        targets.append(target)
        return source, target

    yield build

    for target in targets:
        target.finish_late_writes()


@pytest.fixture
def baseline_and_arm(arm_pair: ArmPair, corpus: list[IndexDocument]) -> tuple[Any, Any]:
    """A filled baseline index and an empty arm index, with settings on the arm."""
    return arm_pair(corpus)


def test_arm_is_built_from_the_same_chunks_re_embedding_only_the_vector(
    opensearch_url: str,
    baseline_and_arm: tuple[Any, Any],
    corpus: list[IndexDocument],
    anchored: AnchoredFixture,
) -> None:
    """test-plan §21: Arm built from the same chunks."""
    from evaluation.build_embedding_arm import build_embedding_arm

    source, target = baseline_and_arm
    per_ticker = Counter(document.metadata["ticker"] for document in corpus)

    counts = build_embedding_arm(
        source, target, fixture=anchored, timeout_s=TIMEOUT_S, poll_s=POLL_S
    )

    assert counts == dict(per_ticker)
    assert source.count_documents() == len(corpus)
    for ticker, expected in per_ticker.items():
        filters = SearchFilters(ticker=ticker)
        assert source.count_documents(filters) == target.count_documents(filters) == expected
    copied = {document.id: document for document in target.iter_documents()}
    original = {document.id: document for document in source.iter_documents()}
    assert copied.keys() == original.keys() == {document.id for document in corpus}
    for chunk_id, document in copied.items():
        assert document.text == original[chunk_id].text
        assert original[chunk_id].metadata["embedding_id"] == OLLAMA.identifier
        assert document.metadata["embedding_id"] == BEDROCK_COHERE.identifier
        unchanged = {k: v for k, v in document.metadata.items() if k != "embedding_id"}
        assert unchanged == {
            k: v for k, v in original[chunk_id].metadata.items() if k != "embedding_id"
        }
        vector = raw_vector(opensearch_url, target.index_name, chunk_id)
        assert len(vector) == BEDROCK_COHERE.dimension
        assert vector == pytest.approx(TARGET_EMBEDDER.embed_documents([document.text])[0])


def test_build_fails_when_the_arm_ends_with_different_chunk_counts_than_the_baseline(
    baseline_and_arm: tuple[Any, Any],
    corpus: list[IndexDocument],
    anchored: AnchoredFixture,
) -> None:
    """test-plan §21: Arm built from the same chunks (counts must match per ticker)."""
    from evaluation.build_embedding_arm import build_embedding_arm

    source, target = baseline_and_arm
    stray = IndexDocument(
        id="stray-chunk-not-in-the-baseline",
        text="a leftover chunk from an earlier build",
        metadata={**corpus[0].metadata, "chunk_id": "stray-chunk-not-in-the-baseline"},
    )
    target.add_documents([stray])

    with pytest.raises(Exception) as failed:  # noqa: B017 - the failure type is not part of the contract
        build_embedding_arm(source, target, fixture=anchored, timeout_s=TIMEOUT_S, poll_s=POLL_S)

    assert not isinstance(failed.value, ImportError | AttributeError | TypeError)


# ---- CM-5b: the build waits for the arm index to become visible -----------------------------


def _chunk_holding_the_anchor_of(
    case_index: int, anchored: AnchoredFixture, chunks: list[IndexDocument]
) -> IndexDocument:
    gold = anchored.cases[case_index].gold_evidence[0]
    return next(
        chunk
        for chunk in chunks
        if chunk.metadata["accession_number"] == gold.accession_number
        and chunk.metadata["section_key"] == gold.section_key
        and gold.anchor_text in chunk.text
    )


def test_build_waits_for_writes_that_become_visible_after_a_delay(
    arm_pair: ArmPair, corpus: list[IndexDocument], anchored: AnchoredFixture
) -> None:
    """test-plan §21 Arm built from the same chunks (CM-5b): delayed visibility still builds, and
    the per-ticker counts equal the baseline's."""
    from evaluation.build_embedding_arm import build_embedding_arm

    source, target = arm_pair(
        corpus,
        defer=lambda document: document.metadata["ticker"] == "MSFT",
        release_after_s=0.5,
    )
    per_ticker = Counter(document.metadata["ticker"] for document in corpus)
    assert len(per_ticker) == 2

    counts = build_embedding_arm(source, target, fixture=anchored, timeout_s=20, poll_s=0.05)

    assert counts == dict(per_ticker)
    assert target.count_documents() == len(corpus)
    target.finish_late_writes()
    assert target.late_failures == []


def test_count_shortfall_fails_at_the_timeout_naming_the_index_and_both_counts(
    arm_pair: ArmPair, corpus: list[IndexDocument], anchored: AnchoredFixture
) -> None:
    """test-plan §21 Arm built from the same chunks (CM-5b): fails naming the index if the arm
    count does not reach the source count before the timeout. Every anchor stays reachable, so the
    count is the only thing wrong."""
    from evaluation.build_embedding_arm import build_embedding_arm

    unanchored = IndexDocument(
        id="0000320193-24-000123_business_999",
        text="A closing paragraph that carries no gold anchor.",
        metadata={**corpus[0].metadata, "chunk_id": "0000320193-24-000123_business_999"},
    )
    chunks = [*corpus, unanchored]
    source, target = arm_pair(
        chunks, defer=lambda document: document.id == unanchored.id, release_after_s=None
    )
    started = time.monotonic()

    with pytest.raises(RuntimeError) as failure:
        build_embedding_arm(source, target, fixture=anchored, timeout_s=TIMEOUT_S, poll_s=POLL_S)

    assert time.monotonic() - started < TIMEOUT_S + 5
    message = str(failure.value)
    assert target.index_name in message
    assert re.search(rf"\b{len(chunks)}\b", message)
    assert re.search(rf"\b{len(corpus)}\b", message)


@pytest.mark.parametrize("decoy", ["absent", "other-section", "other-accession"])
def test_unreachable_gold_anchor_fails_at_the_timeout_naming_the_index_and_case(
    arm_pair: ArmPair,
    corpus: list[IndexDocument],
    anchored: AnchoredFixture,
    decoy: str,
) -> None:
    """test-plan §21 Arm built from the same chunks (CM-5b): fails naming the index if a fixture
    gold anchor matches no chunk before the timeout, even when the counts agree; the same text under
    another section or accession does not satisfy the anchor."""
    from evaluation.build_embedding_arm import build_embedding_arm

    held = _chunk_holding_the_anchor_of(0, anchored, corpus)
    chunks = [chunk for chunk in corpus if chunk.id != held.id]
    if decoy != "absent":
        metadata = dict(held.metadata)
        if decoy == "other-section":
            metadata["section_key"] = next(s for s in SECTIONS if s != held.metadata["section_key"])
        else:
            metadata["accession_number"] = "0000000000-24-000001"
        chunks.append(IndexDocument(id=f"{held.id}_decoy", text=held.text, metadata=metadata))
    source, target = arm_pair(chunks)
    started = time.monotonic()

    with pytest.raises(RuntimeError) as failure:
        build_embedding_arm(source, target, fixture=anchored, timeout_s=TIMEOUT_S, poll_s=POLL_S)

    assert time.monotonic() - started < TIMEOUT_S + 5
    message = str(failure.value)
    assert target.index_name in message
    assert anchored.cases[0].case_id in message


# ---- CM-5b: command line --------------------------------------------------------------------


@pytest.mark.parametrize("option", ["--source-index", "--fixture", "--timeout-s"])
def test_command_line_documents_the_fixture_and_timeout_options(
    option: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """test-plan §21 Arm built from the same chunks (CM-5b): the CLI takes --fixture and
    --timeout-s next to --source-index."""
    from evaluation.build_embedding_arm import main

    with pytest.raises(SystemExit) as exit_:
        main(["--help"])

    assert exit_.value.code == 0
    assert option in capsys.readouterr().out
