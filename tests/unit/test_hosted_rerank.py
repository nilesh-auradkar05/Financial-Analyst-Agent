"""Trace: docs/test-plan.md §25, hosted rerank and no local models (CM-6d, ADR-0011 amendment).

Reranked retrieval talks to a loopback fake of the Foundry Cohere rerank endpoint that
computes relevance from the request and lists results best first (tests/unit/fake_foundry.py).
No network beyond 127.0.0.1, no credentials.
"""

import json
import socket
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable

import pytest

from app.components.retrieval.reranked_hybrid_retrieve import (
    FoundryCohereReranker,
    RerankedHybridSearchUnavailableError,
)
from app.config import settings
from tests.rerank_api import rerank_api
from tests.unit.fake_foundry import RERANK_DEPLOYMENT, RERANK_PATH, FakeFoundry
from tests.unit.fake_retrieval_store import RISK_TEXT, FakeRetrievalStore, make_point

ROOT = Path(__file__).resolve().parents[2]
QUERY = "supply chain supplier risks"
IDENTITY = f"foundry:{RERANK_DEPLOYMENT}"
DOCUMENTS = ["The company sells devices.", RISK_TEXT]


def _score() -> list[float]:
    return FoundryCohereReranker().score(QUERY, DOCUMENTS)


def _edit_results(edit: Callable[[list[dict[str, Any]]], None]) -> Callable[[FakeFoundry], None]:
    """A variant that makes the fake edit the `results` it is about to send."""

    def configure(fake: FakeFoundry) -> None:
        fake.corrupt = lambda body: edit(body["results"])

    return configure


def _drop_an_index(results: list[dict[str, Any]]) -> None:
    results.pop()


def _repeat_an_index(results: list[dict[str, Any]]) -> None:
    results[1]["index"] = results[0]["index"]


def _index_out_of_range(results: list[dict[str, Any]]) -> None:
    results[0]["index"] = len(results) + 5


def _string_score(results: list[dict[str, Any]]) -> None:
    results[0]["relevance_score"] = "high"


def _null_score(results: list[dict[str, Any]]) -> None:
    results[0]["relevance_score"] = None


def _invalid_json(fake: FakeFoundry) -> None:
    fake.raw_reply = b"<html>502 Bad Gateway</html>"


MALFORMED: dict[str, Callable[[FakeFoundry], None]] = {
    "missing_index": _edit_results(_drop_an_index),
    "repeated_index": _edit_results(_repeat_an_index),
    "index_out_of_range": _edit_results(_index_out_of_range),
    "string_score": _edit_results(_string_score),
    "null_score": _edit_results(_null_score),
    "invalid_json": _invalid_json,
}

MISSING = ["azure_openai_endpoint", "azure_api_key", "azure_foundry_rerank_model"]


def test_scores_come_back_in_input_order(foundry_rerank):
    """test-plan §25: Scores in input order."""
    documents = [
        "alpha",
        "alpha beta gamma delta",
        "nothing in common",
        "alpha beta",
    ]

    scores = FoundryCohereReranker().score("alpha beta gamma delta", documents)

    # The provider lists these best first (documents 1, 3, 0, 2); the scores follow the input.
    assert scores == pytest.approx([0.25, 1.0, 0.0, 0.5])


def test_the_reranker_names_its_deployment(foundry_rerank):
    """test-plan §25: Rerank reorders candidates (the identity is `foundry:<deployment>`)."""
    assert FoundryCohereReranker().identity == IDENTITY


def test_request_follows_the_rerank_contract(foundry_rerank):
    """test-plan §25: Request contract."""
    FoundryCohereReranker().score(QUERY, DOCUMENTS)

    (request,) = foundry_rerank.requests
    assert request.path == RERANK_PATH
    assert request.bearer
    assert request.credential == "test-key"
    assert request.body["model"] == RERANK_DEPLOYMENT
    assert request.body["query"] == QUERY
    assert request.body["documents"] == DOCUMENTS
    assert request.body["top_n"] == len(request.body["documents"]) == 2


def test_an_oversized_candidate_text_is_clipped_in_the_request_not_in_the_result(foundry_rerank):
    """test-plan §25: Request contract (the document texts are clipped) with Rerank reorders
    candidates (the returned chunk keeps the store's full text)."""
    long_text = RISK_TEXT + " " + "x" * 200_000
    store = FakeRetrievalStore(
        [
            make_point("business-1", "The company sells devices.", "business"),
            make_point("risk-1", long_text, "risk_factors"),
        ]
    )

    result = rerank_api.search_reranked(
        store, query=QUERY, ticker="AAPL", filing_type="10-K", top_k=2, candidates="dense"
    )

    (request,) = foundry_rerank.requests
    sent = sorted(request.body["documents"], key=len)
    assert sent[0] == "The company sells devices."
    assert len(sent[1]) < len(long_text)
    assert long_text.startswith(sent[1])
    assert request.body["top_n"] == len(request.body["documents"]) == 2
    assert {chunk.id: chunk.text for chunk in result.chunks}["risk-1"] == long_text


@pytest.mark.parametrize("variant", MALFORMED.values(), ids=MALFORMED.keys())
def test_malformed_provider_reply_raises_and_returns_no_ranking(foundry_rerank, variant):
    """test-plan §25: Malformed provider reply."""
    variant(foundry_rerank)

    with pytest.raises(RerankedHybridSearchUnavailableError) as raised:
        _score()

    assert "test-key" not in str(raised.value)


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_provider_failure_is_retried_a_bounded_number_of_times(foundry_rerank, status):
    """test-plan §25: Provider failure bounded."""
    foundry_rerank.status = status

    with pytest.raises(RerankedHybridSearchUnavailableError) as raised:
        _score()

    assert "test-key" not in str(raised.value)
    assert len(foundry_rerank.requests) == 3  # the first attempt plus the two retries


@pytest.mark.parametrize("status", [429, 503])
def test_provider_recovering_within_the_retry_budget_reranks(foundry_rerank, status):
    """test-plan §25: Provider failure bounded (retries are real, not just a cap)."""
    foundry_rerank.fail_first = 2
    foundry_rerank.fail_status = status

    scores = _score()

    assert scores[1] == pytest.approx(1.0)
    assert len(foundry_rerank.requests) == 3


def test_unreachable_provider_raises_the_unavailable_error(foundry_rerank, monkeypatch):
    """test-plan §25: Provider failure bounded (a refused connection is a failure too)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    monkeypatch.setattr(settings.llm, "azure_openai_endpoint", f"http://127.0.0.1:{closed_port}")

    with pytest.raises(RerankedHybridSearchUnavailableError):
        _score()


@pytest.mark.parametrize("missing", MISSING)
def test_unconfigured_rerank_fails_closed_before_any_request(foundry_rerank, monkeypatch, missing):
    """test-plan §25: Fails closed when unconfigured."""
    monkeypatch.setattr(settings.llm, missing, None)

    with pytest.raises(RerankedHybridSearchUnavailableError):
        _score()

    assert foundry_rerank.requests == []


def test_lockfile_holds_no_local_model_package():
    """test-plan §25: No local model packages (uv.lock)."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    names = {package["name"] for package in lock["package"]}

    forbidden = {
        name
        for name in names
        if name
        in {
            "torch",
            "transformers",
            "sentence-transformers",
            "triton",
            "ollama",
            "langchain-ollama",
        }
        or name.startswith(("nvidia-", "cuda-"))
    }

    assert forbidden == set()


def test_importing_retrieval_and_sentiment_loads_no_local_model():
    """test-plan §25: No local model packages (imports)."""
    code = (
        "import json, sys\n"
        "import app.components.retrieval.reranked_hybrid_retrieve\n"
        "import app.services.sentiment\n"
        "import app.components.retrieval.embeddings\n"
        "import app.components.retrieval.ingestion\n"
        "banned = {'torch', 'transformers', 'sentence_transformers', 'ollama', 'langchain_ollama'}\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in banned)\n"
        "print(json.dumps(loaded))\n"
    )

    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=ROOT,
    )

    assert done.returncode == 0, done.stderr[-2000:]
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []


def test_compose_file_sets_no_ollama_variable():
    """test-plan §25: No local model packages (docker-compose.yml)."""
    lines = (ROOT / "docker-compose.yml").read_text(encoding="utf-8").splitlines()

    assert [line for line in lines if "OLLAMA_" in line] == []
