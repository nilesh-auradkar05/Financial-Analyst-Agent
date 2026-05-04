from __future__ import annotations

import pytest

from evaluation.compare_retrieval_results import (
    compare_payloads,
    validate_payload_contract,
)


def _row(case_id: str, mode: str, recall: float, rank: int | None = 1) -> dict:
    return {
        "case_id": case_id,
        "mode": mode,
        "metrics": {
            "precision_at_5": recall,
            "recall_at_5": recall,
            "mrr_at_5": recall,
            "ndcg_at_5": recall,
            "keyword_hit_rate": recall,
            "section_recall_at_k": recall,
            "first_relevant_rank": rank,
            "latency_ms": 100.0,
            "top_k": 5,
            "retrieved_count": 5,
        },
        "evidence_packets": [
            {
                "metadata": {
                    "retrieval_method": "dense_bm25_cross_encoder_rerank",
                }
            }
        ],
    }


def test_compare_payloads_joins_by_shared_case_id_only() -> None:
    baseline = {
        "results": [
            _row("shared_good", "section_aware", 1.0),
            _row("baseline_only_bad", "section_aware", 0.0),
        ]
    }
    candidate = {
        "results": [
            _row("shared_good", "reranked_hybrid", 1.0),
            _row("candidate_only_bad", "reranked_hybrid", 0.0),
        ]
    }

    comparison = compare_payloads(baseline, candidate, bootstrap_samples=0)
    recall = next(
        row for row in comparison["quality"] if row["metric"] == "recall_at_5"
    )

    assert comparison["shared_case_ids"] == ["shared_good"]
    assert comparison["baseline_only_cases"] == ["baseline_only_bad"]
    assert comparison["candidate_only_cases"] == ["candidate_only_bad"]
    assert recall["paired_delta"] == 0.0
    assert recall["wins"] == 0
    assert recall["ties"] == 1
    assert recall["losses"] == 0


def test_compare_payloads_reports_miss_rate_for_null_first_rank() -> None:
    baseline = {"results": [_row("case", "section_aware", 1.0, rank=1)]}
    candidate = {"results": [_row("case", "reranked_hybrid", 1.0, rank=None)]}

    comparison = compare_payloads(baseline, candidate, bootstrap_samples=0)
    miss_rate = next(row for row in comparison["quality"] if row["metric"] == "miss_rate")

    assert miss_rate["baseline_mean"] == 0.0
    assert miss_rate["candidate_mean"] == 1.0
    assert miss_rate["paired_delta"] == 1.0
    assert miss_rate["losses"] == 1


def test_validate_payload_contract_fails_on_wrong_mode() -> None:
    payload = {"results": [_row("case", "hybrid", 1.0)]}

    with pytest.raises(ValueError, match="expected mode"):
        validate_payload_contract(
            payload,
            expected_mode="reranked_hybrid",
            label="candidate",
        )


def test_validate_payload_contract_fails_on_wrong_method() -> None:
    row = _row("case", "reranked_hybrid", 1.0)
    row["evidence_packets"][0]["metadata"]["retrieval_method"] = "langchain_bm25_ensemble_rrf"
    payload = {"results": [row]}

    with pytest.raises(ValueError, match="expected retrieval_method"):
        validate_payload_contract(
            payload,
            expected_method="dense_bm25_cross_encoder_rerank",
            label="candidate",
        )
