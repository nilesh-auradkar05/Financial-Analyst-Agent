# test-plan.md — Behavioral Test Oracle

> **Authority:** Source-of-truth rank 2 (SPEC §0). Defines what correct behavior is. Tests assert behavior through public interfaces, never implementation shape.
> **Companion:** Benchmark definitions defer to `retrieval-benchmark.md`; this file lists behavior cases.

## Principles

Tests observe through public surfaces: API endpoints, workflow inputs/outputs, retrieval contract, ingestion fixtures, benchmark runner/comparator, and documented persistent state.

Forbidden:

- Asserting private attributes.
- Mocking internals only to prove call order. (`scripts/ci/check_test_hygiene.py` enforces this.)
- Deriving tests from generated implementation shape.
- Baked-answer stubs: a fake whose output is hand-built so the metric/assertion passes. Fakes must be faithful — compute results, do not echo the expected answer.
- Editing the oracle to fit bad implementation.

Every test must trace to a behavior case in this file. Retrieval *quality* is owned by the benchmark (`retrieval-benchmark.md`), not by stub-based unit tests.

Allowed:

- Public API calls.
- Workflow input/output assertions.
- Retrieval contract assertions.
- DB/file/state assertions after public actions where documented.
- Event/handler tests only when the event contract is public.

---

## 1. API endpoints

> Route names below are re-baselined to the implemented API (G8). Behavior is unchanged; only endpoint paths track the code.

| Case | Expected behavior |
| --- | --- |
| `POST /analyze` valid ticker | returns memo + evidence + verification + diagnostics |
| `POST /analyze/async` valid ticker | returns `job_id`; status reachable via `/jobs/{job_id}` |
| `GET /jobs/{job_id}` running | returns status `running`; no partial memo claimed final |
| `GET /jobs/{job_id}` terminal | returns final payload or safe error |
| `GET /jobs` (run summaries) | **not implemented** — no list endpoint today; deferred to S7 service-readiness, not a current oracle case |
| `GET /health` | returns service + backend health |
| `GET /stats` | returns documented runtime stats |
| `GET /metrics` | returns runtime metrics |
| Invalid ticker format | rejected with validation error; no run started |
| Unsupported input | 4xx typed error, never stack trace |

---

## 2. Workflow behavior

| Case | Expected behavior |
| --- | --- |
| `include_filing_analysis=false` | no SEC retrieval; response marks filing analysis disabled; no SEC citations |
| `include_news_sentiment=false` | no news/sentiment step; response marks disabled; no fabricated sentiment |
| `max_news_articles=N` | news retrieval bounded by N |
| Evidence-before-memo | memo generation occurs only after evidence context is built |
| Verification downstream | verification runs on produced memo, not before generation |
| Empty retrieval | memo states limitation; verifier reports low/no support |
| Tool failure | run failed/degraded with safe error; result persisted |
| LLM failure | run failed with safe error; no partial memo returned as final |
| Citations | every memo citation resolves to returned `evidence_id` |

---

## 3. Ingestion: S1

| Case | Expected behavior |
| --- | --- |
| Section coverage AAPL/MSFT/NVDA × critical sections | each present critical section yields >0 chunks through store interface |
| Missing optional section | explicit skip/xfail with reason; never silent pass |
| Deterministic `chunk_id` | `chunk_id = f(accession, section_key, chunk_index)` |
| Idempotent re-ingest | same filing yields identical `chunk_id`s, no duplicates |
| Metadata normalization | ticker uppercase; filing_type, filing_date, accession_number, source_url, section_key present and canonical |
| Backend-invariant fields | identical metadata regardless of backend |

---

## 4. Retrieval contract: S3-T04 parity suite

Runs under both Chroma and Qdrant.

| Case | Expected behavior |
| --- | --- |
| Same corpus, both backends | identical total `count_documents()` |
| Same filter | identical filtered count |
| Same query + filter | metadata-valid `EvidencePacket`s on both backends |
| No matches | returns empty list, never raises |
| Invalid filter | validation error |
| Missing-metadata upsert | rejected |
| Re-upsert same chunks | no duplicate chunk IDs |
| Backend client leakage | no backend client object observable above `RetrievalService` |

---

## 5. Qdrant-specific: S3

| Case | Expected behavior |
| --- | --- |
| `VECTOR_BACKEND=qdrant` | resolves to `QdrantVectorStore` |
| Invalid `VECTOR_BACKEND` | config validation fails |
| Qdrant unavailable | `healthcheck()` reports unavailable; no hang/storm |
| Payload index creation | idempotent; indexes present for ticker, filing_type, section_key, filing_date, accession_number |
| Filtered count/search | only matching metadata returned |
| Coverage under Qdrant | S1 section-coverage + idempotency pass with Qdrant |

---

## 6. Benchmark: S2

Definitions live in `retrieval-benchmark.md`.

### Fixture validator

| Case | Expected behavior |
| --- | --- |
| Valid fixture | passes |
| Duplicate `case_id` | fails |
| Missing `gold_evidence` | fails |
| Unknown `section_key` | fails |
| `relevance` not in `{1,2}` | fails |
| `anchor_text` < 8 words | fails |
| Anchor absent from source cache | fails with case_id + anchor |
| `expected_sections` mismatch | fails |
| Fixture below `--min-cases` | fails |
| Invalid ticker format | fails |

### Benchmark runner

| Case | Expected behavior |
| --- | --- |
| Run on fixture | result contains every `case_id` |
| Method/mode labels | present and validated |
| Null first relevant rank | emitted as `null` |
| Provenance | retrieved chunk ids/section/score preserved |
| Cold/warm latency | captured separately |
| Determinism | same fixture + backend + method gives identical metrics |

### Paired comparator

| Case | Expected behavior |
| --- | --- |
| Mismatched `fixture_version` | rejected |
| Both backend and method differ | rejected, single-axis rule |
| Missing case under `--strict-case-ids` | fails |
| Mismatched mode/method when held constant | fails |
| Null first relevant rank | counted as miss |
| Output | paired deltas, win/tie/loss, bootstrap CI, separated latency |

---

## 7. RAG / memo & verification

| Case | Expected behavior |
| --- | --- |
| Grounded claim | tied to retrieved evidence where possible |
| Citation mapping | each citation maps to an `evidence_id` |
| Missing evidence | memo explicitly says evidence is missing |
| Unsupported claim | verifier flags it |
| Personalized advice request | no buy/sell directive |
| Language | cautious analytical phrasing; no guaranteed return language |

---

## 8. Config

| Case | Expected behavior |
| --- | --- |
| Typed settings load | valid config loads; invalid values fail startup |
| Backend selection | `VECTOR_BACKEND` controls store through DI |
| Embedding identity | model/version recorded in ingest metadata and result files |
| Secrets | no `.env` read/printed; no hardcoded credentials |

---

## 9. Documentation-governance checks: S0

| Case | Expected behavior |
| --- | --- |
| No scope residue | `check_no_scope_residue.py` passes against core Alpha docs |
| Sprint map sync | `check_sprint_map.py` confirms SPEC and sprint-plan use S0–S10 |
| Agent manuals sync | `check_doc_sync.py` confirms CLAUDE/AGENTS preserve equivalent task loop, hard stops, coding conventions, verification, correction handling |
| Diagram folder split | HTML files live in `docs/design-html`; Markdown companions live in `docs/design-md` |
