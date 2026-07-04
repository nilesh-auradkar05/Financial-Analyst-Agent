# retrieval-benchmark.md — Retrieval Benchmark Methodology

> **Authority:** Canonical home for fixture schema, gold-label matcher, metric formulas, validator, and paired-comparison policy. Source-of-truth rank 4 (SPEC §0); authoritative for fixture/metric/comparison questions even though sprint-plan.md controls sequencing.
> **Status:** Canonical methodology specified. This anchored, graded methodology is the **committed Gate-A target** (decision 2026-06-04, raise-code-to-spec).
> **Implemented precursor (not Gate-A trustworthy):** the current `evaluation/validate_retrieval_fixture.py` + `evaluation/fixtures/retrieval_shared_benchmark_v1.json` (72 keyword/`expected_sections` cases) are a working pre-cursor with **no content anchors, no graded relevance, and no anchor-in-source cache**. Useful for fast iteration but they do **not** satisfy this doc; any "retrieval got better" claim made on them is provisional. S2 upgrades the fixture to content-anchored graded labels (new `fixture_version`), adds the source-section cache + builder, and extends the validator with the anchor-in-source lie detector before Gate A is claimed.

The benchmark is the credibility centerpiece of this project. Its job is to let us say "retrieval got better" or "the backends tie" and mean it. The single rule behind every decision in this doc: **the oracle must be deterministic ground truth, decoupled from implementation details.** We label immutable source content, not pipeline artifacts, and we compare strictly paired.

---

## 1. Why content-anchored, graded labels

A gold label defines what counts as a relevant retrieved chunk. Labeling by exact `chunk_id` couples the oracle to chunking parameters; re-chunk and every label rots. Labeling by section alone is too coarse for filings: a 10-K `risk_factors` section can span dozens of chunks.

The robust alternative: anchor each label to **immutable source content** — accession, canonical section, and a verbatim sentence — plus a **graded relevance**. Resolve to chunks at scoring time. This decouples the oracle from chunking, makes NDCG honest, and lets the validator mechanically verify every label against source before it corrupts a benchmark.

Do **not** use an LLM judge to decide chunk relevance at scoring time. That reintroduces nondeterminism into the place that must stay fixed. LLM-as-judge is reserved for answer quality at S6.

---

## 2. Fixture schema

File: `evaluation/fixtures/retrieval_shared_benchmark_v1.json`

```json
{
  "fixture_version": "retrieval_shared_benchmark_v1",
  "generated_at": "2026-06-03T00:00:00Z",
  "embedding_model": "<model-id@version>",
  "cases": [
    {
      "case_id": "aapl-risk-001",
      "ticker": "AAPL",
      "query": "What supply-chain concentration risks does Apple disclose?",
      "filters": {
        "ticker": "AAPL",
        "filing_type": "10-K",
        "section_key": "risk_factors",
        "filing_date_from": null,
        "filing_date_to": null,
        "accession_number": null
      },
      "expected_sections": ["risk_factors"],
      "answer_intent": "risk",
      "gold_evidence": [
        {
          "accession_number": "0000320193-23-000106",
          "section_key": "risk_factors",
          "anchor_text": "<verbatim distinctive sentence from the filing>",
          "relevance": 2
        }
      ],
      "expected_keywords": ["supply", "concentration"],
      "notes": "optional"
    }
  ]
}
```

Field rules:

- `case_id` — unique.
- `ticker` — uppercase and in the known set.
- `filters` — a `SearchFilters` object; nulls allowed.
- `expected_sections` — must equal the set of `section_key`s across `gold_evidence`.
- `answer_intent` ∈ `{factual, comparison, risk, financial_performance, section_specific}`.
- `gold_evidence` — non-empty list. Each item has `accession_number`, canonical `section_key`, verbatim `anchor_text` of at least 8 words, and `relevance` ∈ `{1, 2}`.
- `expected_keywords` — optional secondary metric input.
- `embedding_model` — fixture-level; recorded so backend comparisons are embedding-matched.

---

## 3. Matcher

Scoring is chunk-independent: a label resolves to whichever chunks contain its anchor.

Normalization applied to both anchor and chunk text:

```text
norm(t) = NFKC(t)
        |> casefold
        |> map curly quotes/dashes to ascii
        |> collapse whitespace runs to a single space
        |> strip
```

Predicates:

```text
hit(c, g) = c.accession_number == g.accession_number
            AND c.section_key == g.section_key
            AND norm(g.anchor_text) is a substring of norm(c.text)

section_hit(c, g) = c.accession_number == g.accession_number
                    AND c.section_key == g.section_key
```

**Straddle rule:** an anchor counts as a hit only if it is wholly contained in a single chunk. If chunking splits the anchor sentence across chunks, it is a miss. That is acceptable because the benchmark should expose granularity failures rather than over-credit them.

---

## 4. Metrics

Computed per case at `K ∈ {5, 10}`. `R_K` is the ordered top-K retrieved chunks; `G` is the case's gold set.

```text
covered@K(g)        = exists c in R_K : hit(c, g)
pass@K              = 1 if any g in G is covered else 0
recall@K            = count(covered gold items) / count(G)
precision@K         = count(retrieved chunks that hit any gold item) / K
first_relevant_rank = min rank r where R[r] is a gold-hit chunk; null if none
mrr@K               = 1 / first_relevant_rank if first_relevant_rank <= K else 0
section_recall@K    = distinct gold (accession, section) pairs with section_hit in R_K / distinct gold section pairs
keyword_hit_rate    = expected keywords present in concatenated R_K text / expected keywords
```

**NDCG@K:** each chunk's gain is the relevance of the best uncredited anchor it satisfies. Each anchor is counted once.

```text
gain(R[i]) = max relevance(g) among uncredited gold anchors hit by R[i], else 0
DCG@K      = sum_i gain(R[i]) / log2(i + 1)
IDCG@K     = ideal DCG from gold relevance values, capped at K
NDCG@K     = DCG@K / IDCG@K, or 0 if IDCG@K == 0
```

Determinism: ties in retrieval score are broken by `evidence_id` ascending.

---

## 5. Fixture validator

File: `evaluation/validate_benchmark_fixture.py`. A fixture that fails validation never reaches a benchmark. Each failure names the offending `case_id` and rule.

Checks:

- Unique `case_id`s.
- Uppercase known ticker.
- `section_key` ∈ `{business, risk_factors, md&a, market_risk}`.
- Non-empty `gold_evidence`.
- `relevance` ∈ `{1, 2}`.
- `anchor_text` ≥ 8 words.
- `expected_sections` equals the set of gold section keys.
- `answer_intent` in allowed enum.
- `--min-cases`, default 50.
- Per-ticker and per-section coverage.
- **Anchor-in-source:** normalized anchor text must appear in normalized source-section text.

The anchor-in-source check is the lie detector. Stale, typoed, or hallucinated labels fail validation instead of silently poisoning the benchmark.

---

## 6. Source-section cache

The validator reads source text from a committed cache, not the live store.

- Path: `evaluation/fixtures/source_sections/<accession_number>_<section_key>.txt`
- Builder: `evaluation/build_source_section_cache.py`
- Parser: same `edgartools` parse as ingestion.

If parser output changes, rebuild the cache and re-validate the fixture in the same change.

---

## 7. Benchmark runner and result files

File: `evaluation/run_retrieval_benchmark.py`. Produces one result file per `(backend, retrieval_method)`.

Rules:

- Every `case_id` present.
- `retrieval_method` and `mode` labeled and validated.
- Per-case retrieved chunk provenance preserved.
- `first_relevant_rank` emitted as `null` on miss.
- Cold and warm latency captured separately.
- Determinism per §4.

---

## 8. Paired comparison policy

File: `evaluation/compare_retrieval_results.py`. This is where the prior invalid-comparison lesson is enforced.

- **Fixture match:** result files must share `fixture_version`.
- **Single-axis rule:** exactly one of `{backend, retrieval_method}` may differ; the other must match. If both differ, reject.
- **Shared case set:** `--strict-case-ids` fails if case sets differ. Always report shared/baseline-only/candidate-only counts.
- **Paired deltas:** report baseline mean, candidate mean, paired delta, relative delta.
- **Confidence:** bootstrap CI on paired delta, at least 10k resamples, 95%.
- **Win/tie/loss:** per-case sign with small tie epsilon.
- **Misses:** null first relevant rank counts as miss.
- **Latency:** reported separately, never folded into quality means.
- **Power caveat:** echo fixture coverage effective-N.

Prohibitions:

- No comparison from unpaired files, mismatched methods, or different fixtures.
- No editing fixture or oracle to improve a result.
- No claiming quality improvement outside the confidence interval.

**Gate A fixture freeze rule:** once Gate A baseline is generated, fixture edits require fixture_version bump, re-validation, and regenerated baseline before any candidate comparison.

---

## 9. Honest expectation for backend comparison

With identical embeddings and `top_k`, Chroma and Qdrant are doing nearest-neighbor lookup over the same vectors. Their quality metrics are expected to tie within the confidence interval. A tie is correct, not a failure.

The legitimate case for migrating to Qdrant is operational and forward-looking: payload-index filtering and native sparse/hybrid support needed by S5. If the default switches to Qdrant on those grounds, ADR-0003 must say exactly that. No quality fairy tales. We are not running a pitch deck.

---

## 10. Versioning

- `fixture_version` is bumped on any change to cases or labels.
- Result files carry fixture version; comparator requires a match.
- Source-section cache is versioned by accession and section path.
- `embedding_model` is recorded in fixture and result files; mismatched embedding identifiers invalidate backend comparison.
