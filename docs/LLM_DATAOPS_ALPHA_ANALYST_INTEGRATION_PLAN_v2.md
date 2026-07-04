# LLM DataOps & Production Engineering Plan — Alpha Financial Analyst Agent (v2)

> **Document type:** Integration plan / companion design document
> **Status:** Draft v2 — supersedes v1 (`LLM_DATAOPS_ALPHA_ANALYST_INTEGRATION_PLAN.md`)
> **Reviewed against:** repo HEAD `6826034` (2026-07-03) — commit message claims "True rag baseline grounded_claim_rate 0.904 ± 0.062, citation_coverage 0.925 ± 0.049"; pre-flight verification below found that claim is not backed by a clean result artifact.
> **Parent source of truth:** `docs/SPEC.md`
> **Goal change in v2:** the end-state is now a **production-grade, cloud-deployed data engineering + GenAI system**. This retires the v1 §3.4 constraint ("cloud deployment out of scope") and the corresponding SPEC non-goal. That retirement is **not effective until ratified** by ADR-0006 and a SPEC §3 amendment. Until then, Phases 1–3 (all local-first) are the only actionable scope.

---

## 0. Source-of-truth position

Unchanged in spirit: `docs/SPEC.md` wins on conflict. This document proposes; ADRs ratify.

**Pre-flight correction (2026-07-04):** the v2 draft's original "no `docs/` directory" wording was imprecise, but the material finding was right: `docs/` existed in the working tree yet `.gitignore` excluded it, so `git ls-files docs/` was empty and none of the governance docs shipped. `tasks/` was also ignored, hiding the active task ledger. Phase 0 must remove those broad ignores, restore a real SPEC file, commit the docs/task ledger, and wire the governance scripts into CI.

**Action 0 (before any phase):** commit `docs/` (SPEC, sprint-plan, test-plan, retrieval-benchmark, ADRs, this document), commit `tasks/`, and make `check_doc_sync.py`, `check_sprint_map.py`, and `check_no_scope_residue.py` run in CI.

---

## 1. Current-state audit (as of commit `6826034`)

What is actually done, verified against the working tree — not asserted from memory.

| Capability | Status | Evidence |
|---|---|---|
| Grounding instrument fixes (number rounding, markdown news parsing) | **DONE** | commits `1a08689`, `6826034`; `evaluation/grounding.py`, unit tests in `tests/unit/test_grounding_eval.py` |
| Quality gates cleared **with margin** (no longer gate-hugging) | **NOT A CLEAN DATAPOINT** | The `0.904 ± 0.062` value appears in the `6826034` commit message, but the committed result artifacts do not prove it on a clean tree. `quality_baseline_20260703T202116Z.json` and `quality_baseline_20260703T205240Z.json` have `git.dirty: true`; `quality_baseline_20260703T044321Z.json` is clean but records commit `1a08689` and a lower `grounded_claim_rate 0.786 ± 0.051`. Phase 1 must establish a new frozen-evidence baseline before any approved metric is cited. |
| Commit traceability inside the eval instrument | **DONE** | `_git_state()` recorded in every `quality_baseline` result — the dirty-tree lesson is now enforced by code |
| Qdrant behind `RetrievalStore`, default backend | **DONE** | `app/components/retrieval/qdrant_store.py`; `VECTOR_BACKEND` default `qdrant`; `docker-compose.qdrant.yml` (ratified per S0-T04 G4 decision) |
| Hybrid / reranked / section-aware retrieval implementations | **DONE (gains unvalidated)** | `hybrid_retrieve.py`, `reranked_hybrid_retrieve.py`, `section_aware_search.py`; README states no reliable gains yet — pending shared-fixture matrix |
| Shared retrieval benchmark fixtures | **PARTIAL** | `retrieval_shared_benchmark_v1.json` (72 cases), `v2_candidate.json` (96 cases). Both are **keyword/section-based, not content-anchored**; no `source_sections/` cache exists. Gate A methodology (anchored, graded) is not satisfied by either |
| Benchmark run provenance | **PARTIAL** | `run_shared_retrieval_benchmark.py` writes a per-run `manifest.json` — an embryonic release manifest, not registered anywhere |
| Paired comparator + fixture validator | **DONE** | `compare_retrieval_results.py` (`--strict-case-ids`), `validate_retrieval_fixture.py` |
| Answer-quality judge harness | **DONE (config unversioned)** | `run_rag_quality_eval.py`, `judge_models_interface.py` (RAGAS/DeepEval, `judge_config_from_env`) — judge config comes from env, not from a versioned artifact |
| Observability (local) | **DONE** | Prometheus + Grafana dashboards under `monitoring/`, wired via docker-compose |
| CI | **PARTIAL** | `.github/workflows/ci.yml`: ruff, mypy, pytest. Phase 0 wires doc-governance scripts into CI. |
| Containerization | **DONE** | `Dockerfile`, `docker-compose.yml`, `docker-compose.qdrant.yml`, `Makefile` |
| **Evidence fixture freeze / replay** | **NOT DONE** | `quality_baseline.py` runs the live agent — SEC, news, market data, and FinBERT are fetched fresh every run. The ±0.062 stdev still contains uncontrolled feed variance. This remains the open architectural prerequisite |
| Source snapshots / lineage | **NOT DONE** | no snapshot contracts or storage anywhere in the tree |
| Dataset/index release registry | **NOT DONE** | results accumulate as timestamped JSON in `evaluation/quality_res/` with no lifecycle |
| Governance docs in repo | **NOT TRACKED BEFORE PHASE 0** | `docs/` existed but was ignored by `.gitignore`; `docs/SPEC.md` in the working tree was a sprint-plan duplicate, not a product spec. Phase 0 restores and tracks the governance set. |
| Cloud deployment / IaC | **NOT DONE** | none |

**Audit conclusion:** the grounding instrument fixes are committed, but the margin claim is not yet a clean datapoint. Traceability discipline is partially enforced by the instrument itself, and it caught the problem: the old headline baseline was produced on a dirty tree and live-changing evidence. The two structural gaps that remain are exactly the two things v1 under-served: the **evidence freeze** (v1 ignored non-SEC evidence entirely) and a **single versioning system** (v1 proposed a parallel one).

---

## 2. Pitfalls in v1, corrected in v2

**P1 — v1 governed the wrong data.** The `SourceSnapshot` contract was SEC-shaped (`accession_number`, `filing_type`), while the feeds that actually destroyed baseline comparability (news, market data, FinBERT sentiment — pass_rate swinging 0.67 → 0.93 → 0.67 on identical config) were left live and ungoverned. **Fix:** one generalized `EvidenceSnapshot` contract covering all four evidence types. The evidence fixture freeze is not a separate mechanism — it *is* an evidence snapshot release consumed in replay mode.

**P2 — v1 built a parallel versioning system.** The repo already has `fixture_version` conventions, per-run `manifest.json`, and result-metadata rules. v1's registry would have duplicated all three. **Fix:** the registry *generalizes* the existing artifacts. The shared-benchmark `manifest.json`, the fixture files, and the `quality_res/` baselines become registered release types. One system, one lifecycle.

**P3 — v1 violated its own "no speculative abstractions" rule.** A `registry/` package with three modules, a `lineage/` package with an event system, six report writers, six CLI commands, a `created_by` field — for a solo repo. **Fix:** four files, one JSONL registry, lineage as manifest fields (`parent_release_ids`), one report per gate. Structure grows when need is demonstrated, not before.

---

## 3. Revised architecture

### 3.1 The supply chain (unchanged shape, generalized inputs)

```text
ALL evidence sources (SEC + news + market data + sentiment)
  -> immutable evidence snapshots
  -> validated canonical records
  -> quality gates
  -> deterministic chunk / evidence releases
  -> vector index releases (Qdrant default, Chroma parity)
  -> retrieval benchmark releases (anchored, deterministic)
  -> answer-eval releases (judge config versioned)
  -> runtime and eval consume only approved, active releases
```

### 3.2 Generalized evidence contract

```python
class EvidenceSnapshot(BaseModel):
    snapshot_id: str                     # deterministic: f(source_type, natural_key, payload_hash)
    source_type: Literal["sec_filing", "news_article", "market_quote", "sentiment_score"]
    ticker: str
    natural_key: str                     # accession_number | article URL | quote timestamp+symbol | model+input_hash
    payload_hash: str
    fetched_at: datetime
    fetcher_name: str                    # e.g. edgartools_sec_extractor, web_search_tool
    fetcher_version: str
    storage_uri: str                     # local path now; s3:// URI in Phase 4
    metadata: dict[str, str] = {}        # filing_type/filing_date for SEC; publisher/published_at for news; etc.
```

Hard rules (kept from v1, now applied to *all* evidence):
- snapshots are immutable; payload change ⇒ new snapshot,
- every downstream artifact references the snapshot IDs it derives from,
- an **evidence snapshot release** = the pinned set of snapshots for a ticker universe; `quality_baseline.py` gains `--evidence-release <name:version>` to replay it instead of live fetching.

The SEC-specific `CanonicalSectionRecord` and `EvidenceChunkRecord` from v1 are kept as-is (they were correct), with `source_snapshot_id` → `snapshot_id`. Deterministic chunk identity remains `chunk_id = f(accession_number, section_key, chunk_index)`.

### 3.3 One release manifest, one registry

```python
class DatasetReleaseManifest(BaseModel):
    dataset_name: str
    dataset_version: str
    release_type: Literal[
        "evidence_snapshot",        # NEW — subsumes the fixture freeze
        "canonical_sections",
        "evidence_chunks",
        "vector_index",
        "retrieval_benchmark",
        "answer_eval",
        "quality_baseline",         # NEW — registers what quality_res/ already produces
    ]
    release_status: Literal["candidate", "approved", "rejected", "deprecated"]
    snapshot_ids: list[str]
    config_hash: str
    code_version: str               # git hash — the instrument already computes this
    created_at: datetime
    quality_report_uri: str
    artifact_uri: str
    parent_release_ids: list[str]   # this IS the lineage system; no event bus
```

Registry mechanics: append-only `artifacts/dataops/releases.jsonl` + `artifacts/dataops/active/*.yaml` pointers. No database until Phase 4, and then only if JSONL demonstrably hurts.

Active index pointer (kept from v1 §12, one correction — record the embedding model actually in use):

```yaml
active_vector_index:
  dataset_name: alpha-qdrant-index
  dataset_version: 0.1.0
  backend: qdrant
  collection_name: alpha_analyst_sec_v001
  evidence_chunk_release: alpha-evidence-chunks:0.1.0
  embedding_model: <record the configured model id, do not hardcode>
  approved_at: <set at approval>
```

### 3.4 Minimal package layout (replaces v1 §7 tree)

```text
app/dataops/
  contracts.py       # EvidenceSnapshot, CanonicalSectionRecord, EvidenceChunkRecord,
                     # DatasetReleaseManifest, QualityReport — all models, one file
  registry.py        # append/read releases.jsonl; read/write active pointers
  quality_gates.py   # gate_evidence_snapshot(), gate_sections(), gate_chunks(),
                     # gate_vector_index(), gate_benchmark(), gate_answer_eval()
  cli.py             # snapshot | validate | release | activate | show
tests/unit/test_dataops_*.py
artifacts/dataops/
  releases.jsonl
  active/
  releases/<type>/<name>/<version>/{manifest.json, quality_report.json, artifact files}
```

Dependency rules (kept — the strongest part of v1):

```text
api/, agent/  -> no dataops dependency at request time; read active pointers via config only
ingestion/    -> may depend on dataops contracts and gates
evaluation/   -> may depend on dataops manifests and approved releases
rag/          -> does not depend on dataops
dataops/      -> does not import Qdrant/Chroma clients, boto3, or provider SDKs
```

### 3.5 Quality gates (v1 §6 kept, collapsed)

All v1 gate conditions are retained but implemented as one function per release type, each emitting a single `quality_report.json` stored beside the manifest. The v1 §13 six-report catalog is deleted; the gate output *is* the report. Two additions:

- **evidence_snapshot gate:** payload non-empty, hash present, natural key present, duplicate natural key without matching hash ⇒ new version not silent overwrite; news snapshots rejected if they match known garbage signatures (CAPTCHA/bot-block pages — the failure class that cost 0.14 of grounded_claim_rate before it was found by hand),
- **quality_baseline gate:** result must record `git_state` clean, model id, temperature, and the evidence release consumed; a baseline run against live evidence is registered as `candidate` at most, never `approved`.

---

## 4. Phase plan

Replaces v1's LDO-0…LDO-7. Phases 1–3 are local-first and require no SPEC change. Phase 4+ requires ADR-0006. **Phases are strictly sequential; a phase with unmet exit criteria blocks the next.**

### Phase 0 — Docs into repo *(half a day)*

- Commit `docs/` (SPEC, sprint-plan, test-plan, retrieval-benchmark, ADRs, this doc).
- Wire `check_doc_sync.py`, `check_sprint_map.py`, `check_no_scope_residue.py` into `ci.yml`.
- Exit: CI fails if governance docs and code drift.

### Phase 1 — Evidence snapshot & replay *(the open prerequisite — closes the fixture-freeze debt)*

- `contracts.py` + `registry.py` + evidence_snapshot gate, with unit tests.
- Snapshot writer wired into the four existing tools (`edgartools_sec_extractor`, `web_search_tool`, `stock_data_tool`, `sentiment`) behind a `--record-evidence` flag; no runtime behavior change by default.
- Create `alpha-evidence:0.1.0` for the current ticker universe.
- Add `--evidence-release` replay mode to `quality_baseline.py`.
- **Exit criteria:**
  - two consecutive baseline runs on the same evidence release differ only by sampling variance (stdev attributable to temperature, measured once at temp 0.3),
  - one recorded live-vs-frozen delta, so feed drift is quantified instead of contaminating every comparison,
  - `0.904 ± 0.062` is re-established as a *frozen-evidence* baseline — the first truly reproducible number the project has had.

### Phase 2 — Anchored retrieval fixture v3 + source-section cache

`v2_candidate` (96 keyword/section cases) is a precursor, not Gate A material — same verdict the sprint plan already recorded for v1.

- Build `evaluation/build_source_section_cache.py` from committed SEC snapshots (Phase 1 output — the cache is now *derived from a release*, not re-fetched).
- Author anchored, graded cases; validator checks every anchor against the cache.
- Register fixture as `retrieval_benchmark` release; runner's existing `manifest.json` becomes the registered run record.
- **Exit:** Gate A baseline generated per `retrieval-benchmark.md`; retrieval method matrix (dense vs hybrid vs reranked vs section-aware) finally answerable with paired, anchored comparisons.

### Phase 3 — Registry unification

- Register existing artifacts: fixtures, benchmark run manifests, `quality_res/` baselines, current Qdrant index build.
- Vector index gate: vector count == approved chunk count; metadata filter smoke tests; Chroma parity suite.
- Version judge config: `judge_config_from_env` output serialized into every `answer_eval` release manifest.
- Active pointers consumed by runtime config.
- **Exit:** every number quoted in the README traces to a registered, approved release; rollback = flip a pointer.

### Phase 4 — Production data engineering (cloud) — *requires ADR-0006 + SPEC §3 amendment*

Lean, solo-operable, cost-bounded. The DataOps layer built in Phases 1–3 is what makes this deployable *safely* — releases and gates already exist; the cloud only changes where they live and what triggers them.

- **Storage:** evidence snapshots + release artifacts to S3 (versioned bucket); `releases.jsonl` stays the registry until it hurts, then Postgres (RDS) — not before.
- **Vector store:** Qdrant Cloud free/starter tier, or Qdrant container on the same host as the API. No self-managed cluster.
- **Compute:** API container on one managed runtime (ECS Fargate or Cloud Run — pick one, write ADR). No Kubernetes.
- **Orchestration:** scheduled ingestion via GitHub Actions cron invoking the dataops CLI. Dagster/Prefect only if DAG complexity demonstrably outgrows cron. **Airflow explicitly rejected** — it is the "plumbing museum" v1 warned about, at monthly-cost scale.
- **IaC:** Terraform, single environment, one `terraform apply` from zero. No manually clicked resources.
- **Secrets:** SSM Parameter Store / Secret Manager; zero secrets in repo or env files.
- **CI/CD:** deploy on git tag; pipeline runs the frozen-evidence eval and **blocks deploy if `grounded_claim_rate` regresses beyond a stated tolerance vs the active approved baseline**. This gate is the single most production-grade line on the whole roadmap.
- **Observability:** ship the existing Prometheus/Grafana stack or swap to managed equivalents; add token/cost metrics per run.
- **Cost ceiling:** ADR-0006 must state a monthly budget and the teardown path (`terraform destroy` leaves only S3 artifacts). A portfolio project that silently bills $200/month is a liability, not a credential.
- **Exit:** public endpoint serving `/analyze` from an approved index; a merged PR that regresses grounding *cannot deploy*; infra reproducible from clean checkout.

### Phase 5 — Production LLMOps

- Scheduled eval against frozen evidence release; drift report comparing live-feed evidence hash churn vs frozen baseline.
- Alerting on eval-gate failures and cost anomalies.
- Periodic evidence release refresh procedure: new snapshot release → candidate baseline → paired comparison → approve/rollback. (Model or prompt changes ride the same lifecycle.)
- **Exit:** the system can answer "what data, what code, what model produced this memo, and how good was that configuration" for any memo ever served.

---

## 5. Cut from v1, and why

| v1 item | Disposition |
|---|---|
| SEC-only `SourceSnapshot` | **Replaced** by generalized `EvidenceSnapshot` (P1) |
| `registry/` (3 modules), `lineage/` package, `reports/` (2 writers), 6-command CLI | **Cut** to 4 files, JSONL, manifest-field lineage, 5 subcommands (P3) |
| §13 six-report catalog | **Cut** — each gate emits one report |
| `created_by` field | **Cut** — solo repo |
| LDO-6 answer-eval governance as its own sprint | **Folded** into Phase 3 — the judge harness already exists; it only needs config versioning |
| LDO-7 "portfolio observability" (HTML reports, screenshots, read-only FastAPI endpoint) | **Cut** — Grafana already exists; Phase 4 observability supersedes it; the FastAPI release endpoint was scope-creep seed |
| §17 README positioning + demo script | **Cut** to: update README after Phase 3 with registered-release numbers only |
| "Cloud as documented future extension" | **Promoted** to Phase 4 as committed end-state, gated by ADR-0006 |
| §8.2 sprint-mapping table | **Replaced** by the phase plan; Phases 1–3 slot into the existing S1/S2 arc (Phase 2 *is* S2's fixture work, executed on Phase 1's snapshots) |

Kept intact from v1 because they were right: dependency rules, deterministic chunk identity, release lifecycle (`candidate → approved → active → deprecated`), active pointer pattern, "runtime stays boring," hard stops for AI coding agents (v1 §15 — copy into `CLAUDE.md`/`AGENTS.md` verbatim, minus the cloud prohibition once ADR-0006 lands).

---

## 6. Hard rules (revised)

Stop implementation if a task:
- bypasses the `RetrievalStore` contract or imports backend clients outside `rag/`,
- uses an LLM judge for **retrieval relevance** (judges are for answer quality only; retrieval stays deterministic and anchored),
- approves a baseline produced against live evidence,
- marks a release `approved` without a passing quality report,
- creates a cloud resource outside Terraform, or deploys without the eval gate,
- adds multi-agent orchestration or long-term memory (still gated by S10/Gate E — production deployment does not reopen this),
- deletes a release instead of deprecating it.

---

## 7. ADR outlines

**ADR-0005 — Unified DataOps release registry** *(Proposed)*
Decision: adopt `EvidenceSnapshot` + `DatasetReleaseManifest` + JSONL registry as the single versioning system; existing `fixture_version` and benchmark `manifest.json` conventions become release types within it. Consequences: fixture freeze implemented as evidence snapshot releases; all published metrics trace to approved releases. Explicit non-decision: no database, no orchestration framework, no cloud.

**ADR-0006 — Production cloud deployment** *(Proposed — amends SPEC §3 non-goals)*
Decision: deploy as single-environment, Terraform-managed, eval-gated cloud service per Phase 4. States: chosen cloud/runtime, monthly cost ceiling, teardown guarantee. Explicit non-decision: no Kubernetes, no Airflow, no multi-region, no multi-agent.

---

## 8. Resume bullets (post-Phase-4 honest versions)

- Built a governed LLM evidence supply chain for a financial RAG system: immutable multi-source evidence snapshots (SEC, news, market data, sentiment), deterministic chunk releases, and a versioned release registry enabling byte-reproducible evaluation baselines.
- Diagnosed and eliminated uncontrolled live-feed variance in LLM evaluation, converting an unreproducible grounding metric into a frozen-evidence baseline (grounded claim rate 0.90, citation coverage 0.93) with instrument-enforced git traceability.
- Deployed the system to production on Terraform-managed cloud infrastructure with CI/CD eval gates that block any release regressing grounding quality against the approved baseline.
- Designed anchored, deterministic retrieval benchmarks with paired case-level comparison across Chroma and Qdrant backends and dense/hybrid/reranked/section-aware retrieval methods.

---

## 9. Immediate next actions

1. Phase 0: commit `docs/` and wire doc checks into CI. *(hours)*
2. Ratify ADR-0005; open ADR-0006 as Proposed (decide cloud provider + budget — decision needed from owner, nothing else blocks on it).
3. Phase 1: `contracts.py`, `registry.py`, snapshot writers, `--evidence-release` replay, re-baseline on frozen evidence.
4. Only then Phase 2 (anchored fixture) — which is the already-committed S2 work, now executable on governed data.
