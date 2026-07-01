# AGENTS.md — Operating Manual for Code Agents (Alpha Financial Analyst Agent)

This is the operating manual for Codex and other non-Claude coding agents on this repo. Claude Code uses `CLAUDE.md`. This repo is SPEC-driven: every non-trivial change traces to a documented requirement and is verified before it is called done. `CLAUDE.md` carries semantically equivalent rules; Claude-specific mechanics live there without weakening the shared rules.

---

## 1. Identity & goal

Single-agent financial-analysis RAG system: ticker → market/news/SEC evidence → grounded, verified memo. The active arc is Qdrant migration behind a stable retrieval contract, protected by a trusted retrieval benchmark.

Scope lives in SPEC §3. Source-of-truth order lives in SPEC §0. Do not restate or improvise either one here.

The active focus is whatever sprint is marked active in `tasks/todo.md` and `docs/sprint-plan.md` (S0 closed green 2026-06-04; S1 now active). Do not work ahead of it.

---

## 2. Read order before any non-trivial work

1. This file.
2. SPEC §0 for conflict-resolution order.
3. SPEC §3 for scope.
4. Active task in `tasks/todo.md`.
5. Relevant task in `docs/sprint-plan.md`.
6. Relevant behavior cases in `docs/test-plan.md`.
7. `docs/retrieval-benchmark.md` if the task touches fixtures, metrics, retrieval comparison, or evaluation outputs.
8. Relevant ADR under `docs/adr/` if architecture is affected.
9. `tasks/lessons.md` so old mistakes do not return wearing novelty glasses.

If `CLAUDE.md` and `AGENTS.md` conflict on task loop, hard stops, conventions, verification, or correction handling, stop and surface it.

---

## 3. Task loop: every non-trivial task

1. Read the active task in `tasks/todo.md`.
2. **Trace** it to SPEC / test-plan / sprint-plan / retrieval-benchmark / ADR. No trace → hard stop.
3. Add or update the task entry before implementing.
4. **Plan first** for anything non-trivial: 3+ steps, architecture choice, migration, evaluation methodology, or multi-file change. Write the plan to `tasks/todo.md` as checkable items and confirm it before building.
5. If something goes sideways mid-task, stop and re-plan. Do not push through wreckage because momentum feels productive.
6. Write behavior-first tests when behavior changes. The test-plan is the oracle.
7. Make the smallest necessary change. No unrelated refactors, no broad formatting churn.
8. Run the relevant verification commands.
9. Run the Doc Sync Check.
10. Record command results or an honest failure summary in `tasks/todo.md`.
11. Update `tasks/sprint-review.md` at sprint boundaries.
12. Update `tasks/lessons.md` after any user correction.

Deliverables are **complete updated files**, never diffs or patches.

Use subagents, parallel helpers, or isolated work agents when the tool supports them to keep the main context clean: one focused task per helper for research, exploration, and parallel analysis. Helper agents do not get to bypass traceability or verification. Apparently even the tiny imaginary interns need supervision.

---

## 4. Verification before done

Never mark a task done without proving it works. Diff behavior against baseline when relevant. Ask whether a staff engineer would approve the change. If a command fails, record the failure honestly; do not bury it under “minor issue.”

For bug reports: find the root cause and fix it. No temporary patches. Point at logs/errors/failing tests, resolve them, verify, and record.

---

## 5. Coding conventions

- Typed Python.
- Pydantic v2 for schemas/config.
- PEP 8 / PEP 484.
- Adapters behind interfaces.
- Backend client objects never escape `rag/`.
- Thin FastAPI route handlers; business logic lives in services.
- No speculative abstractions.
- No unrelated refactors.
- No broad formatting churn.
- No hardcoded credentials.
- Do not read or print `.env` unless explicitly approved.
- Tests are behavior-first: assert through public interfaces.
- Do not assert private attributes.
- Do not mock internals just to prove call order.
- Do not derive tests from implementation shape.
- Do not change the oracle to fit a bad implementation.
- No baked-answer stubs: fakes must be faithful (compute results, not echo the expected answer).
- Every test traces to a `docs/test-plan.md` behavior case; `scripts/ci/check_test_hygiene.py` blocks call-order spying.
- Demand elegance, balanced: for non-trivial changes, pause and ask whether there is a simpler solution. If a fix feels hacky, redo it properly. Skip ceremony for small obvious fixes.

---

## 6. Hard stops

Stop and ask or report if:

- The task lacks traceability to SPEC / test-plan / sprint-plan / retrieval-benchmark / ADR.
- Source-of-truth documents conflict.
- Secrets are required, or the task needs reading/printing `.env`.
- Destructive commands are needed.
- The change alters benchmark methodology without updating `retrieval-benchmark.md` and an ADR if architectural weight exists.
- Retrieval results would be compared across mismatched fixtures or methods.
- Both backend and method differ in one retrieval comparison, violating the single-axis rule.
- Gold labels would be authored against unstable chunk IDs before S1 is verified done.
- A new architecture is introduced without an ADR.
- A dependency points away from `rag/`, violating SPEC §5.
- A judge/eval/test failure is being hidden or worked around.
- The work violates project scope in SPEC §3.
- Work is about to be marked done without recorded verification.

---

## 7. Doc Sync Check

Run before marking any task done.

1. Did the change touch scope, sprint map, source-of-truth order, benchmark methodology, task loop, hard stops, coding conventions, verification discipline, or correction handling?
2. If yes, update the single canonical home first:
   - Scope → SPEC §3.
   - Source-of-truth order → SPEC §0.
   - Benchmark methodology → `docs/retrieval-benchmark.md`.
   - Sequencing → `docs/sprint-plan.md`.
3. Keep these rule families semantically equivalent between `CLAUDE.md` and `AGENTS.md`:
   - read order and task loop,
   - hard stops,
   - coding conventions,
   - verification-before-done discipline,
   - correction handling.
4. Confirm sprint IDs in `docs/sprint-plan.md` match SPEC §12.
5. Run:
   ```bash
   python scripts/ci/check_no_scope_residue.py
   python scripts/ci/check_sprint_map.py
   python scripts/ci/check_doc_sync.py
   python scripts/ci/check_test_hygiene.py
   ```
6. Record the sync result in `tasks/todo.md`.

---

## 8. Verification commands

Run the relevant subset for the task:

```bash
# core
uv run ruff check .
uv run pyright . || uv run mypy .
uv run pytest

# document governance
python scripts/ci/check_no_scope_residue.py
python scripts/ci/check_sprint_map.py
python scripts/ci/check_doc_sync.py
python scripts/ci/check_test_hygiene.py

# ingestion / retrieval (tests/ingestion + tests/rag are created at S1/S3; today: uv run pytest tests/unit)
uv run pytest tests/ingestion tests/rag

# backend parity (parity suite created at S3-T04; today: tests/unit/test_vector_store_factory.py)
VECTOR_BACKEND=chroma uv run pytest tests/rag/test_vector_store_contract.py
VECTOR_BACKEND=qdrant uv run pytest tests/rag/test_vector_store_contract.py

# benchmark (precursor validator today; anchored validator lands at S2)
uv run python evaluation/validate_retrieval_fixture.py \
  evaluation/fixtures/retrieval_shared_benchmark_v1.json
uv run python evaluation/compare_retrieval_results.py \
  <baseline_result.json> <candidate_result.json> --strict-case-ids
```

---

## 9. Pointers

- Scope: SPEC §3.
- Source-of-truth order: SPEC §0.
- Benchmark methodology: `docs/retrieval-benchmark.md`.
- Behavioral oracle: `docs/test-plan.md`.
- Architecture/contracts/sequences: SPEC §§4–9 and diagram companions.

---

## 10. Agent-specific mechanics

If your coding tool supports subagents, parallel workers, isolated agents, or worktrees, use them for research, exploration, and parallel analysis when the task is large enough to benefit. One focused task per helper. Main agent remains responsible for integration, traceability, and verification.

Tool-specific hooks and commands are created manually outside these docs. They must not contradict the hard stops or task loop.

---

## 11. Correction handling

When the user corrects you: update `tasks/lessons.md` with correction, root cause, prevention rule, applied yes/no, and verification result. Then apply the correction. Corrections are project data; losing them is how the same bug reincarnates under a new filename.
