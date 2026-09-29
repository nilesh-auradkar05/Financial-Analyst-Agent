# tasks/lessons.md

## 2026-06-03 — Spec-pack correction: agent files, grep, sprint map, benchmark status, diagram folders

**Correction:** User pointed out missing/ambiguous control files and requested P0/P1/P2 cleanup: fix self-failing Terraform grep, correct sprint-plan header to S5–S10, keep CLAUDE/AGENTS semantically equivalent without omitting instructions, update premature benchmark status wording, clarify benchmark authority vs sprint sequencing, add scripted doc-sync checks, and separate HTML diagrams from Markdown companions.

**Root cause:** The previous pack mixed generated residue from a different project with Alpha-specific governance, over-claimed mirror identity between agent manuals, and let documentation checks remain manual instead of executable.

**Prevention rule:** Before closing any spec-pack task, run the Doc Sync Check and verify: source-of-truth order lives only in SPEC §0, scope only in SPEC §3, benchmark semantics only in retrieval-benchmark.md, sprint IDs match S0–S10, and CLAUDE/AGENTS remain semantically equivalent on task loop, hard stops, coding conventions, verification discipline, and correction handling.

**Applied:** Yes. Updated docs, added scripts/ci checks, separated diagram folders, and converted uploaded HTML diagrams to Markdown companions.

L: An aggregate is never proof; the item-level dump is.

Symptom: read pass_rate 0.67 -> 0.93 as "the number fix worked."
Reality: the fix was proven only by P/E claims flipping NUM n->y in the per-claim
dump. The aggregate had two other variables moving simultaneously (news corpus,
dirty tree), so it could not attribute the gain to anything.
Rule: to prove a checker change, cite the specific claims that flipped. Never let
a headline metric substitute for reading the items.

L: Never benchmark on a dirty tree or a live-changing input.

Symptom: four consecutive baselines, all WORKING TREE DIRTY, all against live
news that changed between runs (FinBERT 3/2/5 -> 3/3/3 -> 3/3/4; registry
renumbered mid-arc). None reproducible; the temperature effect stayed confounded.
Rule: a baseline requires (a) a clean commit hash and (b) a frozen evidence
fixture. Missing either -> it is a probe, discarded, not a baseline. Single-axis
discipline includes the axis you did not choose: the input.

L: Validate every component of a metric, not just the tunable one.

Symptom: hand-calibrated the semantic threshold (0.45) and trusted the rest. The
number matcher (string-subset -> false negatives on rounded values) and the claim
extractor (bold headings scored as claims via the "market"/"risk" hints) were
both broken, biasing grounded_claim_rate and citation_coverage DOWN every run.
Rule: before optimizing a metric's aggregate, validate each component at the item
level — extractor, number match, threshold. A biased instrument makes every
downstream comparison lie, including the "razor-thin margin" it appears to show.

L: The riskiest claim gets the most scrutiny, not an exemption.

Symptom: considered exempting forward-looking recommendations from coverage
because "you can't cite the future."
Reality: the recommendation is where an uncitable fact (e.g. a "$300 price
target") is most dangerous and most likely to hide. Exemption blinds the metric
exactly where a user acts on it.
Rule: keep judgments in the coverage denominator; scrutinize their factual
premises hardest. Let recommendations be the explainable ceiling on coverage.
L: Confirm which plan document is the active driver before answering "what's next."

Symptom: answered "what's next" from sprint-plan.md (S3 Qdrant) when the operative
driver was LLM_DATAOPS_ALPHA_ANALYST_INTEGRATION_PLAN_v2.md (Phase 0 docs -> Phase 1
freeze). Both are source-of-truth, at different ranks and for different tracks.
Rule: at session start and before any "what's next," name the active plan doc and its
current phase/sprint explicitly; when multiple plans exist, state which one steers and
how it maps to the others before sequencing anything.

L: A passing doc-sync check can validate paths that were never committed.

Symptom: v2 audit found no committed docs/ while DOC-SETUP reported check_doc_sync.py
green and SPEC/todo reference docs/ throughout — a contradiction that means a "green"
check may have run against working-tree files that never shipped.
Rule: a check is only trustworthy if it fails when it should. Verify committed state
(git ls-files) before trusting any doc/CI check, and prove every new check goes red on
a deliberate drift before marking it done.

## 2026-09-06 — Done conditions: failing sprint-map + dirty tree

**Correction:** User rejected "done" because the working tree was uncommitted and `check_sprint_map.py` failed (`SPEC missing S0/S1/S3/S4`).

**Root cause:** The 2026-08-24 session overwrote `docs/SPEC.md` with amendment text (no S0/S1/S3/S4). This task then recorded that failure as "pre-existing, unchanged" and marked done without a commit or an explicit uncommitted-scope statement. A passing check against a dirty overlay is not verification.

**Prevention rule:** Never mark done while a Doc Sync Check is red. If the red check is outside this task, restore the canonical file so the check is green on HEAD-equivalent SPEC, then either commit this task's files or name each leftover path and why it stays out. Do not overwrite `tasks/todo.md`; append.

**Applied:** Yes. Restored `docs/SPEC.md` v1.2 from HEAD; amendment lives at `docs/SPEC-AMENDMENT-v1.3.md`. Restored HEAD `tasks/todo.md` and appended this task. Re-ran Doc Sync: sprint-map / doc-sync / scope-residue PASS. Committing only this task's artefacts.

**Verification:** `python scripts/ci/check_sprint_map.py` PASS; `check_doc_sync.py` PASS; `check_no_scope_residue.py` PASS.

L: `.claude/hooks/stop_gate.py` blocks the turn on any dirty tree. The "or state why uncommitted" clause is agent-facing text only; the hook does not parse a reason. A leftover overlay must be committed or reverted before the turn can end. Do not delete `docs/LLM_DATAOPS_ALPHA_ANALYST_INTEGRATION_PLAN_v2.md` while SPEC §0 still names it.

## 2026-09-06 — Scale interview versus practical performance testing

**Correction:** User clarified that the production companion explains how the project could handle thousands to millions of documents and requests; practical testing should remain affordable, with performance logged at each achievable level. Million-user execution is not a deliverable.

**Root cause:** The review framed full production load/soak gates without clearly separating hypothetical employer-scale acceptance from the project's practical validation budget.

**Prevention rule:** Distinguish implemented, measured, proposed and projected claims. Specify document counts, chunk counts, request rate and concurrency separately. Provide resource-capped experiments and extrapolation assumptions; never present projections as measured results or require unaffordable scale tests for an interview artifact.

**Applied:** Yes. Revised the interview, all four production diagrams and proposed ADR-0007 to distinguish affordable measured experiments from large-scale interview projections.

**Verification:** 63 Q&A entries and four SVG embeds; local links, source/export consistency, offline SVG display and visual inspection pass. Scope/sprint/doc-sync checks pass; existing test-hygiene failures at tests/unit/test_llm.py:25,32 remain recorded in the task ledger. No scale tests or runtime changes were claimed.

## 2026-09-23 — Bound correction loops and honor exact model constraints

**Correction:** User requested no more than two correction attempts and subagents only with gpt-6-sol at high effort.

**Root cause:** Repeated implementation/review cycles expanded the verification handoff instead of converging on a bounded result; earlier available-model substitution no longer matches the tightened instruction.

**Prevention rule:** Count correction attempts explicitly, perform one final review/check after the cap, and report residual findings without another fix loop. Never substitute a model after an exact model-only restriction; if unavailable, disclose it and perform authorized remaining work locally.

**Applied:** Yes. Preserved the second correction, spawned no additional agents, made no further runtime fixes, and performed the final all-route audit and clean-environment suite locally.

**Verification:** Final suite279passed,1knownembeddingfailure,2skipped; all nine OpenAPI route contracts passed. Live credential approval remains pending.

## 2026-09-28 — S8 frontend pulled forward by explicit user override

**Correction:** User asked for the frontend while S8 was marked "Horizon: do not start; gated". Surfaced as a §6 hard stop; user chose to override and start S8 now without re-sequencing SPEC/sprint-plan.

**Root cause:** Not an agent error; a sequencing override. Recorded so later sessions don't treat `frontend/` as scope drift or re-open the gate question.

**Prevention rule:** Gated sprint work starts only on an explicit, recorded user override naming the sprint; the override does not alter SPEC §3 scope or sprint order unless the user asks for that too.

**Applied:** Yes. Task S8-FE added to tasks/todo.md with the trace and user-confirmed decisions.

**Verification:** Recorded in the S8-FE entry in tasks/todo.md.


## 2026-09-29 — Preserve credential names through deployment configuration

**Correction:** User wanted LANGCHAIN and LANGSMITH keys kept under their existing names instead of renaming the LangSmith key.

**Root cause:** The app already accepted modern LANGSMITH aliases, but Compose forwarded only legacy LANGCHAIN variables. Earlier guidance treated that deployment omission as a reason to rename user configuration.

**Prevention rule:** Check each configuration boundary and forward supported names intact. Do not recommend moving credentials between variable names when the application already supports the intended name. Preserve legacy fallback without injecting blank higher-priority aliases.

**Applied:** Yes. Added optional LANGSMITH_TRACING, LANGSMITH_API_KEY and LANGSMITH_PROJECT pass-through entries on frontend-impl; retained all legacy entries. No credential files read or changed.

**Verification:** Dummy-only Compose check reproduced the failure before the fix and passed all five configuration cases afterward; all four Doc Sync checks passed.
