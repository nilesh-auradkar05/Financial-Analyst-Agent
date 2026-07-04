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