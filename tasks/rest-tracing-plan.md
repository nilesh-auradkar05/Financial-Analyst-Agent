# REST contracts and complete request tracing

Reconstructed on 2026-09-22 after temporary worktree removal; committed implementation survives through 7ba7216. Explicit user request authorizes focused S7 / S2-T00d sequencing exception. Trace: SPEC §§3/5/8/11/12, test-plan §§1/2/8/11/12. Keep current URLs, single application, retrieval contracts and benchmark methods. No new dependencies or speculative infrastructure. Typed Python/Pydantic v2; behavior-first public-boundary tests. No private assertions/call-order spies. Never read/load/print .env without explicit approval. No nested subagents. Controller owns plan/todo and live request. No push/merge. Preserve existing line endings.

- [x] Task 1: REST contracts/access controls and independent review. Commits7199ff1/a3c487c. All nine routes audited; async202+Location, safe typed errors/statuses, input validation, API-key auth, rate limits, atomic async idempotency, accurate configured-provider health, OpenAPI schemas/media, Docker API_KEY and maintained smoke client. Prior independent review passed after fixes.
- [x] Task 2: Finish correlated tracing and offline review (minor HTTP final-send timing limitation recorded). Implemented a7e558a; safe timeout correction7ba7216. HTTP -> background job -> agent/compiled graph -> tools/retrieval/sentiment/embedding/native LLM -> verifier -> response. Original request_id/trace_id persisted; poll IDs independent; retries retain original IDs. Boolean modern/legacy tracing aliases, default off, one lifespan client, native parent restoration with explicit client, fault isolation and bounded flush. Actual model/provider/temp metadata; text-only memo with provider-visible reasoning retained natively. No hidden thought claims. Need verify app-owned nested decorated failures do not export raw exceptions before outer catches sanitize them.
- [ ] Task 3: Resolve full-mypy smoke module identity collision; final independent all-route/tracing audit; broad tests/lint/types and four Doc Sync checks. Execute one real bounded async AAPL request, then inspect stored native trace and save safe IDs/span summary/URL, after any required explicit runtime .env approval.

Prior reported validation:108 focused tests passed including marked integration; focused ruff/mypy and four governance checks passed. Controller broader suite before timeout correction:275 passed,2failed,2skipped. One new timeout regression is fixed in7ba7216; one inherited embeddings override failure remains. Whole-repo ruff4 inherited findings; mypy5 inherited errors in web_search_tool and evaluation quality/latency scripts, but newly added smoke tests now cause duplicate module identity collection error. Do not hide failures or change test oracle.

## Final bounded handoff — 2026-09-23

User capped corrections at two attempts and permits only gpt-6-sol subagents. That model is not exposed in this session; no substitute agents were spawned after this instruction. Existing agents are no longer active. Controller reviewed the pending second correction and performed final verification locally. No further runtime correction rounds are authorized in this task.

- [x] Preserve second correction: safe caught-tool errors and returned error messages; individual filing-query child spans.
- [x] Local final audit of all nine business routes and OpenAPI contracts. Existing action URLs retained; no claim of every possible REST convention.
- [x] Final default suite:279passed,1knownembeddingfailure,2skipped. The two offline integrations are checked separately and recorded in the review.
- [ ] Real request/export inspection: awaiting explicit runtime credential-loading approval; no live trace ID or URL exists yet.

Details and residual limits: tasks/rest-tracing-review.md. The earlier five-round skill loop is superseded by the user's two-attempt cap.
