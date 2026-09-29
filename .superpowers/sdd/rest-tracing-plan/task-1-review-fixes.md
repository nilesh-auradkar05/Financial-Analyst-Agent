# Task 1 review fixes

Resolved both Important findings in task-1-review.md:
- All nine routes now document their actual success schema/media type. Metrics is Prometheus text; root, stats and ingestion status have concrete response models. Typed 405 is documented, and non-submission routes no longer advertise submission-only failures.
- Existing live smoke client reads API_KEY from the process environment without printing it; polls all four terminal states and exits 1 honestly on non-completed outcomes. --async-only skips the sync analysis, enabling one analysis with --skip-ingest. Compose forwards API_KEY to the API container.

Trace: docs/test-plan.md §1 REST hardening, task-1-review.md, controller-authorized maintained-caller corrections. No dotenv read/load and no tracing changes.

Verification: five newly added behavior cases failed before fixes; API integration plus smoke tests now 36 passed. Focused ruff PASS; four Doc Sync checks PASS. Mypy result recorded in final handoff. Original CRLF conventions preserved. Compose parsed as YAML without evaluating environment or loading .env. Controller plan/todo excluded.
