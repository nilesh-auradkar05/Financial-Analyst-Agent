// test-plan §17: own runs, runs summary.
import { test } from "node:test";
import assert from "node:assert/strict";
import type { RunSummary } from "../../lib/api-types.ts";
import { summarize, toRunRow } from "./_data.ts";

const id = (n: number) => `${String(n).padStart(8, "0")}-1111-4111-8111-111111111111`;

const run = (n: number, status: RunSummary["status"], grounded?: number): RunSummary => ({
  job_id: id(n), ticker: "ACME", status, started_at: "2026-09-29T12:00:00Z",
  grounded_claim_rate: grounded ?? null, citation_coverage_rate: grounded == null ? null : 1, execution_time_ms: grounded == null ? null : 1500,
});

test("runs summary is computed from the listed runs", () => {
  const rows = [run(3, "completed", 0.9), run(2, "evidence_missing", 0.7), run(1, "failed")].map(toRunRow);
  assert.equal(rows[0].latencyS, 1.5);
  assert.equal(rows[2].grounded, null);
  const s = summarize(rows);
  assert.equal(s.groundedClaimRate.value, "0.800");
  assert.deepEqual(s.outcomes.map((o) => [o.status, o.weight]), [["completed", 1], ["evidence_missing", 1], ["failed", 1]]);
  assert.deepEqual(s.bars.map((b) => b.bad), [true, false]);
});
