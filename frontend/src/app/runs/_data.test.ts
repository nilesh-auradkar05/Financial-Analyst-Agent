// test-plan §17: own runs, runs summary.
import { test } from "node:test";
import assert from "node:assert/strict";
import type { RunSummary } from "../../lib/api-types.ts";
import { selectRuns, withJob } from "../../lib/validate.ts";
import { summarize, toRunRow } from "./_data.ts";

const id = (n: number) => `${String(n).padStart(8, "0")}-1111-4111-8111-111111111111`;

test("submitted job ids are remembered newest-first, deduplicated, capped and validated", () => {
  assert.equal(withJob(undefined, id(1)), id(1));
  assert.equal(withJob(`${id(1)},<script>,${id(2)}`, id(2)), `${id(2)},${id(1)}`);
  const many = Array.from({ length: 30 }, (_, i) => id(i)).join(",");
  const kept = withJob(many, id(99)).split(",");
  assert.equal(kept.length, 20);
  assert.equal(kept[0], id(99));
});

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

// test-plan §19: which runs a viewer sees after data loss.
test("viewer sees surviving own runs, else the server's runs, else nothing", () => {
  const server = [run(5, "completed", 0.9), run(4, "completed", 0.8), run(3, "failed")];
  // Two remembered runs were lost (1, 2); one survives.
  assert.deepEqual(selectRuns([id(1), id(4), id(2)], server), { runs: [server[1]], own: true });
  // None of the remembered runs survive: fall back to what the server has, flagged as not own.
  assert.deepEqual(selectRuns([id(1), id(2)], server), { runs: server, own: false });
  assert.deepEqual(selectRuns([], server), { runs: server, own: false });
  assert.deepEqual(selectRuns([id(1)], []), { runs: [], own: false });
});
