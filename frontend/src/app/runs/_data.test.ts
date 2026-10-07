// test-plan §17: own runs, runs summary.
import { test } from "node:test";
import assert from "node:assert/strict";
import type { JobPollResponse } from "../../lib/api-types.ts";
import { withJob } from "../../lib/validate.ts";
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

test("runs summary is computed from the listed runs", () => {
  const job = (n: number, status: JobPollResponse["status"], grounded?: number): JobPollResponse => ({
    job_id: id(n),
    ticker: "ACME",
    status,
    started_at: "2026-09-29T12:00:00Z",
    result: grounded == null ? null : {
      ticker: "ACME", company_name: "Acme", status, news_articles: [], citations: [], errors: [], missing: [], execution_time_ms: 1500,
      verification: { passed: true, total_claims: 1, cited_claims: 1, grounded_claims: 1, citation_coverage_rate: 1, grounded_claim_rate: grounded, claims: [], orphan_citations: [] },
    },
  });
  const rows = [job(3, "completed", 0.9), job(2, "evidence_missing", 0.7), job(1, "failed")].map(toRunRow);
  assert.equal(rows[0].latencyS, 1.5);
  assert.equal(rows[2].grounded, null);
  const s = summarize(rows);
  assert.equal(s.groundedClaimRate.value, "0.800");
  assert.deepEqual(s.outcomes.map((o) => [o.status, o.weight]), [["completed", 1], ["evidence_missing", 1], ["failed", 1]]);
  assert.deepEqual(s.bars.map((b) => b.bad), [true, false]);
});
