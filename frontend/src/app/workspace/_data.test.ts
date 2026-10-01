// test-plan §16: missing filings keep usable news, stock, sentiment and memo.
import { test } from "node:test";
import assert from "node:assert/strict";
import { isTerminalJobStatus, type JobPollResponse, type JobStatus } from "../../lib/api-types.ts";
import { toWorkspaceRun } from "./_data.ts";

test("all final job outcomes stop polling and streaming", () => {
  for (const status of ["completed", "degraded", "evidence_missing", "failed"] satisfies JobStatus[]) {
    assert.equal(isTerminalJobStatus(status), true);
  }
  for (const status of ["pending", "running"] satisfies JobStatus[]) {
    assert.equal(isTerminalJobStatus(status), false);
  }
});

test("missing filings still render available analysis", () => {
  const job: JobPollResponse = {
    job_id: "11111111-1111-4111-8111-111111111111",
    ticker: "ACME",
    status: "evidence_missing",
    started_at: "2026-09-29T12:00:00Z",
    completed_at: "2026-09-29T12:00:10Z",
    result: {
      ticker: "ACME",
      company_name: "Acme",
      status: "evidence_missing",
      executive_summary: "Available news supports a cautious outlook.",
      investment_memo: "SEC Filings: Not Available",
      stock_data: { ticker: "ACME", company_name: "Acme", current_price: 42 },
      sentiment: { overall_sentiment: "neutral", positive_count: 1, negative_count: 0, neutral_count: 0, average_positive_score: 0.5, average_negative_score: 0 },
      news_articles: [{ title: "Acme news", url: "https://example.com", source: "Example", snippet: "News", relevance_score: 0.8 }],
      citations: [],
      verification: null,
      errors: [],
      missing: ["filings"],
    },
  };
  const view = toWorkspaceRun(job);
  assert.equal(view.streaming, false);
  assert.equal(view.evidence.sec, "Not Available");
  assert.equal(view.evidence.newsCount, "1");
  assert.equal(view.evidence.sentiment, "1");
  assert.equal(view.price, "$42.00");
  assert.deepEqual(view.memo.body, [{ kind: "text", text: "Available news supports a cautious outlook." }]);
});
