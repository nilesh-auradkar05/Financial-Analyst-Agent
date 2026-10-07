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

// test-plan §17 (graph, evidence tiles) and §18 (frontend from progress).
const statusOf = (view: ReturnType<typeof toWorkspaceRun>) => Object.fromEntries(view.graph.map((n) => [n.id, n.status]));
const JOB = "11111111-1111-4111-8111-111111111111";
const at = (s: number) => `2026-09-29T12:00:${String(s).padStart(2, "0")}.000Z`;

test("a job in flight shows only what progress reports; nothing is invented without it", () => {
  for (const status of ["pending", "running"] satisfies JobStatus[]) {
    const view = toWorkspaceRun({ job_id: JOB, ticker: "ACME", status, started_at: at(0) });
    assert.ok(view.graph.every((n) => n.status === "queued"), status);
    assert.deepEqual(view.time.bars, []);
    assert.deepEqual(view.trace, []);
  }
  const view = toWorkspaceRun({
    job_id: JOB, ticker: "ACME", status: "running", started_at: at(0),
    progress: [
      { node: "fetch_stock", status: "completed", started_at: at(0), ended_at: at(1), duration_ms: 1000 },
      { node: "retrieve_filings", status: "degraded", started_at: at(0), ended_at: at(2), duration_ms: 2000 },
      { node: "research_news", status: "running", started_at: at(0) },
    ],
  });
  assert.deepEqual(statusOf(view), {
    research_news: "running", fetch_stock: "completed", retrieve_filings: "degraded",
    analyze_sentiment: "queued", draft_memo: "queued", verify_memo: "queued",
  });
  assert.equal(view.graph.find((n) => n.id === "fetch_stock")?.badge, "1.0s");
  assert.deepEqual(view.time.bars, [{ h: 0.5, tone: "green", label: "stock" }, { h: 1, tone: "amber", label: "filings" }]);
  assert.deepEqual(view.trace.map((t) => `${t.agent} ${t.message}`), [
    "stock started", "filings started", "news started", "stock completed · 1.0s", "filings degraded · 2.0s",
  ]);
});

test("finished graph, evidence tiles and usage reflect the result", () => {
  const result: NonNullable<JobPollResponse["result"]> = {
    ticker: "ACME",
    company_name: "Acme",
    status: "evidence_missing",
    investment_memo: "## Executive Summary\nAcme grew [1].",
    stock_data: { ticker: "ACME", company_name: "Acme", current_price: 42 },
    sentiment: { overall_sentiment: "neutral", positive_count: 1, negative_count: 0, neutral_count: 0, average_positive_score: 0.5, average_negative_score: 0 },
    news_articles: [{ title: "Acme news", url: "https://example.com", source: "Example", snippet: "News", relevance_score: 0.8 }],
    citations: [{ index: 1, source_type: "news", title: "Acme news" }],
    verification: { passed: true, total_claims: 4, cited_claims: 4, grounded_claims: 3, citation_coverage_rate: 1, grounded_claim_rate: 0.75, claims: [], orphan_citations: [] },
    errors: [],
    missing: ["filings"],
  };
  const job: JobPollResponse = { job_id: JOB, ticker: "ACME", status: "evidence_missing", started_at: at(0), result };
  // A run recorded before progress existed still resolves every node from its result.
  const missing = toWorkspaceRun(job);
  assert.deepEqual(statusOf(missing), {
    research_news: "completed", fetch_stock: "completed", retrieve_filings: "degraded",
    analyze_sentiment: "completed", draft_memo: "completed", verify_memo: "completed",
  });
  assert.equal(missing.graph.find((n) => n.id === "verify_memo")?.meta, "3/4 grounded");
  assert.equal(missing.memo.body[0].kind === "text" && missing.memo.body[0].text, "Acme grew [1].");
  assert.equal(missing.time.tokens, "—");

  const full = toWorkspaceRun({
    ...job,
    status: "completed",
    result: {
      ...result, status: "completed", missing: [], filing_chunk_count: 7,
      usage: { model: "echo-1", input_tokens: 1200, output_tokens: 300 },
      citations: [...result.citations, { index: 2, source_type: "sec_filing", title: "10-K" }, { index: 3, source_type: "sec_filing", title: "10-K" }],
    },
  });
  assert.equal(full.evidence.sec, "7");
  assert.equal(full.evidence.third.value, "3");
  assert.equal(statusOf(full).retrieve_filings, "completed");
  assert.equal(full.time.tokens, "1.5k");
  assert.equal(full.time.cap, "echo-1");
});
