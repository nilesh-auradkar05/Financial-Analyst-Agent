// Run: node --test --experimental-strip-types "src/app/memos/[id]/_map.test.ts"
import { test } from "node:test";
import assert from "node:assert/strict";
import type { AnalysisResponse } from "../../../lib/api-types.ts";
import { executiveSummary, isDisclaimer, parseInline, parseMemo, safeHttpUrl, toMemoView } from "./_map.ts";

const known = new Set(["1", "2"]);

test("marker becomes a cite node and surrounding text is preserved", () => {
  assert.deepEqual(parseInline("Rev rose [1]. Margin fell [2][1].", known), [
    { t: "text", text: "Rev rose " },
    { t: "cite", id: "1" },
    { t: "text", text: ". Margin fell " },
    { t: "cite", id: "2" },
    { t: "cite", id: "1" },
    { t: "text", text: "." },
  ]);
});

test("HTML-looking input stays literal text", () => {
  const s = '<img src=x onerror=alert(1)> <b>hi</b> [1]';
  const out = parseInline(s, known);
  assert.equal(out[0].t, "text");
  assert.equal((out[0] as { text: string }).text, "<img src=x onerror=alert(1)> <b>hi</b> ");
  assert.deepEqual(out[1], { t: "cite", id: "1" });
});

test("unknown citation index stays as literal text", () => {
  assert.deepEqual(parseInline("Claim [9] and [1, 7].", known), [
    { t: "text", text: "Claim [9] and " },
    { t: "cite", id: "1" },
    { t: "text", text: "[7]." },
  ]);
});

test("memo sections and bullets", () => {
  const s = parseMemo("## Executive Summary\nGrowth [1].\n\n**Risks**\n- One [2].\n- Two.");
  assert.equal(s[0].heading, "Executive Summary");
  assert.deepEqual(s[1].blocks, [{ kind: "ul", items: ["One [2].", "Two."] }]);
});

test("safeHttpUrl rejects non-http schemes", () => {
  assert.equal(safeHttpUrl("javascript:alert(1)"), null);
  assert.equal(safeHttpUrl("https://sec.gov/x"), "https://sec.gov/x");
});

// test-plan §17: summary source, memo header stats, disclaimer.
const base: AnalysisResponse = {
  job_id: "11111111-1111-4111-8111-111111111111",
  ticker: "ACME",
  company_name: "Acme",
  status: "evidence_missing",
  executive_summary: "Analysis completed for Acme",
  investment_memo: "# Memo\n\n---\n\n## 1. Executive Summary\n\nAcme grew revenue 12% [1]. Margins held.\n\n---\n\n*Disclaimer: not advice.*",
  news_articles: [],
  citations: [{ index: 1, source_type: "news", title: "Acme news" }],
  verification: { passed: true, total_claims: 4, cited_claims: 4, grounded_claims: 3, citation_coverage_rate: 1, grounded_claim_rate: 0.75, claims: [], orphan_citations: [] },
  errors: [],
  missing: ["filings"],
};

test("summary comes from the memo, never the API placeholder", () => {
  assert.equal(executiveSummary(base), "Acme grew revenue 12% [1]. Margins held.");
  assert.equal(executiveSummary({ ...base, investment_memo: "No sections here." }), null);
  assert.equal(executiveSummary({ ...base, investment_memo: "", executive_summary: "Cautious outlook." }), "Cautious outlook.");
  assert.equal(toMemoView(base).subtitle, "Acme grew revenue 12%.");
});

test("memo header stats come from the result and name missing evidence", () => {
  const stats = Object.fromEntries(toMemoView(base).stats.map((s) => [s.label, s]));
  assert.equal(stats["Status"].value, "evidence missing");
  assert.equal(stats["Grounded claims"].value, "0.75");
  assert.equal(stats["Citation coverage"].value, "1.00");
  assert.equal(stats["Evidence"].value, "Missing · SEC filings");
  assert.equal(stats["Evidence"].href, "/evidence?job=11111111-1111-4111-8111-111111111111");
  assert.equal(Object.fromEntries(toMemoView({ ...base, missing: [] }).stats.map((s) => [s.label, s.value]))["Evidence"], "Complete");
});

test("model-written disclaimers are classified; rules and markers are not body text", () => {
  const blocks = toMemoView(base).sections.flatMap((s) => s.blocks);
  const texts = blocks.flatMap((b) => (b.kind === "p" ? [b.text] : b.items));
  assert.ok(!texts.some((t) => t.includes("---")));
  assert.deepEqual(texts.filter(isDisclaimer), ["Disclaimer: not advice."]);
  assert.equal(isDisclaimer("Acme grew revenue 12% [1]."), false);
  assert.equal(isDisclaimer("This memo does not constitute an offer."), true);
});
