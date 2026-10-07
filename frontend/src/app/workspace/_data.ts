import { isTerminalJobStatus, type AnalysisResponse, type JobPollResponse } from "../../lib/api-types.ts";
import { executiveSummary, groupOf } from "../memos/[id]/_map.ts";

export type NodeStatus = "completed" | "running" | "degraded" | "queued";

/** The workflow as built in app/agents/graph.py: three parallel evidence nodes, then three in sequence. */
export const NODE_IDS = ["research_news", "fetch_stock", "retrieve_filings", "analyze_sentiment", "draft_memo", "verify_memo"] as const;
export type NodeId = (typeof NODE_IDS)[number];

export interface GraphNodeData {
  id: NodeId;
  title: string;
  /** Right-aligned mono badge: duration, "running" or "degraded". */
  badge: string;
  meta: string;
  status: NodeStatus;
}

export type TraceTone = "green" | "amber" | "navy" | "blue" | "body";
export interface TraceRow {
  time: string;
  agent: string;
  tone: TraceTone;
  message: string;
}

export type MemoSegment =
  | { kind: "text"; text: string }
  | { kind: "cite"; label: string; tone: "sec" | "news" }
  | { kind: "cursor" };

export type BarTone = "green" | "amber" | "navy" | "blue";

export interface WorkspaceRun {
  live: boolean;
  company: string;
  ticker: string;
  price: string;
  change: string;
  changeTone: "up" | "down" | "flat";
  runId: string;
  started: string;
  elapsed: string;
  pill: string;
  graphSubtitle: string;
  graph: GraphNodeData[];
  evidence: {
    typesLabel: string;
    sec: string;
    newsCount: string;
    newsOf: string | null;
    newsWarn: boolean;
    /** Third tile: price bars in the sample; cited sources for live runs (bars are not exposed). */
    third: { value: string; label: string };
    sentiment: string;
  };
  grounding: {
    badge: string;
    grounded: number | null;
    citation: number | null;
    gate: number;
  };
  time: { cap: string; bars: { h: number; tone: BarTone; label: string }[]; tokens: string; cost: string };
  trace: TraceRow[];
  traceNote: string | null;
  streaming: boolean;
  memo: {
    version: string;
    headline: string | null;
    body: MemoSegment[];
    href: string;
  };
}

const DASH = "—";

export const FIXTURE_RUN: WorkspaceRun = {
  live: false,
  company: "Northwind Semiconductor",
  ticker: "NWSC",
  price: "$142.18",
  change: "+3.36 (+2.42%)",
  changeTone: "up",
  runId: "r_0f3a91",
  started: "14:02:07",
  elapsed: "21.4s",
  pill: "Verifier · pass 2 of 3",
  graphSubtitle: "langgraph · 6 nodes",
  graph: [
    { id: "research_news", title: "News", badge: "degraded", meta: "tavily · 13 articles · 1 timeout", status: "degraded" },
    { id: "fetch_stock", title: "Market data", badge: "0.6s", meta: "yfinance · quote $142.18", status: "completed" },
    { id: "retrieve_filings", title: "SEC filings", badge: "1.8s", meta: "184 chunks", status: "completed" },
    { id: "analyze_sentiment", title: "Sentiment", badge: "1.1s", meta: "FinBERT · positive", status: "completed" },
    { id: "draft_memo", title: "Memo writer", badge: "9.9s", meta: "22 claims · 31 sources", status: "completed" },
    { id: "verify_memo", title: "Verifier", badge: "running", meta: "checking claims", status: "running" },
  ],
  evidence: {
    typesLabel: "4 of 4 types",
    sec: "184",
    newsCount: "13",
    newsOf: "/14",
    newsWarn: true,
    third: { value: "252", label: "Price bars" },
    sentiment: "13",
  },
  grounding: { badge: "candidate", grounded: 0.91, citation: 0.93, gate: 0.85 },
  time: {
    cap: "21.4s",
    // h = fraction of the chart height.
    bars: [
      { h: 0.31, tone: "amber", label: "news" },
      { h: 0.06, tone: "green", label: "stock" },
      { h: 0.18, tone: "green", label: "filings" },
      { h: 0.11, tone: "green", label: "sent." },
      { h: 1, tone: "green", label: "draft" },
      { h: 0.5, tone: "blue", label: "verify" },
    ],
    tokens: "41.2k",
    cost: "$0.084",
  },
  trace: [
    { time: "14:02:07.112", agent: "stock", tone: "body", message: "started" },
    { time: "14:02:07.113", agent: "filings", tone: "body", message: "started" },
    { time: "14:02:07.113", agent: "news", tone: "body", message: "started" },
    { time: "14:02:07.740", agent: "stock", tone: "green", message: "completed · 0.6s" },
    { time: "14:02:08.901", agent: "filings", tone: "green", message: "completed · 1.8s" },
    { time: "14:02:10.220", agent: "news", tone: "amber", message: "degraded · 3.1s" },
    { time: "14:02:11.320", agent: "sent.", tone: "green", message: "completed · 1.1s" },
    { time: "14:02:21.207", agent: "draft", tone: "green", message: "completed · 9.9s" },
    { time: "14:02:21.210", agent: "verify", tone: "blue", message: "started" },
  ],
  traceNote: null,
  streaming: true,
  memo: {
    version: "v2",
    headline: "Data-center demand carries the quarter; inventory is the watch item.",
    body: [
      { kind: "text", text: "Data-center revenue grew to 61% of the mix " },
      { kind: "cite", label: "S2", tone: "sec" },
      { kind: "text", text: ", while channel inventory rose for a second quarter " },
      { kind: "cite", label: "S4", tone: "sec" },
      { kind: "text", text: ". News flow leans positive " },
      { kind: "cite", label: "N3", tone: "news" },
      { kind: "cursor" },
    ],
    href: "/memos/sample",
  },
};

function fmtTime(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? DASH : d.toLocaleTimeString("en-GB", { hour12: false });
}

function fmtChange(pct: number | null | undefined): { text: string; tone: WorkspaceRun["changeTone"] } {
  if (pct == null) return { text: DASH, tone: "flat" };
  const sign = pct > 0 ? "+" : "";
  return { text: `${sign}${pct.toFixed(2)}%`, tone: pct > 0 ? "up" : pct < 0 ? "down" : "flat" };
}

const NODE: Record<NodeId, { title: string; short: string }> = {
  research_news: { title: "News", short: "news" },
  fetch_stock: { title: "Market data", short: "stock" },
  retrieve_filings: { title: "SEC filings", short: "filings" },
  analyze_sentiment: { title: "Sentiment", short: "sent." },
  draft_memo: { title: "Memo writer", short: "draft" },
  verify_memo: { title: "Verifier", short: "verify" },
};
const secs = (ms: number) => `${(ms / 1000).toFixed(1)}s`;
const fmtMs = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? DASH : `${d.toLocaleTimeString("en-GB", { hour12: false })}.${String(d.getMilliseconds()).padStart(3, "0")}`;
};

const secChunks = (r: AnalysisResponse) => r.filing_chunk_count ?? r.citations.filter((c) => groupOf(c.source_type) === "sec").length;

/** What the finished result proves about each node (used for meta text, and for status on runs with no progress). */
function facts(r: AnalysisResponse): Record<NodeId, { ok: boolean; meta: string }> {
  const miss = new Set(r.missing);
  const v = r.verification;
  const price = r.stock_data?.current_price;
  return {
    research_news: { ok: !miss.has("news"), meta: `${r.news_articles.length} articles` },
    fetch_stock: { ok: !miss.has("stock"), meta: price != null ? `quote $${price.toFixed(2)}` : "not available" },
    retrieve_filings: { ok: !miss.has("filings"), meta: miss.has("filings") ? "not available" : `${secChunks(r)} chunks` },
    analyze_sentiment: { ok: !miss.has("sentiment"), meta: r.sentiment ? `FinBERT · ${r.sentiment.overall_sentiment}` : "not available" },
    draft_memo: { ok: !!r.investment_memo, meta: v ? `${v.total_claims} claims · ${r.citations.length} sources` : DASH },
    verify_memo: { ok: v?.passed === true, meta: v ? `${v.grounded_claims}/${v.total_claims} grounded` : "not run" },
  };
}

/** Node state from the job's recorded progress; nodes that have not started are queued. */
function liveGraph(job: JobPollResponse): GraphNodeData[] {
  const f = job.result ? facts(job.result) : null;
  const byNode = new Map((job.progress ?? []).map((p) => [p.node, p]));
  return NODE_IDS.map((id) => {
    const p = byNode.get(id);
    const status: NodeStatus =
      p?.status === "running" ? "running" : f ? (f[id].ok && p?.status !== "degraded" ? "completed" : "degraded") : (p?.status ?? "queued");
    return {
      id,
      title: NODE[id].title,
      status,
      meta: f ? f[id].meta : DASH,
      badge: status === "running" ? "running" : p?.duration_ms != null ? secs(p.duration_ms) : status === "degraded" ? "degraded" : "",
    };
  });
}

function liveTrace(job: JobPollResponse): TraceRow[] {
  const rows = (job.progress ?? []).flatMap((p) => {
    const agent = NODE[p.node as NodeId]?.short ?? p.node;
    const start = { at: p.started_at, row: { time: fmtMs(p.started_at), agent, tone: "body" as TraceTone, message: "started" } };
    if (!p.ended_at) return [start];
    const tone: TraceTone = p.status === "degraded" ? "amber" : "green";
    return [start, { at: p.ended_at, row: { time: fmtMs(p.ended_at), agent, tone, message: `${p.status}${p.duration_ms != null ? ` · ${secs(p.duration_ms)}` : ""}` } }];
  });
  return rows.sort((a, b) => a.at.localeCompare(b.at)).map((r) => r.row);
}

function liveTime(job: JobPollResponse): WorkspaceRun["time"] {
  const done = (job.progress ?? []).filter((p) => p.duration_ms != null);
  const max = Math.max(...done.map((p) => p.duration_ms!), 1);
  const u = job.result?.usage;
  const tokens = u?.input_tokens != null && u?.output_tokens != null ? u.input_tokens + u.output_tokens : null;
  return {
    cap: u?.model ?? DASH,
    bars: done.map((p) => ({ h: p.duration_ms! / max, tone: p.status === "degraded" ? "amber" : "green", label: NODE[p.node as NodeId]?.short ?? p.node })),
    tokens: tokens == null ? DASH : tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : String(tokens),
    cost: DASH, // not reported by the API
  };
}

/**
 * Overlay a real job onto the view model. Anything the API does not expose
 * (cost, snapshot hashes) is a neutral placeholder — never a fixture number.
 * `now` is injectable so the mapping stays pure.
 */
export function toWorkspaceRun(job: JobPollResponse, now: number = Date.now()): WorkspaceRun {
  const r = job.result ?? null;
  const change = fmtChange(r?.stock_data?.price_change_percent);
  const price = r?.stock_data?.current_price;
  const startedMs = new Date(job.started_at).getTime();
  const elapsed =
    r?.execution_time_ms != null
      ? `${(r.execution_time_ms / 1000).toFixed(1)}s elapsed`
      : Number.isNaN(startedMs)
        ? DASH
        : `${(Math.max(0, (job.completed_at ? new Date(job.completed_at).getTime() : now) - startedMs) / 1000).toFixed(1)}s elapsed`;
  const s = r?.sentiment;
  const terminal = isTerminalJobStatus(job.status);
  const statusLabel = job.status.charAt(0).toUpperCase() + job.status.slice(1);
  return {
    live: true,
    company: r?.company_name || job.ticker,
    ticker: job.ticker,
    price: price != null ? `$${price.toFixed(2)}` : DASH,
    change: change.text,
    changeTone: change.tone,
    runId: job.job_id.slice(0, 8),
    started: fmtTime(job.started_at),
    elapsed: elapsed.replace(/ elapsed$/, ""),
    pill: `Status · ${statusLabel}`,
    graphSubtitle: "langgraph · 6 nodes",
    graph: liveGraph(job),
      evidence: {
      typesLabel: !r ? DASH : r.missing.length ? `Missing ${r.missing.join(", ")}` : "4 of 4 types",
      sec: !r ? DASH : r.missing.includes("filings") ? "Not Available" : String(secChunks(r)),
      newsCount: r ? String(r.news_articles.length) : DASH,
      newsOf: null,
      newsWarn: false,
      third: { value: r ? String(r.citations.length) : DASH, label: "Cited sources" },
      sentiment: s ? String(s.positive_count + s.negative_count + s.neutral_count) : DASH,
    },
    grounding: {
      badge: job.status,
      grounded: r?.verification?.grounded_claim_rate ?? null,
      citation: r?.verification?.citation_coverage_rate ?? null,
      gate: 0.85,
    },
    time: liveTime(job),
    trace: liveTrace(job),
    traceNote: job.error ?? (terminal ? "No step timings were recorded for this run." : "Waiting for the first step…"),
    streaming: !terminal,
    memo: {
      version: DASH,
      headline: null,
      body: [{ kind: "text", text: (r && executiveSummary(r)) ?? DASH }],
      href: `/memos/${job.job_id}`,
    },
  };
}
