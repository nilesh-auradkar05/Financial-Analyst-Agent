import { isTerminalJobStatus, type JobPollResponse } from "../../lib/api-types.ts";

export type NodeStatus = "completed" | "running" | "degraded" | "queued";
export type NodeVariant = "default" | "sealed" | "progress" | "gate";

export interface GraphNodeData {
  id: "orchestrator" | "market" | "sec" | "news" | "sentiment" | "snapshot" | "writer" | "verifier" | "publish";
  title: string;
  /** Right-aligned mono badge (timing, version, "degraded", "sealed", "14/22"). */
  badge: string;
  meta: string;
  status: NodeStatus;
  variant: NodeVariant;
  /** 0..1 progress bar (verifier). */
  progress?: number;
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
  reviseLabel: string;
  gateLabel: string;
  evidence: {
    typesLabel: string;
    sec: string;
    newsCount: string;
    newsOf: string | null;
    newsWarn: boolean;
    priceBars: string;
    sentiment: string;
  };
  grounding: {
    badge: string;
    grounded: number | null;
    citation: number | null;
    gate: number;
  };
  time: { cap: string; bars: { h: number; tone: BarTone }[]; tokens: string; cost: string };
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
  graphSubtitle: "langgraph · 8 nodes · reviewer loop",
  graph: [
    { id: "orchestrator", title: "Orchestrator", badge: "0.2s", meta: "plan · fan-out 4", status: "completed", variant: "default" },
    { id: "market", title: "Market data", badge: "0.6s", meta: "yfinance · 252d OHLCV", status: "completed", variant: "default" },
    { id: "sec", title: "SEC filings", badge: "1.8s", meta: "edgartools · 184 chunks", status: "completed", variant: "default" },
    { id: "news", title: "News", badge: "degraded", meta: "tavily · 14 art · 1 timeout", status: "degraded", variant: "default" },
    { id: "sentiment", title: "Sentiment", badge: "1.1s", meta: "FinBERT · net +0.31", status: "completed", variant: "default" },
    { id: "snapshot", title: "EvidenceSnapshot", badge: "sealed", meta: "sha 9f3a…c21e · RAG top-k 12", status: "completed", variant: "sealed" },
    { id: "writer", title: "Memo writer", badge: "v2", meta: "22 claims · 31 cites", status: "completed", variant: "default" },
    { id: "verifier", title: "Verifier", badge: "14/22", meta: "", status: "running", variant: "progress", progress: 14 / 22 },
    { id: "publish", title: "Publish memo", badge: "", meta: "gate: grounded ≥ 0.85", status: "queued", variant: "gate" },
  ],
  reviseLabel: "revise · 2 claims",
  gateLabel: "on pass",
  evidence: {
    typesLabel: "4 of 4 types",
    sec: "184",
    newsCount: "13",
    newsOf: "/14",
    newsWarn: true,
    priceBars: "252",
    sentiment: "13",
  },
  grounding: { badge: "candidate", grounded: 0.91, citation: 0.93, gate: 0.85 },
  time: {
    cap: "90s cap",
    // h = fraction of the chart height (62.5px in the reference).
    bars: [
      { h: 0.048, tone: "green" },
      { h: 0.2, tone: "green" },
      { h: 0.312, tone: "amber" },
      { h: 0.152, tone: "green" },
      { h: 0.96, tone: "navy" },
      { h: 0.416, tone: "navy" },
      { h: 0.688, tone: "navy" },
      { h: 0.28, tone: "blue" },
    ],
    tokens: "41.2k",
    cost: "$0.084",
  },
  trace: [
    { time: "14:02:07.112", agent: "orchestr", tone: "body", message: "plan → market, sec, news, sentiment" },
    { time: "14:02:07.340", agent: "market", tone: "green", message: "OHLCV 252d · 1 evidence item" },
    { time: "14:02:08.901", agent: "sec", tone: "green", message: "10-Q 2026Q2 parsed · 184 chunks" },
    { time: "14:02:09.455", agent: "news", tone: "amber", message: "14 fetched · 1 timeout → degraded" },
    { time: "14:02:11.020", agent: "sentiment", tone: "green", message: "FinBERT 13/13 · net +0.31" },
    { time: "14:02:11.207", agent: "snapshot", tone: "navy", message: "sealed 9f3a…c21e (4 types)" },
    { time: "14:02:18.644", agent: "writer", tone: "navy", message: "draft v1 · 23 claims" },
    { time: "14:02:21.930", agent: "verifier", tone: "amber", message: "2 claims < 0.45 cos → revise" },
    { time: "14:02:27.415", agent: "writer", tone: "navy", message: "draft v2 · 22 claims · 31 cites" },
    { time: "14:02:28.002", agent: "verifier", tone: "blue", message: "checking claim 14/22 …" },
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

const PLACEHOLDER_NODES: GraphNodeData[] = FIXTURE_RUN.graph.map((n) => ({
  ...n,
  badge: n.id === "publish" ? "" : DASH,
  meta: n.id === "publish" ? FIXTURE_RUN.graph.find((g) => g.id === "publish")!.meta : DASH,
  status: "queued",
  progress: undefined,
  variant: n.variant === "progress" ? "default" : n.variant,
}));

/**
 * Overlay a real job onto the view model. Anything the API does not expose
 * (trace events, per-node timings, tokens, hashes, graph state) is a neutral
 * placeholder — never a fixture number.
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
    graphSubtitle: "langgraph · 8 nodes · reviewer loop",
    graph: PLACEHOLDER_NODES,
    reviseLabel: DASH,
    gateLabel: "on pass",
    evidence: {
      typesLabel: r?.missing?.length ? `Missing ${r.missing.join(", ")}` : DASH,
      sec: r?.missing?.includes("filings") ? "Not Available" : DASH,
      newsCount: r ? String(r.news_articles.length) : DASH,
      newsOf: null,
      newsWarn: false,
      priceBars: DASH,
      sentiment: s ? String(s.positive_count + s.negative_count + s.neutral_count) : DASH,
    },
    grounding: {
      badge: job.status,
      grounded: r?.verification?.grounded_claim_rate ?? null,
      citation: r?.verification?.citation_coverage_rate ?? null,
      gate: 0.85,
    },
    time: { cap: DASH, bars: [], tokens: DASH, cost: DASH },
    trace: [],
    traceNote: job.error ?? "Trace events are not exposed by the API yet.",
    streaming: !terminal,
    memo: {
      version: DASH,
      headline: null,
      body: [{ kind: "text", text: r?.executive_summary ?? DASH }],
      href: `/memos/${job.job_id}`,
    },
  };
}
