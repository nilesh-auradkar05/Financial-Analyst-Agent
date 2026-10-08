import type { JobStatus, RunSummary } from "../../lib/api-types.ts";

// RUNS/SUMMARY/OUTCOMES/GROUNDED_BARS are the design sample, shown only when this browser has no
// runs. Real rows come from toRunRow(); model, cost and snapshot id are not exposed by the API.

export type RunStatus = JobStatus;

export const STATUSES: readonly RunStatus[] = ["completed", "degraded", "evidence_missing", "failed", "running"];

export type RunRow = {
  id: string;
  ticker: string;
  snapshot: string | null;
  model: string | null;
  status: RunStatus;
  grounded: number | null;
  coverage: number | null;
  latencyS: number | null;
  costUsd: number | null;
  /** Live runs link to their workspace and memo; sample rows do not. */
  jobId?: string;
};

export function toRunRow(job: RunSummary): RunRow {
  const ms = job.execution_time_ms;
  return {
    id: job.job_id.slice(0, 8),
    jobId: job.job_id,
    ticker: job.ticker,
    snapshot: null,
    model: null,
    status: job.status,
    grounded: job.grounded_claim_rate ?? null,
    coverage: job.citation_coverage_rate ?? null,
    latencyS: ms != null ? ms / 1000 : null,
    costUsd: null,
  };
}

const COLOR: Partial<Record<RunStatus, string>> = { completed: "#0b7a55", degraded: "#c47a0a", evidence_missing: "#b42318", failed: "#7a271a" };
export const GATE = 0.85;

function meanSd(xs: number[]): { value: string; pm: string } {
  if (!xs.length) return { value: "—", pm: "" };
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length);
  return { value: mean.toFixed(3), pm: xs.length > 1 ? `± ${sd.toFixed(3)}` : "" };
}

/** Summary tiles computed from real rows (newest first). Bars are oldest → newest, 50px tall at 1.0. */
export function summarize(rows: readonly RunRow[], own = true) {
  const nums = (k: "grounded" | "coverage") => rows.flatMap((r) => (r[k] == null ? [] : [r[k]]));
  return {
    groundedClaimRate: meanSd(nums("grounded")),
    citationCoverage: meanSd(nums("coverage")),
    tag: `${rows.length} run${rows.length === 1 ? "" : "s"} · ${own ? "this browser" : "this server"}`,
    outcomes: Object.entries(COLOR)
      .map(([status, color]) => ({ status, color, weight: rows.filter((r) => r.status === status).length }))
      .filter((o) => o.weight > 0),
    bars: nums("grounded").slice(0, 12).reverse().map((g) => ({ h: g * 50, bad: g < GATE })),
  };
}

export const RUNS: readonly RunRow[] = [
  { id: "r_0f3a91", ticker: "NWSC", snapshot: "9f3a…c21e", model: "Bedrock · Claude", status: "degraded", grounded: 0.91, coverage: 0.93, latencyS: 31.8, costUsd: 0.112 },
  { id: "r_0f3a7c", ticker: "NWSC", snapshot: "9f3a…c21e", model: "Bedrock · DeepSeek V3.2", status: "completed", grounded: 0.88, coverage: 0.9, latencyS: 27.2, costUsd: 0.041 },
  { id: "r_0f39e2", ticker: "HLXR", snapshot: "51bd…07aa", model: "Bedrock · Claude", status: "completed", grounded: 0.95, coverage: 0.96, latencyS: 29.4, costUsd: 0.104 },
  { id: "r_0f39b8", ticker: "QRTZ", snapshot: null, model: "Bedrock · Claude", status: "evidence_missing", grounded: null, coverage: null, latencyS: 3.1, costUsd: 0.002 },
  { id: "r_0f3960", ticker: "HLXR", snapshot: "51bd…07aa", model: "Ollama · fallback", status: "completed", grounded: 0.84, coverage: 0.87, latencyS: 58.6, costUsd: 0 },
  { id: "r_0f390d", ticker: "VNTA", snapshot: "c7e0…9b13", model: "Bedrock · Claude", status: "completed", grounded: 0.93, coverage: 0.94, latencyS: 30.0, costUsd: 0.098 },
  { id: "r_0f38c4", ticker: "VNTA", snapshot: "c7e0…9b13", model: "Bedrock · OpenAI", status: "completed", grounded: 0.9, coverage: 0.92, latencyS: 26.5, costUsd: 0.087 },
  { id: "r_0f3871", ticker: "ORBL", snapshot: "2a19…e4f0", model: "Bedrock · Claude", status: "degraded", grounded: 0.86, coverage: 0.89, latencyS: 34.9, costUsd: 0.109 },
  { id: "r_0f3802", ticker: "ORBL", snapshot: "2a19…e4f0", model: "Bedrock · Claude", status: "completed", grounded: 0.92, coverage: 0.95, latencyS: 28.7, costUsd: 0.101 },
];

export const SUMMARY = {
  groundedClaimRate: { value: "0.904", pm: "± 0.062" },
  citationCoverage: { value: "0.925", pm: "± 0.049" },
  tag: "candidate · live evidence",
};

/** Run-outcome segment weights (proportional widths; PNG values, not derived from the 9 rows above). */
export const OUTCOMES = [
  { status: "completed", weight: 219, color: "#0b7a55" },
  { status: "degraded", weight: 43, color: "#c47a0a" },
  { status: "evidence_missing", weight: 20.5, color: "#b42318" },
] as const;

/** Grounded-rate bars, last 12 runs: heights in CSS px, red = below threshold. */
export const GROUNDED_BARS = [
  { h: 46, bad: false },
  { h: 48.5, bad: false },
  { h: 34.5, bad: true },
  { h: 47.5, bad: false },
  { h: 49, bad: false },
  { h: 43.5, bad: true },
  { h: 46.5, bad: false },
  { h: 48.5, bad: false },
  { h: 46, bad: false },
  { h: 50, bad: false },
  { h: 44.5, bad: false },
  { h: 47.5, bad: false },
] as const;
