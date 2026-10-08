const TICKER = /^[A-Z][A-Z0-9.\-]{0,9}$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function normalizeTicker(s: string): string {
  return s.trim().toUpperCase();
}

export function isTicker(s: unknown): s is string {
  return typeof s === "string" && TICKER.test(normalizeTicker(s));
}

const MAX_JOBS = 20;

/** Cookie value for "my runs": `jobId` first, then prior valid ids, deduplicated and capped. */
export function withJob(existing: string | undefined, jobId: string): string {
  const ids = (existing ?? "").split(",").filter((id) => isJobId(id) && id !== jobId);
  return [jobId, ...ids].slice(0, MAX_JOBS).join(",");
}

/** Own runs that still exist; if none do, everything still available (not the viewer's own). */
export function selectRuns<T extends { job_id: string }>(ownIds: readonly string[], available: readonly T[]): { runs: T[]; own: boolean } {
  const mine = new Set(ownIds);
  const own = available.filter((r) => mine.has(r.job_id));
  return own.length ? { runs: own, own: true } : { runs: [...available], own: false };
}

export function isJobId(s: unknown): s is string {
  return typeof s === "string" && UUID.test(s);
}
