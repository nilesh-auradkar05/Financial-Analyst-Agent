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

export function isJobId(s: unknown): s is string {
  return typeof s === "string" && UUID.test(s);
}
