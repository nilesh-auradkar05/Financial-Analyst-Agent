const TICKER = /^[A-Z][A-Z0-9.\-]{0,9}$/;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function normalizeTicker(s: string): string {
  return s.trim().toUpperCase();
}

export function isTicker(s: unknown): s is string {
  return typeof s === "string" && TICKER.test(normalizeTicker(s));
}

export function isJobId(s: unknown): s is string {
  return typeof s === "string" && UUID.test(s);
}
