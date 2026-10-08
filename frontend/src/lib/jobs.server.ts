import "server-only";
import { cookies } from "next/headers";
import { listRuns } from "./api.server";
import type { RunSummary } from "./api-types";
import { isJobId, selectRuns } from "./validate";

// ponytail: "my runs" = job ids this browser submitted, kept in a cookie (20 max, 30 days),
// with the server's remaining runs as the fallback. Replace both with account-scoped runs when auth lands.
export const JOBS_COOKIE = "alpha_jobs";

/** Job ids submitted from this browser, newest first. */
export async function recentJobIds(): Promise<string[]> {
  return ((await cookies()).get(JOBS_COOKIE)?.value ?? "").split(",").filter(isJobId);
}

/**
 * Runs to show this viewer, newest first: their own runs that still exist, otherwise whatever
 * the server still has (`own: false`). Empty when nothing exists or the API is unreachable.
 */
export async function availableRuns(): Promise<{ runs: RunSummary[]; own: boolean }> {
  const available = await listRuns().catch(() => []);
  return selectRuns(await recentJobIds(), available);
}

/** `?job=` when valid, else the newest run available to this viewer. */
export async function resolveJobId(param: string | string[] | undefined): Promise<string | null> {
  if (typeof param === "string" && isJobId(param)) return param;
  return (await availableRuns()).runs[0]?.job_id ?? null;
}
