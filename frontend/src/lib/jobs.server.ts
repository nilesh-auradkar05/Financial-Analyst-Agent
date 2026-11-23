import "server-only";
import { listRuns } from "./api.server";
import type { RunSummary } from "./api-types";
import { isJobId } from "./validate";

/** The visitor's own runs, newest first (the API scopes `GET /jobs` to the caller). Empty when none or the API is unreachable. */
export async function availableRuns(): Promise<{ runs: RunSummary[]; own: boolean }> {
  return { runs: await listRuns().catch(() => []), own: true };
}

/** `?job=` when valid, else the newest run available to this viewer. */
export async function resolveJobId(param: string | string[] | undefined): Promise<string | null> {
  if (typeof param === "string" && isJobId(param)) return param;
  return (await availableRuns()).runs[0]?.job_id ?? null;
}
