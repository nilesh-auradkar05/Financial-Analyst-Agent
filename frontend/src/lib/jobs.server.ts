import "server-only";
import { cookies } from "next/headers";
import { getJob } from "./api.server";
import type { JobPollResponse } from "./api-types";
import { isJobId } from "./validate";

// ponytail: "my runs" = job ids this browser submitted, kept in a cookie (20 max, 30 days).
// Move to account-scoped storage when the API gains users.
export const JOBS_COOKIE = "alpha_jobs";

/** Job ids submitted from this browser, newest first. */
export async function recentJobIds(): Promise<string[]> {
  return ((await cookies()).get(JOBS_COOKIE)?.value ?? "").split(",").filter(isJobId);
}

/** This browser's runs that the API still knows about, newest first. */
export async function recentJobs(): Promise<JobPollResponse[]> {
  const settled = await Promise.allSettled((await recentJobIds()).map(getJob));
  return settled.flatMap((s) => (s.status === "fulfilled" ? [s.value] : []));
}

/** `?job=` when valid, else this browser's latest run. */
export async function resolveJobId(param: string | string[] | undefined): Promise<string | null> {
  if (typeof param === "string" && isJobId(param)) return param;
  return (await recentJobIds())[0] ?? null;
}
