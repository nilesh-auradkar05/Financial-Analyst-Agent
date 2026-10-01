import "server-only";
import type { JobAcceptedResponse, JobPollResponse } from "./api-types";

const BASE = (process.env.ALPHA_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");
const TIMEOUT_MS = 15_000;

/** Failure carrying only a safe, client-presentable status; upstream bodies are never kept. */
export class UpstreamError extends Error {
  readonly status: 404 | 502 | 504;

  constructor(status: 404 | 502 | 504) {
    super(status === 404 ? "not found" : "upstream unavailable");
    this.status = status;
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    const headers = new Headers(init?.headers);
    const apiKey = process.env.API_KEY?.trim();
    if (apiKey) headers.set("Authorization", `Bearer ${apiKey}`);
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers,
      cache: "no-store",
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch (e) {
    throw new UpstreamError(e instanceof Error && e.name === "TimeoutError" ? 504 : 502);
  }
  if (res.status === 404) throw new UpstreamError(404);
  if (!res.ok) throw new UpstreamError(502);
  try {
    return (await res.json()) as T;
  } catch {
    throw new UpstreamError(502);
  }
}

export function startAnalysis(ticker: string): Promise<JobAcceptedResponse> {
  return call<JobAcceptedResponse>("/analyze/async", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker }),
  });
}

export function getJob(jobId: string): Promise<JobPollResponse> {
  return call<JobPollResponse>(`/jobs/${encodeURIComponent(jobId)}`);
}
