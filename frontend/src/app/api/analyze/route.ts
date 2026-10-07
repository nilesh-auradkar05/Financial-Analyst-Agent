import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { startAnalysis, UpstreamError } from "@/lib/api.server";
import { JOBS_COOKIE } from "@/lib/jobs.server";
import { isTicker, normalizeTicker, withJob } from "@/lib/validate";

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }
  const ticker = (body as { ticker?: unknown } | null)?.ticker;
  if (!isTicker(ticker)) return NextResponse.json({ error: "invalid ticker" }, { status: 400 });
  try {
    const job = await startAnalysis(normalizeTicker(ticker));
    const res = NextResponse.json(job, { status: 202 });
    res.cookies.set(JOBS_COOKIE, withJob((await cookies()).get(JOBS_COOKIE)?.value, job.job_id), {
      httpOnly: true,
      sameSite: "lax",
      secure: new URL(req.url).protocol === "https:",
      path: "/",
      maxAge: 60 * 60 * 24 * 30,
    });
    return res;
  } catch (e) {
    const status = e instanceof UpstreamError ? e.status : 502;
    return NextResponse.json({ error: "upstream unavailable" }, { status: status === 404 ? 502 : status });
  }
}
