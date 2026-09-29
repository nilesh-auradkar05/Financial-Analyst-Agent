import { NextResponse } from "next/server";
import { startAnalysis, UpstreamError } from "@/lib/api.server";
import { isTicker, normalizeTicker } from "@/lib/validate";

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
    return NextResponse.json(await startAnalysis(normalizeTicker(ticker)), { status: 202 });
  } catch (e) {
    const status = e instanceof UpstreamError ? e.status : 502;
    return NextResponse.json({ error: "upstream unavailable" }, { status: status === 404 ? 502 : status });
  }
}
