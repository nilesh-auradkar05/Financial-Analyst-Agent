import { NextResponse } from "next/server";
import { getJob, UpstreamError } from "@/lib/api.server";
import { isJobId } from "@/lib/validate";

export async function GET(_req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  if (!isJobId(id)) return NextResponse.json({ error: "invalid job id" }, { status: 400 });
  try {
    return NextResponse.json(await getJob(id));
  } catch (e) {
    if (e instanceof UpstreamError && e.status === 404) {
      return NextResponse.json({ error: "job not found" }, { status: 404 });
    }
    return NextResponse.json({ error: "upstream unavailable" }, { status: 502 });
  }
}
