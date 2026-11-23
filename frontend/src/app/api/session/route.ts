import { NextResponse } from "next/server";
import { newGuestSession, SESSION_COOKIE, SESSION_TTL_SECONDS, sealSession } from "@/lib/session.server";
import { verifyTurnstile } from "@/lib/turnstile.server";

const clientIp = (req: Request): string =>
  req.headers.get("x-forwarded-for")?.split(",")[0].trim() || req.headers.get("cf-connecting-ip") || "unknown";

export async function POST(req: Request) {
  const { TURNSTILE_SECRET_KEY: turnstileSecret, SESSION_SECRET: sessionSecret, GUEST_IP_SALT: ipSalt } = process.env;
  if (!turnstileSecret || !sessionSecret || !ipSalt) {
    return NextResponse.json({ error: "session service not configured" }, { status: 503 });
  }
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }
  const token = (body as { turnstile_token?: unknown } | null)?.turnstile_token;
  if (typeof token !== "string" || !token) return NextResponse.json({ error: "turnstile_token required" }, { status: 400 });

  const ip = clientIp(req);
  if (!(await verifyTurnstile(token, ip, { secret: turnstileSecret }))) {
    return NextResponse.json({ error: "verification failed" }, { status: 403 });
  }
  const res = NextResponse.json({ ok: true });
  res.cookies.set(SESSION_COOKIE, sealSession(newGuestSession(ip, ipSalt), sessionSecret), {
    httpOnly: true,
    sameSite: "lax",
    secure: new URL(req.url).protocol === "https:",
    path: "/",
    maxAge: SESSION_TTL_SECONDS,
  });
  return res;
}
