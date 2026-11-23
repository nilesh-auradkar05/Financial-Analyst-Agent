// Test-only fakes for system boundaries: the Turnstile verification endpoint, the Alpha API and the environment.
import { createVerify, generateKeyPairSync, randomBytes, randomUUID, type KeyObject } from "node:crypto";

export const TURNSTILE_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify";
/** Cloudflare's documented dummy secrets: the first always passes, the second always fails. */
export const TURNSTILE_ALWAYS_PASS = "1x0000000000000000000000000000000AA";
export const TURNSTILE_ALWAYS_FAIL = "2x0000000000000000000000000000000AA";

export const ISSUER = "https://alpha.test/guest";
export const AUDIENCE = "alpha-api";

/** Throwaway P-256 key pair; the private half is PKCS#8 PEM, the form a Parameter Store SecureString holds. */
export function newKeyPair(): { privateKeyPem: string; publicKey: KeyObject } {
  const { privateKey, publicKey } = generateKeyPairSync("ec", { namedCurve: "P-256" });
  return { privateKeyPem: privateKey.export({ type: "pkcs8", format: "pem" }).toString(), publicKey };
}

export const newSessionSecret = (): string => randomBytes(32).toString("base64");

const b64 = (s: string): Buffer => Buffer.from(s, "base64url");

export type VerifiedJwt = { header: Record<string, unknown>; claims: Record<string, unknown> };

/** Verifies a compact ES256 JWT signature (IEEE P1363 r||s). Returns null when the signature does not verify. */
export function verifyEs256(token: string, publicKey: KeyObject): VerifiedJwt | null {
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  const signature = b64(parts[2]);
  if (signature.length !== 64) return null;
  const ok = createVerify("SHA256").update(`${parts[0]}.${parts[1]}`).verify({ key: publicKey, dsaEncoding: "ieee-p1363" }, signature);
  if (!ok) return null;
  return { header: JSON.parse(b64(parts[0]).toString()), claims: JSON.parse(b64(parts[1]).toString()) };
}

export type RecordedRequest = { url: string; method: string; headers: Headers; body: string; signal: AbortSignal | null | undefined };

async function record(input: string | URL | Request, init?: RequestInit): Promise<RecordedRequest> {
  const req = new Request(input, init);
  return { url: req.url, method: req.method, headers: req.headers, body: await req.text(), signal: init?.signal };
}

export type NetworkFault = "network" | 500;

/** Faithful stand-in for Cloudflare siteverify: success only for the always-pass secret with a non-empty response token. */
export function fakeTurnstile(faults: NetworkFault[] = []): { fetch: typeof fetch; requests: RecordedRequest[] } {
  const requests: RecordedRequest[] = [];
  const queue = [...faults];
  const fakeFetch: typeof fetch = async (input, init) => {
    const req = await record(input, init);
    requests.push(req);
    const fault = queue.shift();
    if (fault === "network") throw new TypeError("fetch failed");
    if (fault === 500) return new Response("internal error", { status: 500 });
    if (req.url !== TURNSTILE_URL || req.method !== "POST") return new Response("not found", { status: 404 });
    let body: { secret?: unknown; response?: unknown };
    try {
      body = JSON.parse(req.body);
    } catch {
      return Response.json({ success: false, "error-codes": ["bad-request"] });
    }
    const passes = body.secret === TURNSTILE_ALWAYS_PASS && typeof body.response === "string" && body.response.length > 0;
    return Response.json(passes ? { success: true } : { success: false, "error-codes": ["invalid-input-response"] });
  };
  return { fetch: fakeFetch, requests };
}

type FakeJob = { job_id: string; owner: string; ticker: string; status: "pending"; started_at: string };

/**
 * Stand-in for the Alpha API as slice (a) defines it: bearer ES256 JWT from the allow-listed issuer, jobs scoped to the
 * token's `sub`, `GET /jobs` returning only the caller's jobs newest first. Anything else is 401 / 404.
 */
export function fakeAlphaApi(publicKey: KeyObject, opts: { issuer?: string; audience?: string } = {}) {
  const issuer = opts.issuer ?? ISSUER;
  const audience = opts.audience ?? AUDIENCE;
  const jobs: FakeJob[] = [];
  const requests: RecordedRequest[] = [];

  const seed = (owner: string, ticker = "ACME"): string => {
    const job_id = randomUUID();
    jobs.push({ job_id, owner, ticker, status: "pending", started_at: new Date(Date.UTC(2026, 10, 1) + jobs.length * 60_000).toISOString() });
    return job_id;
  };
  const view = (j: FakeJob) => ({ job_id: j.job_id, ticker: j.ticker, status: j.status, started_at: j.started_at });

  function subject(headers: Headers): string | null {
    const match = /^Bearer (.+)$/.exec(headers.get("authorization") ?? "");
    const jwt = match ? verifyEs256(match[1], publicKey) : null;
    if (!jwt) return null;
    const { iss, aud, exp, sub, role } = jwt.claims;
    const now = Date.now() / 1000;
    if (iss !== issuer || aud !== audience || typeof exp !== "number" || exp <= now) return null;
    return typeof sub === "string" && role === "guest" ? sub : null;
  }

  const fakeFetch: typeof fetch = async (input, init) => {
    const req = await record(input, init);
    requests.push(req);
    const url = new URL(req.url);
    const sub = subject(req.headers);
    if (!sub) return Response.json({ detail: "unauthorized" }, { status: 401 });
    if (req.method === "POST" && url.pathname === "/analyze/async") {
      const id = seed(sub, String((JSON.parse(req.body) as { ticker?: unknown }).ticker));
      return Response.json(view(jobs.find((j) => j.job_id === id)!), { status: 202 });
    }
    if (req.method === "GET" && url.pathname === "/jobs") {
      return Response.json(jobs.filter((j) => j.owner === sub).map(view).reverse());
    }
    const one = /^\/jobs\/([^/]+)$/.exec(url.pathname);
    if (req.method === "GET" && one) {
      const job = jobs.find((j) => j.job_id === one[1] && j.owner === sub);
      return job ? Response.json(view(job)) : Response.json({ detail: "not found" }, { status: 404 });
    }
    return Response.json({ detail: "not found" }, { status: 404 });
  };
  return { fetch: fakeFetch, requests, seed };
}

/** Replaces global fetch (the BFF's only egress) with a boundary fake; returns the restore function. */
export function installFetch(fake: typeof fetch): () => void {
  const original = globalThis.fetch;
  globalThis.fetch = fake;
  return () => {
    globalThis.fetch = original;
  };
}

/** Sets (string) or removes (undefined) env vars; returns the restore function. */
export function setEnv(vars: Record<string, string | undefined>): () => void {
  const saved = Object.fromEntries(Object.keys(vars).map((k) => [k, process.env[k]]));
  for (const [k, v] of Object.entries(vars)) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
  return () => {
    for (const [k, v] of Object.entries(saved)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
  };
}

/** Name=value of the first Set-Cookie header whose name matches, plus its lower-cased attributes. */
export function parseSetCookie(headerValue: string): { name: string; value: string; attrs: string[] } {
  const [pair, ...attrs] = headerValue.split(";").map((s) => s.trim());
  const eq = pair.indexOf("=");
  return { name: pair.slice(0, eq), value: pair.slice(eq + 1), attrs: attrs.map((a) => a.toLowerCase()) };
}

/** True when `text` contains anything shaped like a signed JWT (three dot-separated base64url parts, JSON header with `alg`). */
export function containsJwt(text: string): boolean {
  for (const m of text.matchAll(/[\w-]{10,}\.[\w-]{10,}\.[\w-]{10,}/g)) {
    try {
      const header = JSON.parse(Buffer.from(m[0].split(".")[0], "base64url").toString()) as { alg?: unknown };
      if (typeof header.alg === "string") return true;
    } catch {
      // not a JWT header
    }
  }
  return false;
}
