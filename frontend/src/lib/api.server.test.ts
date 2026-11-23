// ADR-0010 §2 + amendment 2026-10-11 §1: the BFF authenticates to the API with a guest access token minted from the
// server-side session, never with a shared API key; the browser never sees a token.
// test-plan §20 "Guest access" (a guest token cannot read another guest's job) and "Issuer allowlist".
import { test } from "node:test";
import assert from "node:assert/strict";
import { AUDIENCE, ISSUER, fakeAlphaApi, installFetch, newKeyPair, newSessionSecret, setEnv, verifyEs256 } from "../test-support/fakes.ts";
import { withRequest } from "../test-support/next-harness.ts";

const { getJob, listRuns, startAnalysis, UpstreamError } = await import("./api.server.ts");
const { SESSION_COOKIE, newGuestSession, sealSession } = await import("./session.server.ts");

const SECRET = newSessionSecret();
const keys = newKeyPair();
const CONFIG = {
  SESSION_SECRET: SECRET,
  GUEST_SIGNING_KEY: keys.privateKeyPem,
  GUEST_ISSUER: ISSUER,
  GUEST_AUDIENCE: AUDIENCE,
  API_KEY: "shared-api-key-must-never-be-sent",
};

const now = () => Math.floor(Date.now() / 1000);
// Flip a character well inside the final bytes (the very last base64 char may carry only unused bits).
const flip = (v: string) => v.slice(0, v.length - 5) + (v[v.length - 5] === "A" ? "B" : "A") + v.slice(v.length - 4);
const guest = (ip = "203.0.113.7") => newGuestSession(ip, "salt");
const cookieFor = (value?: string): Record<string, string> => (value === undefined ? {} : { cookie: `${SESSION_COOKIE}=${encodeURIComponent(value)}` });
const inRequest = <T>(cookieValue: string | undefined, fn: () => Promise<T>) =>
  withRequest(new Request("https://alpha.test/api/x", { headers: cookieFor(cookieValue) }), fn).then((r) => r.result);

function setup(env: Record<string, string | undefined> = CONFIG) {
  const api = fakeAlphaApi(keys.publicKey);
  const restoreFetch = installFetch(api.fetch);
  const restoreEnv = setEnv({ ...CONFIG, ...env });
  return { api, restore: () => (restoreFetch(), restoreEnv()) };
}

test("ADR-0010 amend. §1: a valid guest session sends a minted ES256 guest token, and never API_KEY", async () => {
  const { api, restore } = setup();
  try {
    const session = guest();
    const sealed = sealSession(session, SECRET);
    const accepted = await inRequest(sealed, () => startAnalysis("ACME"));
    const polled = await inRequest(sealed, () => getJob(accepted.job_id));
    const runs = await inRequest(sealed, () => listRuns());
    assert.equal(polled.job_id, accepted.job_id);
    assert.deepEqual(runs.map((r) => r.job_id), [accepted.job_id]);

    assert.equal(api.requests.length, 3);
    for (const req of api.requests) {
      const auth = req.headers.get("authorization") ?? "";
      assert.match(auth, /^Bearer [\w-]+\.[\w-]+\.[\w-]+$/);
      assert.equal(auth.includes(CONFIG.API_KEY), false, "API_KEY is never forwarded");
      const jwt = verifyEs256(auth.slice("Bearer ".length), keys.publicKey);
      assert.ok(jwt, "token verifies against the guest public key");
      assert.equal(jwt.claims.iss, ISSUER);
      assert.equal(jwt.claims.aud, AUDIENCE);
      assert.equal(jwt.claims.sub, session.customer_id);
      assert.equal(jwt.claims.role, "guest");
      assert.equal(jwt.claims.ip_hash, session.ip_hash);
      const iat = jwt.claims.iat as number;
      assert.ok(Math.abs(iat - now()) <= 5);
      assert.equal((jwt.claims.exp as number) - iat, 900);
    }
    assert.equal(api.requests[0].headers.get("content-type"), "application/json");
    assert.equal(JSON.stringify([accepted, polled, runs]).includes("shared-api-key"), false);
  } finally {
    restore();
  }
});

test("ADR-0010 amend. §1: with no valid session or missing signing config the BFF sends no Authorization header, and never API_KEY", async () => {
  const valid = sealSession(guest(), SECRET);
  const expired = sealSession({ ...guest(), exp: now() - 60 }, SECRET);
  const foreign = sealSession(guest(), newSessionSecret());
  const tampered = flip(valid);
  const cases: [string, string | undefined, Record<string, string | undefined>][] = [
    ["no cookie", undefined, {}],
    ["garbage cookie", "garbage", {}],
    ["tampered cookie", tampered, {}],
    ["expired session", expired, {}],
    ["session sealed under another secret", foreign, {}],
    ["valid session, no signing key", valid, { GUEST_SIGNING_KEY: undefined }],
    ["valid session, no issuer", valid, { GUEST_ISSUER: undefined }],
    ["valid session, no audience", valid, { GUEST_AUDIENCE: undefined }],
    ["valid session, no session secret", valid, { SESSION_SECRET: undefined }],
  ];
  for (const [label, cookie, env] of cases) {
    const { api, restore } = setup(env);
    try {
      await inRequest(cookie, () => getJob("11111111-1111-4111-8111-111111111111")).catch(() => undefined);
      await inRequest(cookie, () => listRuns()).catch(() => undefined);
      await inRequest(cookie, () => startAnalysis("ACME")).catch(() => undefined);
      for (const req of api.requests) {
        assert.equal(req.headers.has("authorization"), false, `${label}: no Authorization header`);
      }
    } finally {
      restore();
    }
  }
});

test("test-plan §20 Guest access: another guest's job is not readable through the BFF (upstream 404 surfaces as UpstreamError 404)", async () => {
  const { restore } = setup();
  try {
    const alice = sealSession(guest("203.0.113.1"), SECRET);
    const bob = sealSession(guest("203.0.113.2"), SECRET);
    const job = await inRequest(alice, () => startAnalysis("ACME"));
    assert.equal((await inRequest(alice, () => getJob(job.job_id))).job_id, job.job_id);
    await assert.rejects(
      inRequest(bob, () => getJob(job.job_id)),
      (e: unknown) => e instanceof UpstreamError && e.status === 404,
    );
    assert.deepEqual(await inRequest(bob, () => listRuns()), []);
  } finally {
    restore();
  }
});

test("ADR-0010 §2 + test-plan §20 Issuer allowlist: without a session the API refuses and the BFF reports upstream unavailable, not a token", async () => {
  const { restore } = setup();
  try {
    await assert.rejects(
      inRequest(undefined, () => listRuns()),
      (e: unknown) => e instanceof UpstreamError && e.status === 502,
    );
  } finally {
    restore();
  }
});
