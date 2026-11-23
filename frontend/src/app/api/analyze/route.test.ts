// test-plan §20 "Guest access" (only a session-holding guest can start analyses; guest tokens are minted server-side)
// and "Issuer allowlist" (the browser never receives an access token); ADR-0010 §2.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  AUDIENCE, ISSUER, containsJwt, fakeAlphaApi, installFetch, newKeyPair, newSessionSecret, setEnv, verifyEs256,
} from "../../../test-support/fakes.ts";
import { withRequest } from "../../../test-support/next-harness.ts";

const { POST } = await import("./route.ts");
const { SESSION_COOKIE, newGuestSession, sealSession } = await import("../../../lib/session.server.ts");

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

function setup() {
  const api = fakeAlphaApi(keys.publicKey);
  const restoreFetch = installFetch(api.fetch);
  const restoreEnv = setEnv(CONFIG);
  return { api, restore: () => (restoreFetch(), restoreEnv()) };
}

async function analyze(ticker: unknown, cookieValue?: string) {
  const req = new Request("https://alpha.test/api/analyze", {
    method: "POST",
    headers: { "content-type": "application/json", ...(cookieValue === undefined ? {} : { cookie: `${SESSION_COOKIE}=${encodeURIComponent(cookieValue)}` }) },
    body: JSON.stringify({ ticker }),
  });
  const { result: res } = await withRequest(req, () => POST(req));
  const text = await res.text();
  return { res, text, headers: [...res.headers].flat().join("\n") };
}

test("§20 Guest access: /api/analyze without a valid session is 401 {error: 'session required'} and makes no upstream call", async () => {
  const { api, restore } = setup();
  try {
    const valid = sealSession(newGuestSession("203.0.113.7", "salt"), SECRET);
    const cases: [string, string | undefined][] = [
      ["no cookie", undefined],
      ["garbage cookie", "garbage"],
      ["tampered cookie", flip(valid)],
      ["expired session", sealSession({ ...newGuestSession("203.0.113.7", "salt"), exp: now() - 60 }, SECRET)],
      ["session sealed under another secret", sealSession(newGuestSession("203.0.113.7", "salt"), newSessionSecret())],
    ];
    for (const [label, cookie] of cases) {
      const { res, text } = await analyze("ACME", cookie);
      assert.equal(res.status, 401, label);
      assert.deepEqual(JSON.parse(text), { error: "session required" }, label);
    }
    assert.equal(api.requests.length, 0, "no upstream call for any of them");
  } finally {
    restore();
  }
});

test("§20 Guest access: with a valid session /api/analyze starts the job as that guest and never exposes a token or API_KEY", async () => {
  const { api, restore } = setup();
  try {
    const session = newGuestSession("203.0.113.7", "salt");
    const { res, text, headers } = await analyze("acme", sealSession(session, SECRET));
    assert.equal(res.status, 202);
    const job = JSON.parse(text);
    assert.equal(job.ticker, "ACME", "ticker is normalized before it reaches the API");
    assert.equal(typeof job.job_id, "string");

    assert.equal(api.requests.length, 1);
    const auth = api.requests[0].headers.get("authorization") ?? "";
    assert.equal(auth.includes(CONFIG.API_KEY), false);
    const jwt = verifyEs256(auth.replace(/^Bearer /, ""), keys.publicKey);
    assert.ok(jwt, "upstream request carries a verifying guest token");
    assert.equal(jwt.claims.sub, session.customer_id);
    assert.equal(jwt.claims.role, "guest");

    assert.equal(containsJwt(text + "\n" + headers), false, "the browser never receives an access token");
    assert.equal((text + headers).includes(CONFIG.API_KEY), false);
  } finally {
    restore();
  }
});

test("§20 Guest access: with a valid session an invalid ticker is still 400 and makes no upstream call", async () => {
  const { api, restore } = setup();
  try {
    const { res, text } = await analyze("not a ticker!", sealSession(newGuestSession("203.0.113.7", "salt"), SECRET));
    assert.equal(res.status, 400);
    assert.equal(typeof JSON.parse(text).error, "string");
    assert.equal(api.requests.length, 0);
  } finally {
    restore();
  }
});
