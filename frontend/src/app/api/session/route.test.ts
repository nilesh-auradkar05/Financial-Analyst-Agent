// test-plan §20 "Guest access" (a Turnstile-verified visitor gets a guest session) and "Issuer allowlist"
// (the browser never receives an access token); ADR-0010 §2 and amendment 2026-10-11 §1.
import { test } from "node:test";
import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import {
  TURNSTILE_ALWAYS_FAIL, TURNSTILE_ALWAYS_PASS, fakeAlphaApi, fakeTurnstile, installFetch, newKeyPair, newSessionSecret,
  containsJwt, parseSetCookie, setEnv, verifyEs256, ISSUER, AUDIENCE,
} from "../../../test-support/fakes.ts";
import { withRequest } from "../../../test-support/next-harness.ts";

const { POST } = await import("./route.ts");
const { POST: analyze } = await import("../analyze/route.ts");
const { SESSION_COOKIE, openSession } = await import("../../../lib/session.server.ts");

const SECRET = newSessionSecret();
const SALT = "ip-salt-for-tests";
const keys = newKeyPair();
const CONFIG = {
  TURNSTILE_SECRET_KEY: TURNSTILE_ALWAYS_PASS,
  SESSION_SECRET: SECRET,
  GUEST_IP_SALT: SALT,
  GUEST_SIGNING_KEY: keys.privateKeyPem,
  GUEST_ISSUER: ISSUER,
  GUEST_AUDIENCE: AUDIENCE,
};

/** Cloudflare + API boundary fakes behind a single global fetch, dispatched on the request URL. */
function setup(env: Record<string, string | undefined> = {}) {
  const turnstile = fakeTurnstile();
  const api = fakeAlphaApi(keys.publicKey);
  const restoreFetch = installFetch((input, init) =>
    String(input instanceof Request ? input.url : input).startsWith("https://challenges.cloudflare.com/") ? turnstile.fetch(input, init) : api.fetch(input, init),
  );
  const restoreEnv = setEnv({ ...CONFIG, ...env });
  return { turnstile, api, restore: () => (restoreFetch(), restoreEnv()) };
}

const post = (body: unknown, { url = "https://alpha.test/api/session", headers = {} as Record<string, string> } = {}) =>
  new Request(url, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });

async function call(req: Request) {
  const { result: res, setCookies } = await withRequest(req, () => POST(req));
  const cookies = [...new Set([...res.headers.getSetCookie(), ...setCookies])].map(parseSetCookie).filter((c) => c.name === SESSION_COOKIE);
  const text = await res.text();
  return { res, cookies, text };
}

const rawIp = "203.0.113.7";

test("§20 Guest access: a verified visitor gets an HttpOnly guest session cookie and {ok:true}, no token or session contents", async () => {
  const { turnstile, restore } = setup();
  try {
    const { res, cookies, text } = await call(post({ turnstile_token: "visitor-ok" }, { headers: { "x-forwarded-for": `${rawIp}, 10.0.0.1` } }));
    assert.equal(res.status, 200);
    assert.deepEqual(JSON.parse(text), { ok: true });
    assert.ok(turnstile.requests.length >= 1, "the token was verified with Turnstile");

    assert.equal(cookies.length, 1, "exactly one alpha_session cookie");
    const [cookie] = cookies;
    assert.ok(cookie.attrs.includes("httponly"));
    assert.ok(cookie.attrs.includes("samesite=lax"));
    assert.ok(cookie.attrs.includes("path=/"));
    assert.ok(cookie.attrs.includes("max-age=86400"));
    assert.equal(cookie.attrs.includes("secure"), true, "https request -> Secure");

    const nowS = Math.floor(Date.now() / 1000);
    const session = openSession(decodeURIComponent(cookie.value), SECRET);
    assert.ok(session, "the cookie is a session sealed under SESSION_SECRET");
    assert.match(session.customer_id, /^guest:[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
    assert.ok(Math.abs(session.exp - (nowS + 86400)) <= 10);
    assert.equal(session.ip_hash, createHmac("sha256", SALT).update(rawIp).digest("hex"), "hash of the first x-forwarded-for hop");

    const everything = [text, ...[...res.headers].flat()].join("\n");
    assert.equal(containsJwt(everything), false, "no JWT in the response");
    assert.equal(everything.includes(session.customer_id), false, "no session contents in the response");
    assert.equal(everything.includes(session.ip_hash), false);
    assert.equal(everything.includes(rawIp), false);
    // the cookie value itself is opaque
    assert.equal(cookie.value.includes(session.customer_id), false);
  } finally {
    restore();
  }
});

test("§20 Guest access: the Secure attribute follows the request protocol", async () => {
  const { restore } = setup();
  try {
    const { cookies } = await call(post({ turnstile_token: "visitor-ok" }, { url: "http://localhost:3000/api/session" }));
    assert.equal(cookies.length, 1);
    assert.equal(cookies[0].attrs.includes("secure"), false);
  } finally {
    restore();
  }
});

test("§20 Guest access: client IP is the first x-forwarded-for hop, else cf-connecting-ip, else 'unknown' (assumed order)", async () => {
  const { restore } = setup();
  try {
    const cases: [string, Record<string, string>, string][] = [
      ["first hop of a chain", { "x-forwarded-for": "198.51.100.9, 10.0.0.1, 10.0.0.2", "cf-connecting-ip": "192.0.2.1" }, "198.51.100.9"],
      ["single x-forwarded-for", { "x-forwarded-for": "198.51.100.9" }, "198.51.100.9"],
      ["cf-connecting-ip when no x-forwarded-for", { "cf-connecting-ip": "192.0.2.1" }, "192.0.2.1"],
      ["neither header", {}, "unknown"],
    ];
    for (const [label, headers, ip] of cases) {
      const { res, cookies } = await call(post({ turnstile_token: "visitor-ok" }, { headers }));
      assert.equal(res.status, 200, label);
      const session = openSession(decodeURIComponent(cookies[0].value), SECRET);
      assert.equal(session?.ip_hash, createHmac("sha256", SALT).update(ip).digest("hex"), label);
    }
  } finally {
    restore();
  }
});

test("§20 Guest access: each visit gets its own guest id", async () => {
  const { restore } = setup();
  try {
    const ids = new Set<string>();
    for (let i = 0; i < 3; i++) {
      const { cookies } = await call(post({ turnstile_token: "visitor-ok" }));
      ids.add(openSession(decodeURIComponent(cookies[0].value), SECRET)?.customer_id ?? "none");
    }
    assert.equal(ids.size, 3);
    assert.equal(ids.has("none"), false);
  } finally {
    restore();
  }
});

test("§20 Guest access: an unverified Turnstile token is refused with 403 {error} and no cookie", async () => {
  const { restore } = setup({ TURNSTILE_SECRET_KEY: TURNSTILE_ALWAYS_FAIL });
  try {
    const { res, cookies, text } = await call(post({ turnstile_token: "visitor-bad" }));
    assert.equal(res.status, 403);
    assert.equal(typeof JSON.parse(text).error, "string");
    assert.equal(cookies.length, 0);
    assert.equal(containsJwt(text), false);
  } finally {
    restore();
  }
});

test("§20 Guest access: malformed bodies are refused with 400 {error} and no cookie", async () => {
  const { restore } = setup();
  try {
    const bodies: [string, unknown][] = [
      ["not JSON", "{nope"],
      ["empty body", ""],
      ["missing token", {}],
      ["token is a number", { turnstile_token: 7 }],
      ["token is null", { turnstile_token: null }],
      ["body is an array", ["visitor-ok"]],
    ];
    for (const [label, body] of bodies) {
      const { res, cookies, text } = await call(post(body));
      assert.equal(res.status, 400, label);
      assert.equal(typeof JSON.parse(text).error, "string", label);
      assert.equal(cookies.length, 0, label);
    }
    const empty = await call(post({ turnstile_token: "" }));
    assert.ok(empty.res.status === 400 || empty.res.status === 403, "empty token is refused");
    assert.equal(empty.cookies.length, 0);
  } finally {
    restore();
  }
});

test("§20 Guest access: missing configuration fails closed (no session is issued)", async () => {
  for (const missing of ["TURNSTILE_SECRET_KEY", "SESSION_SECRET", "GUEST_IP_SALT"]) {
    const { restore } = setup({ [missing]: undefined });
    try {
      const { res, cookies, text } = await call(post({ turnstile_token: "visitor-ok" }));
      assert.ok(res.status >= 400, `${missing} missing -> status ${res.status}`);
      assert.equal(cookies.length, 0, `${missing} missing -> no cookie`);
      assert.equal(containsJwt(text), false);
    } finally {
      restore();
    }
  }
});

test("§20 Guest access + Issuer allowlist: the issued cookie authorizes /api/analyze, and the API sees a guest token for that guest only", async () => {
  const { api, restore } = setup();
  try {
    const { cookies } = await call(post({ turnstile_token: "visitor-ok" }));
    const cookie = `${SESSION_COOKIE}=${cookies[0].value}`;
    const session = openSession(decodeURIComponent(cookies[0].value), SECRET);
    assert.ok(session);

    const req = new Request("https://alpha.test/api/analyze", {
      method: "POST",
      headers: { "content-type": "application/json", cookie },
      body: JSON.stringify({ ticker: "acme" }),
    });
    const { result: res } = await withRequest(req, () => analyze(req));
    assert.equal(res.status, 202);
    const everything = [await res.text(), ...[...res.headers].flat()].join("\n");
    assert.equal(containsJwt(everything), false, "the browser never receives an access token");

    assert.equal(api.requests.length, 1);
    const token = (api.requests[0].headers.get("authorization") ?? "").replace(/^Bearer /, "");
    assert.equal(verifyEs256(token, keys.publicKey)?.claims.sub, session.customer_id);
  } finally {
    restore();
  }
});
