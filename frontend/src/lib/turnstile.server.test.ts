// ADR-0010 amendment 2026-10-11 §1 (guest access only after Cloudflare Turnstile); test-plan §20 "Guest access"
// (a Turnstile-verified visitor gets a guest session). Fails closed; outbound call has a timeout and a bounded retry (CLAUDE.md §5).
import { test } from "node:test";
import assert from "node:assert/strict";
import { TURNSTILE_ALWAYS_FAIL, TURNSTILE_ALWAYS_PASS, TURNSTILE_URL, fakeTurnstile, type NetworkFault } from "../test-support/fakes.ts";
import "../test-support/next-harness.ts";

const { verifyTurnstile } = await import("./turnstile.server.ts");

test("§20 Guest access: Turnstile passes only for a valid secret and a non-empty token, via a JSON POST to siteverify", async () => {
  const net = fakeTurnstile();
  assert.equal(await verifyTurnstile("visitor-token", "203.0.113.7", { secret: TURNSTILE_ALWAYS_PASS, fetch: net.fetch }), true);
  assert.equal(net.requests.length, 1);
  const req = net.requests[0];
  assert.equal(req.url, TURNSTILE_URL);
  assert.equal(req.method, "POST");
  assert.match(req.headers.get("content-type") ?? "", /application\/json/);
  assert.deepEqual(JSON.parse(req.body), { secret: TURNSTILE_ALWAYS_PASS, response: "visitor-token", remoteip: "203.0.113.7" });

  assert.equal(await verifyTurnstile("visitor-token", "203.0.113.7", { secret: TURNSTILE_ALWAYS_FAIL, fetch: fakeTurnstile().fetch }), false);
  assert.equal(await verifyTurnstile("visitor-token", undefined, { secret: TURNSTILE_ALWAYS_PASS, fetch: fakeTurnstile().fetch }), true, "remoteip is optional");
});

test("§20 Guest access: Turnstile fails closed on missing secret, empty token, or an unusable reply", async () => {
  const closed: [string, () => Promise<boolean>][] = [
    ["secret undefined", () => verifyTurnstile("t", "1.2.3.4", { secret: undefined, fetch: fakeTurnstile().fetch })],
    ["secret empty", () => verifyTurnstile("t", "1.2.3.4", { secret: "", fetch: fakeTurnstile().fetch })],
    ["token empty", () => verifyTurnstile("", "1.2.3.4", { secret: TURNSTILE_ALWAYS_PASS, fetch: fakeTurnstile().fetch })],
    ["reply is HTML", () => verifyTurnstile("t", "1.2.3.4", { secret: TURNSTILE_ALWAYS_PASS, fetch: async () => new Response("<html>ok</html>") })],
    ["reply is empty", () => verifyTurnstile("t", "1.2.3.4", { secret: TURNSTILE_ALWAYS_PASS, fetch: async () => new Response("") })],
    ["success missing", () => verifyTurnstile("t", "1.2.3.4", { secret: TURNSTILE_ALWAYS_PASS, fetch: async () => Response.json({}) })],
    ["success is the string 'true'", () => verifyTurnstile("t", "1.2.3.4", { secret: TURNSTILE_ALWAYS_PASS, fetch: async () => Response.json({ success: "true" }) })],
    ["success is false", () => verifyTurnstile("t", "1.2.3.4", { secret: TURNSTILE_ALWAYS_PASS, fetch: async () => Response.json({ success: false }) })],
  ];
  for (const [label, run] of closed) assert.equal(await run(), false, label);
});

test("§20 Guest access: a network failure or 5xx is retried at most once, then Turnstile fails closed", async () => {
  const table: [string, NetworkFault[], boolean, number][] = [
    ["network error then success", ["network"], true, 2],
    ["5xx then success", [500], true, 2],
    ["two network errors", ["network", "network"], false, 2],
    ["two 5xx", [500, 500], false, 2],
    ["network error, 5xx, then would-be success", ["network", 500], false, 2],
  ];
  for (const [label, faults, expected, attempts] of table) {
    const net = fakeTurnstile(faults);
    assert.equal(await verifyTurnstile("t", "1.2.3.4", { secret: TURNSTILE_ALWAYS_PASS, fetch: net.fetch }), expected, label);
    assert.equal(net.requests.length, attempts, `${label}: attempts`);
  }
});

test("§20 Guest access: every Turnstile attempt carries an abort signal (timeout)", async () => {
  const net = fakeTurnstile(["network"]);
  await verifyTurnstile("t", "1.2.3.4", { secret: TURNSTILE_ALWAYS_PASS, fetch: net.fetch });
  assert.equal(net.requests.length, 2);
  for (const req of net.requests) {
    assert.ok(req.signal instanceof AbortSignal, "fetch init.signal");
    assert.equal(req.signal.aborted, false);
  }
});
