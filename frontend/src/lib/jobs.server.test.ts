// test-plan §19 "Own runs that still exist" and "Nothing of mine survives" (guest branch: a guest whose runs are all
// gone is shown sample data, because a guest's GET /jobs returns only that guest's runs); ADR-0010 amendment 2026-10-11.
import { test } from "node:test";
import assert from "node:assert/strict";
import { AUDIENCE, ISSUER, fakeAlphaApi, installFetch, newKeyPair, newSessionSecret, setEnv } from "../test-support/fakes.ts";
import { withRequest } from "../test-support/next-harness.ts";
import type { GuestSession } from "./session.server.ts";

const { availableRuns } = await import("./jobs.server.ts");
const { SESSION_COOKIE, newGuestSession, sealSession } = await import("./session.server.ts");

const SECRET = newSessionSecret();
const keys = newKeyPair();
const CONFIG = { SESSION_SECRET: SECRET, GUEST_SIGNING_KEY: keys.privateKeyPem, GUEST_ISSUER: ISSUER, GUEST_AUDIENCE: AUDIENCE };
const LOST = ["aaaaaaaa-1111-4111-8111-111111111111", "bbbbbbbb-2222-4222-8222-222222222222"];

function setup(fetchOverride?: typeof fetch) {
  const api = fakeAlphaApi(keys.publicKey);
  const restoreFetch = installFetch(fetchOverride ?? api.fetch);
  const restoreEnv = setEnv(CONFIG);
  return { api, restore: () => (restoreFetch(), restoreEnv()) };
}

const alice = newGuestSession("203.0.113.1", "salt");
const bob = newGuestSession("203.0.113.2", "salt");
const cookieFor = (session: GuestSession, extra = "") =>
  [`${SESSION_COOKIE}=${encodeURIComponent(sealSession(session, SECRET))}`, extra].filter(Boolean).join("; ");
const runsFor = (cookie: string) => withRequest(new Request("https://alpha.test/runs", { headers: cookie ? { cookie } : {} }), () => availableRuns()).then((r) => r.result);

test("§19 Own runs that still exist: a guest sees exactly their own runs, newest first, flagged own — not another guest's", async () => {
  const { api, restore } = setup();
  try {
    const mine = alice.customer_id;
    const a1 = api.seed(mine, "AAA");
    api.seed(bob.customer_id, "BBB");
    const a2 = api.seed(mine, "CCC");
    api.seed(bob.customer_id, "DDD");
    const a3 = api.seed(mine, "EEE");

    // `alpha_jobs` listing runs that no longer exist must not change the answer (the old remembered-ids cookie is moot).
    const { runs, own } = await runsFor(cookieFor(alice, `alpha_jobs=${LOST.join(",")}`));
    assert.deepEqual(runs.map((r) => r.job_id), [a3, a2, a1], "own runs only, in the API's newest-first order");
    assert.deepEqual(runs.map((r) => r.ticker), ["EEE", "CCC", "AAA"]);
    assert.equal(own, true);
  } finally {
    restore();
  }
});

test("§19 Nothing of mine survives (guest): no runs, an unreachable API or no session yields no runs, so the page shows sample data", async () => {
  const { api, restore } = setup();
  try {
    api.seed(bob.customer_id, "BBB");
    assert.deepEqual((await runsFor(cookieFor(alice, `alpha_jobs=${LOST.join(",")}`))).runs, [], "guest with nothing left");
    assert.deepEqual((await runsFor("")).runs, [], "no session");
    assert.deepEqual((await runsFor(`alpha_jobs=${LOST.join(",")}`)).runs, [], "remembered ids alone never surface server runs");
  } finally {
    restore();
  }

  const down = setup(async () => {
    throw new TypeError("fetch failed");
  });
  try {
    assert.deepEqual((await runsFor(cookieFor(alice))).runs, [], "API unreachable");
  } finally {
    down.restore();
  }

  const broken = setup(async () => new Response("boom", { status: 500 }));
  try {
    assert.deepEqual((await runsFor(cookieFor(alice))).runs, [], "API error");
  } finally {
    broken.restore();
  }
});
