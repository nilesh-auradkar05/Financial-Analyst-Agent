// test-plan §1: protected analysis routes receive server credentials without exposing them to the browser.
import { test } from "node:test";
import assert from "node:assert/strict";
import { getJob, startAnalysis } from "./api.server.ts";

test("server API client forwards API_KEY only upstream", async () => {
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.API_KEY;
  const requests: { url: string; headers: Headers }[] = [];
  globalThis.fetch = async (input, init) => {
    requests.push({ url: String(input), headers: new Headers(init?.headers) });
    return Response.json({ job_id: "job-1", ticker: "ACME", status: "pending", started_at: "now" });
  };
  try {
    process.env.API_KEY = "test-key";
    const accepted = await startAnalysis("ACME");
    const polled = await getJob("job-1");
    assert.equal(requests.length, 2);
    assert.equal(requests[0].headers.get("Authorization"), "Bearer test-key");
    assert.equal(requests[0].headers.get("Content-Type"), "application/json");
    assert.equal(requests[1].headers.get("Authorization"), "Bearer test-key");
    assert.equal(JSON.stringify([accepted, polled]).includes("test-key"), false);

    delete process.env.API_KEY;
    await getJob("job-1");
    assert.equal(requests[2].headers.has("Authorization"), false);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.API_KEY;
    else process.env.API_KEY = originalKey;
  }
});
