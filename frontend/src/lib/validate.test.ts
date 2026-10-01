// Run: node --test --experimental-strip-types src/lib/validate.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { isJobId, isTicker } from "./validate.ts";

test("isTicker", () => {
  for (const ok of ["AAPL", " aapl ", "BRK.B", "BF-B", "A"]) assert.ok(isTicker(ok), ok);
  for (const bad of ["", "1ABC", "AAPL$", "ABCDEFGHIJK", "A B", 5, null]) assert.ok(!isTicker(bad), String(bad));
});

test("isJobId", () => {
  assert.ok(isJobId("123e4567-e89b-12d3-a456-426614174000"));
  for (const bad of ["abc123", "../etc", "", 1]) assert.ok(!isJobId(bad), String(bad));
});
