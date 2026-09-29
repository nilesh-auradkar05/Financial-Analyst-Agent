// Run: node --test --experimental-strip-types "src/app/memos/[id]/_map.test.ts"
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseInline, parseMemo, safeHttpUrl } from "./_map.ts";

const known = new Set(["1", "2"]);

test("marker becomes a cite node and surrounding text is preserved", () => {
  assert.deepEqual(parseInline("Rev rose [1]. Margin fell [2][1].", known), [
    { t: "text", text: "Rev rose " },
    { t: "cite", id: "1" },
    { t: "text", text: ". Margin fell " },
    { t: "cite", id: "2" },
    { t: "cite", id: "1" },
    { t: "text", text: "." },
  ]);
});

test("HTML-looking input stays literal text", () => {
  const s = '<img src=x onerror=alert(1)> <b>hi</b> [1]';
  const out = parseInline(s, known);
  assert.equal(out[0].t, "text");
  assert.equal((out[0] as { text: string }).text, "<img src=x onerror=alert(1)> <b>hi</b> ");
  assert.deepEqual(out[1], { t: "cite", id: "1" });
});

test("unknown citation index stays as literal text", () => {
  assert.deepEqual(parseInline("Claim [9] and [1, 7].", known), [
    { t: "text", text: "Claim [9] and " },
    { t: "cite", id: "1" },
    { t: "text", text: "[7]." },
  ]);
});

test("memo sections and bullets", () => {
  const s = parseMemo("## Executive Summary\nGrowth [1].\n\n**Risks**\n- One [2].\n- Two.");
  assert.equal(s[0].heading, "Executive Summary");
  assert.deepEqual(s[1].blocks, [{ kind: "ul", items: ["One [2].", "Two."] }]);
});

test("safeHttpUrl rejects non-http schemes", () => {
  assert.equal(safeHttpUrl("javascript:alert(1)"), null);
  assert.equal(safeHttpUrl("https://sec.gov/x"), "https://sec.gov/x");
});
