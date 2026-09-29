// Usage: node scripts/parity.mjs <home|workspace|memo|runs|all> [--base http://localhost:3100]
import { chromium } from "@playwright/test";
import { PNG } from "pngjs";
import pixelmatch from "pixelmatch";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const refDir = path.resolve(root, "../frontend-ref");
const outDir = path.join(root, ".parity");

const SCREENS = {
  home: { route: "/", ref: "Product website · home@2x.png", w: 1440, h: 900, full: true },
  workspace: { route: "/workspace", ref: "Workspace · live multi-agent run@2x.png", w: 1440, h: 1024 },
  memo: { route: "/memos/sample", ref: "Memo reader · evidence drawer@2x.png", w: 1440, h: 1100 },
  runs: { route: "/runs", ref: "Runs · evaluation registry@2x.png", w: 1440, h: 900 },
};
const HEADER_PX = 112; // header is 56 CSS px (measured from the PNGs) x2

const args = process.argv.slice(2);
const bi = args.indexOf("--base");
const base = bi >= 0 ? args[bi + 1] : "http://localhost:3100";
const target = args.find((a, i) => !a.startsWith("--") && (bi < 0 || i !== bi + 1)) ?? "all";
const names = target === "all" ? Object.keys(SCREENS) : [target];
if (!names.every((n) => n in SCREENS)) {
  console.error(`unknown screen "${target}"; use ${Object.keys(SCREENS).join("|")}|all`);
  process.exit(2);
}

/** Copy img into a w x h canvas (page-bg padded), so sizes never crash the diff. */
function pad(img, w, h) {
  const out = new PNG({ width: w, height: h });
  out.data.fill(255);
  PNG.bitblt(img, out, 0, 0, Math.min(img.width, w), Math.min(img.height, h), 0, 0);
  return out;
}

function compare(a, b, region) {
  const w = Math.max(a.width, b.width);
  const h = Math.min(region?.h ?? Infinity, Math.max(a.height, b.height));
  const A = pad(a, w, h), B = pad(b, w, h);
  const diff = new PNG({ width: w, height: h });
  const n = pixelmatch(A.data, B.data, diff.data, w, h, { threshold: 0.1 });
  return { n, total: w * h, diff, A, B };
}

const browser = await chromium.launch();
fs.mkdirSync(outDir, { recursive: true });
for (const name of names) {
  const s = SCREENS[name];
  const ctx = await browser.newContext({
    viewport: { width: s.w, height: s.h },
    deviceScaleFactor: 2,
    reducedMotion: "reduce",
  });
  const page = await ctx.newPage();
  await page.goto(base + s.route, { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  await page.addStyleTag({
    content: "*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}",
  });
  const actualPath = path.join(outDir, `${name}.actual.png`);
  await page.screenshot({ path: actualPath, fullPage: !!s.full, animations: "disabled", caret: "hide" });
  await ctx.close();

  const ref = PNG.sync.read(fs.readFileSync(path.join(refDir, s.ref)));
  const act = PNG.sync.read(fs.readFileSync(actualPath));
  if (ref.width !== act.width || ref.height !== act.height) {
    console.log(`[${name}] SIZE MISMATCH ref ${ref.width}x${ref.height} vs actual ${act.width}x${act.height}; diffing padded union`);
  }
  const full = compare(ref, act);
  const head = compare(ref, act, { h: HEADER_PX });
  fs.writeFileSync(path.join(outDir, `${name}.diff.png`), PNG.sync.write(full.diff));
  const side = new PNG({ width: ref.width + act.width, height: Math.max(ref.height, act.height) });
  side.data.fill(255);
  PNG.bitblt(ref, side, 0, 0, ref.width, ref.height, 0, 0);
  PNG.bitblt(act, side, 0, 0, act.width, act.height, ref.width, 0);
  fs.writeFileSync(path.join(outDir, `${name}.side.png`), PNG.sync.write(side));
  const pct = (r) => ((100 * r.n) / r.total).toFixed(3);
  console.log(`[${name}] mismatch ${pct(full)}% (${full.n}px) | header strip (top ${HEADER_PX}px @2x) ${pct(head)}%`);
}
await browser.close();
