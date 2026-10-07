import type { AnalysisResponse, CitationResponse } from "../../../lib/api-types.ts";
import type { Block, CitationView, Group, MemoView, Section, Stat } from "./_data.ts";

// Backend citation format: the memo prompt asks for numeric markers "[1]" / "[2][3]" that index
// the registry built in app/agents/graph.py (_build_citation_registry / _build_api_citations),
// exposed as CitationResponse{index, source_type, title, url, date}. Sections are model-written
// markdown-ish text ("## Executive Summary", "1. Key Risk Factors", "**Conclusion**", "- bullets").

export type Inline = { t: "text"; text: string } | { t: "cite"; id: string };

const MARKER = /\[((?:[A-Z]?\d+)(?:\s*,\s*[A-Z]?\d+)*)\]/g;

/** Split text into literal text and citation markers. Unknown ids stay literal text. Never emits HTML. */
export function parseInline(text: string, known: ReadonlySet<string>): Inline[] {
  const out: Inline[] = [];
  let last = 0;
  const pushText = (s: string) => {
    if (!s) return;
    const prev = out[out.length - 1];
    if (prev?.t === "text") prev.text += s;
    else out.push({ t: "text", text: s });
  };
  for (const m of text.matchAll(MARKER)) {
    pushText(text.slice(last, m.index));
    for (const id of m[1].split(",").map((s) => s.trim())) {
      if (known.has(id)) out.push({ t: "cite", id });
      else pushText(`[${id}]`);
    }
    last = m.index + m[0].length;
  }
  pushText(text.slice(last));
  return out;
}

const PREFIX: Record<Group, string> = { sec: "S", news: "N", market: "M", sentiment: "N" };

export function groupOf(sourceType: string): Group {
  const t = sourceType.toLowerCase();
  if (t.includes("sec") || t.includes("filing")) return "sec";
  if (t.includes("market") || t.includes("quote")) return "market";
  if (t.includes("sentiment")) return "sentiment";
  return "news"; // unknown types fall back to news
}

/** Only http(s) URLs are ever rendered as links. */
export function safeHttpUrl(u: string | null | undefined): string | null {
  if (!u) return null;
  try {
    const p = new URL(u);
    return p.protocol === "http:" || p.protocol === "https:" ? p.href : null;
  } catch {
    return null;
  }
}

function toCitation(c: CitationResponse): CitationView {
  const group = groupOf(c.source_type);
  const url = safeHttpUrl(c.url);
  return {
    id: String(c.index),
    label: `${PREFIX[group]}${c.index}`,
    group,
    title: c.title,
    meta: c.date ?? null,
    quote: null,
    score: null,
    threshold: null,
    embedder: null,
    numbersMatched: null,
    chunkId: null,
    extractor: null,
    retrieved: null,
    url,
    linkLabel: url ? "Open source" : null,
  };
}

const clean = (s: string) =>
  s
    .replace(/\*\*|__/g, "")
    .trim()
    .replace(/^\*(.+)\*$/, "$1");
const RULE = /^-{3,}$/;
const HEADING_HASH = /^#{1,6}\s+(.*)$/;
const HEADING_BOLD = /^(?:\d+[.)]\s*)?\*\*([^*]+?)\*\*:?$/;
const HEADING_NUM = /^\d+[.)]\s+([^.!?\[\]]{2,60})$/;
const BULLET = /^(?:[-*•]|\d+[.)])\s+(.*)$/;

/** Parse the memo's markdown-ish text into sections of paragraphs and bullet lists. */
export function parseMemo(memo: string): Section[] {
  const sections: Section[] = [];
  let cur: Section = { heading: "", blocks: [] };
  const flush = () => {
    if (cur.heading || cur.blocks.length) sections.push(cur);
  };
  let para: string[] = [];
  const endPara = () => {
    if (para.length) cur.blocks.push({ kind: "p", text: para.join(" ") });
    para = [];
  };
  for (const raw of memo.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || RULE.test(line)) {
      endPara();
      continue;
    }
    const h = HEADING_HASH.exec(line)?.[1] ?? HEADING_BOLD.exec(line)?.[1] ?? HEADING_NUM.exec(line)?.[1];
    if (h) {
      endPara();
      flush();
      cur = { heading: clean(h).replace(/^\d+[.)]\s*/, ""), blocks: [] };
      continue;
    }
    const b = BULLET.exec(line);
    if (b) {
      endPara();
      const last = cur.blocks[cur.blocks.length - 1];
      if (last?.kind === "ul") last.items.push(clean(b[1]));
      else cur.blocks.push({ kind: "ul", items: [clean(b[1])] } satisfies Block);
      continue;
    }
    para.push(clean(line));
  }
  endPara();
  flush();
  return sections;
}

function firstSentence(s: string | null | undefined): string | null {
  if (!s) return null;
  const plain = clean(s).replace(MARKER, "").replace(/\s+/g, " ").replace(/ ([.,;:!?])/g, "$1").trim();
  const m = /^.*?[.!?](?=\s|$)/.exec(plain);
  const out = (m ? m[0] : plain).slice(0, 200);
  return out || null;
}

/** Model-written legal boilerplate; rendered as a warning, not body text. */
export function isDisclaimer(text: string): boolean {
  return /^(disclaimer|important caveat)\b|does not constitute|informational purposes only/i.test(text);
}

/**
 * The memo's own Executive Summary paragraph. The API's `executive_summary` is used only as a
 * fallback because the backend currently fills it with "Analysis completed for <company>".
 */
export function executiveSummary(r: AnalysisResponse): string | null {
  const section = parseMemo(r.investment_memo ?? "").find((s) => /executive summary/i.test(s.heading));
  const para = section?.blocks.find((b) => b.kind === "p");
  if (para?.kind === "p") return para.text;
  const api = r.executive_summary?.trim();
  return api && !/^Analysis completed for /i.test(api) ? api : null;
}

const MISSING_LABEL: Record<string, string> = { filings: "SEC filings", stock: "market data", news: "news", sentiment: "sentiment" };
const rate = (n: number | undefined) => (n == null ? "—" : n.toFixed(2));

/** Pure AnalysisResponse -> MemoView. Fields the API does not provide render as "—". */
export function toMemoView(r: AnalysisResponse): MemoView {
  const citations = r.citations.map(toCitation);
  const counts: Record<Group, number> = { sec: 0, news: 0, market: 0, sentiment: 0 };
  for (const c of citations) counts[c.group]++;
  const v = r.verification;
  const jobId = r.job_id ?? null;
  const missing = r.missing.map((m) => MISSING_LABEL[m] ?? m);
  // Stance, confidence and snapshot id are not exposed by the API; show what is.
  const stats: Stat[] = [
    { label: "Status", value: r.status.replace("_", " "), tone: r.status === "completed" ? "green" : "amber" },
    { label: "Grounded claims", value: rate(v?.grounded_claim_rate), tone: "ink", mono: true },
    { label: "Citation coverage", value: rate(v?.citation_coverage_rate), tone: "ink", mono: true },
    {
      label: "Evidence",
      value: missing.length ? `Missing · ${missing.join(", ")}` : r.errors.length ? "Degraded" : "Complete",
      tone: missing.length || r.errors.length ? "amber" : "green",
      href: jobId ? `/evidence?job=${encodeURIComponent(jobId)}` : undefined,
    },
  ];
  return {
    title: r.company_name || r.ticker,
    subtitle: firstSentence(executiveSummary(r)),
    runLabel: jobId ? `Run ${jobId.slice(0, 8)}` : `Run ${r.ticker}`,
    runHref: jobId ? `/workspace?job=${encodeURIComponent(jobId)}` : "/workspace",
    draft: null,
    verified: v
      ? v.passed
        ? { ok: true, text: `Verified · ${v.grounded_claims}/${v.total_claims} claims checked` }
        : { ok: false, text: `Not verified · ${v.grounded_claims}/${v.total_claims} claims grounded` }
      : null,
    stats,
    sections: parseMemo(r.investment_memo ?? ""),
    callout: null,
    citations,
    counts,
    totalCitations: citations.length,
    sourceCount: GROUPS_WITH(counts),
    initialId: citations[0]?.id ?? null,
  };
}

function GROUPS_WITH(counts: Record<Group, number>): number {
  return Object.values(counts).filter((n) => n > 0).length;
}
