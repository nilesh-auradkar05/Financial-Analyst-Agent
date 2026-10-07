// View model for the memo reader. `sample` is the design fixture (/memos/sample);
// live jobs are mapped into the same shape by toMemoView() in _map.ts.

export type Group = "sec" | "news" | "market" | "sentiment";

export const GROUP_LABEL: Record<Group, string> = {
  sec: "SEC",
  news: "News",
  market: "Market",
  sentiment: "Sentiment",
};
export const GROUP_ORDER: Group[] = ["sec", "news", "market", "sentiment"];

/** A piece of quote text; `hl` marks the highlighted (grounded) span. */
export interface QuoteSpan {
  text: string;
  hl?: boolean;
}

export interface CitationView {
  /** Marker id as it appears in memo text: "S2" (fixture) or "3" (live, [3]). */
  id: string;
  /** Chip label, e.g. "S2". */
  label: string;
  group: Group;
  title: string;
  /** Right-hand meta in the drawer card, e.g. "Item 2 · MD&A". */
  meta: string | null;
  quote: QuoteSpan[] | null;
  /** Grounding cosine 0..1, "exact" for deterministic matches, null when unknown. */
  score: number | "exact" | null;
  threshold: number | null;
  embedder: string | null;
  numbersMatched: string | null;
  chunkId: string | null;
  extractor: string | null;
  retrieved: string | null;
  url: string | null;
  linkLabel: string | null;
}

export type Block = { kind: "p"; text: string } | { kind: "ul"; items: string[] };

export interface Section {
  heading: string;
  blocks: Block[];
}

export type Tone = "green" | "amber" | "ink" | "navy";

export interface Stat {
  label: string;
  value: string;
  tone: Tone;
  mono?: boolean;
  href?: string;
}

export interface MemoView {
  title: string;
  subtitle: string | null;
  runLabel: string;
  runHref: string;
  draft: string | null;
  verified: { ok: boolean; text: string } | null;
  stats: Stat[];
  sections: Section[];
  callout: { title: string; body: string } | null;
  citations: CitationView[];
  counts: Record<Group, number>;
  totalCitations: number;
  sourceCount: number;
  /** Marker id of the citation shown in the drawer initially. */
  initialId: string | null;
}

const blank = {
  meta: null,
  quote: null,
  score: null,
  threshold: null,
  embedder: null,
  numbersMatched: null,
  chunkId: null,
  extractor: null,
  retrieved: null,
  url: null,
  linkLabel: null,
} as const;

export const sample: MemoView = {
  title: "Northwind Semiconductor",
  subtitle: "Data-center demand carries the quarter; inventory is the watch item.",
  runLabel: "Run r_0f3a91",
  runHref: "/workspace",
  draft: "Draft v2",
  verified: { ok: true, text: "Verified · 22/22 claims checked" },
  stats: [
    { label: "Stance", value: "Constructive", tone: "green" },
    { label: "Confidence", value: "Medium", tone: "ink" },
    { label: "Snapshot", value: "9f3a…c21e", tone: "navy", mono: true },
    { label: "Evidence", value: "Degraded · news", tone: "amber" },
  ],
  sections: [
    {
      heading: "Thesis",
      blocks: [
        {
          kind: "p",
          text: "Data-center revenue rose to 61% of total mix in the quarter, up from 48% a year earlier [S2]. Management guided next-quarter revenue above consensus, citing accelerator attach rates [S3]. Price action has outrun the sector over 90 days [M1], and news sentiment is net positive [N3].",
        },
      ],
    },
    {
      heading: "Risks",
      blocks: [
        {
          kind: "ul",
          items: [
            "Channel inventory rose for a second consecutive quarter [S4].",
            "Top-two customer concentration is disclosed at 38% of revenue [S7].",
            "Export-control exposure is referenced but unquantified in filings [S9].",
          ],
        },
      ],
    },
  ],
  callout: {
    title: "Reviewer loop revised 2 claims",
    body: "v1 stated gross margin “expanded 400 bp”; nearest evidence scored 0.38 (threshold 0.45). v2 cites the 10-Q figure directly. A second claim was dropped for lack of support.",
  },
  citations: [
    {
      ...blank,
      id: "S2",
      label: "S2",
      group: "sec",
      title: "Form 10-Q · Q2 FY26",
      meta: "Item 2 · MD&A",
      quote: [
        { text: "“…revenue from the Data Center segment represented " },
        { text: "61% of total net revenue,", hl: true },
        { text: " " },
        { text: "compared with 48%", hl: true },
        { text: " in the prior-year period, driven primarily by accelerator shipments…”" },
      ],
      score: 0.78,
      threshold: 0.45,
      embedder: "qwen3-embedding",
      numbersMatched: "2/2",
      chunkId: "sec:10q:0027:mdna:031",
      extractor: "edgartools 2.x · v3",
      retrieved: "rank 1 of 12",
      // Design fixture: a generic SEC EDGAR landing page stands in for the real filing URL.
      url: "https://www.sec.gov/edgar/search/",
      linkLabel: "Open filing at page 31",
    },
    { ...blank, id: "S3", label: "S3", group: "sec", title: "8-K · Ex. 99.1 · Outlook", score: 0.71 },
    { ...blank, id: "S4", label: "S4", group: "sec", title: "10-Q · Note 5 · Inventories", score: 0.66 },
    { ...blank, id: "S7", label: "S7", group: "sec", title: "10-Q · Concentration of risk", score: 0.74 },
    { ...blank, id: "S9", label: "S9", group: "sec", title: "10-K · Item 1A · Risk factors", score: 0.52 },
    { ...blank, id: "M1", label: "M1", group: "market", title: "yfinance · 90d relative return", score: "exact" },
    { ...blank, id: "N3", label: "N3", group: "sentiment", title: "FinBERT aggregate · 13 articles", score: 0.61 },
  ],
  counts: { sec: 18, news: 7, market: 4, sentiment: 2 },
  totalCitations: 31,
  sourceCount: 4,
  initialId: "S2",
};
