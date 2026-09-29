import Link from "next/link";

const REPO = "https://github.com/nilesh-auradkar05/Financial-Analyst-Agent";
const EXT = { target: "_blank", rel: "noopener noreferrer" } as const;
const FOCUS =
  "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-navy";

const NAV = [
  { label: "How it works", href: "#how-it-works" },
  { label: "Evidence", href: "#evidence" },
  { label: "Evaluation", href: "/runs" },
  { label: "Docs", href: `${REPO}/tree/main/docs`, external: true },
  { label: "GitHub", href: REPO, external: true },
];

const STACK = ["AWS Bedrock", "LangGraph", "SEC EDGAR", "FinBERT", "FastAPI", "Qdrant"];

const STEPS = [
  {
    n: "01",
    title: "Gather",
    body: "Market, SEC, news and sentiment agents run in parallel, each owning one evidence type.",
    tag: "yfinance · edgartools · tavily · FinBERT",
  },
  {
    n: "02",
    title: "Seal",
    body: "Everything retrieved is frozen into an EvidenceSnapshot and hashed. Replay, caching and lineage all key off it.",
    tag: "snapshot_hash",
  },
  {
    n: "03",
    title: "Write & review",
    body: "A writer drafts with inline citations. A reviewer scores each claim and sends weak ones back.",
    tag: "reviewer loop · ≤ 3 passes",
  },
  {
    n: "04",
    title: "Gate",
    body: "Only memos above the grounding threshold publish. Missing evidence is reported, never hidden.",
    tag: "completed · degraded · evidence_missing",
  },
];

const FEATURES = [
  {
    icon: "#",
    title: "Replayable runs",
    body: "Re-run any memo against its sealed snapshot to reproduce the output or compare models on identical evidence.",
  },
  {
    icon: "!",
    title: "Honest status",
    body: "A timed-out source marks the run degraded. An un-ingested ticker returns evidence_missing instead of a confident memo.",
  },
  {
    icon: "⇄",
    title: "Model routing",
    body: "Claude, DeepSeek and OpenAI models on AWS Bedrock behind one typed interface, with a local Ollama fallback.",
  },
  {
    icon: "∑",
    title: "Evaluation registry",
    body: "Every run appends grounding and coverage scores to an append-only registry, one variable changed at a time.",
  },
];

const SOURCES = [
  { dot: "bg-green", text: "SEC filings · 184 chunks" },
  { dot: "bg-green", text: "Market data · 252 bars" },
  { dot: "bg-amber", text: "News · degraded, flagged" },
  { dot: "bg-blue", text: "Verifier · pass 2" },
];

function Chip({ kind, children }: { kind: "navy" | "blue" | "gold"; children: string }) {
  const c =
    kind === "navy"
      ? "bg-navy text-white"
      : kind === "blue"
        ? "bg-[#e8edf5] text-navy"
        : "bg-gold-tint text-gold";
  return (
    <span className={`mx-px rounded px-1.5 py-[2px] align-baseline font-mono text-[11px] leading-none ${c}`}>
      {children}
    </span>
  );
}

function Arrow() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3 8h10M9 4l4 4-4 4" />
    </svg>
  );
}

function MemoCard() {
  return (
    <div className="relative h-[438px] w-full max-w-[534px]">
      <div className="absolute inset-0 overflow-hidden rounded-2xl border border-border bg-surface px-8 pt-[25px] shadow-[0_18px_40px_-18px_rgba(15,26,42,0.18)]">
        <div className="flex items-center text-[11px]">
          <span className="font-semibold tracking-[0.14em] text-navy">INVESTMENT MEMO</span>
          <span className="mx-3 h-3 w-px bg-border" />
          <span className="text-muted">NWSC · Draft v2</span>
          <span className="ml-auto flex items-center gap-1.5 text-[12px] font-medium text-green">
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M3 8.5l3.2 3L13 4.5" />
            </svg>
            22/22 verified
          </span>
        </div>
        <h2 className="mt-[14px] font-serif text-[26px] leading-[34px] font-medium text-ink">
          Data-center demand carries the quarter; inventory is the watch item.
        </h2>
        <p className="mt-[17px] text-[14.6px] leading-[26.3px] text-body">
          Data-center revenue rose to 61% of total mix, up from 48% a year earlier{" "}
          <Chip kind="navy">S2</Chip>. Channel inventory rose for a second consecutive quarter{" "}
          <Chip kind="blue">S4</Chip>, and news sentiment is net positive <Chip kind="gold">N3</Chip>.
        </p>
        <div className="absolute inset-x-8 top-[360px] border-t border-border" />
        <span className="absolute top-[391px] right-[208px] text-[12px] font-medium text-ink">Claim sources</span>
      </div>
      <div
        role="note"
        aria-label="Source S2 excerpt"
        className="absolute top-[355px] -left-10 h-[165px] w-[359px] rounded-[14px] bg-ink px-5 pt-4 text-white shadow-[0_24px_40px_-12px_rgba(15,26,42,0.45)]"
      >
        <div className="flex items-center gap-2.5 text-[13px] font-semibold">
          <span className="rounded bg-white px-1.5 py-[3px] font-mono text-[11px] leading-none font-medium text-ink">S2</span>
          Form 10-Q · MD&amp;A · p.31
        </div>
        <p className="mt-[13px] font-serif text-[15px] leading-[22px] text-[#c7ced9] italic">
          “...the Data Center segment represented{" "}
          <mark className="bg-[#2a4775] text-white [box-decoration-break:clone]">
            61% of total net revenue, compared with 48%
          </mark>{" "}
          in the prior-year period...”
        </p>
        <div className="mt-[9px] flex items-center gap-3">
          <div className="relative h-1 flex-1 rounded-full bg-[#2a3a52]">
            <div className="h-full w-[78%] rounded-full bg-[#5fe0a8]" />
            <span className="absolute top-[-4px] left-[45.5%] h-3 w-[2px] bg-[#e8d5a8]" />
          </div>
          <span className="font-mono text-[12px] text-[#5fe0a8]">0.78</span>
        </div>
      </div>
      <ul className="absolute top-[373px] right-[23px] w-[180px] rounded-[10px] border border-border bg-surface px-3 py-[9px] shadow-[0_12px_28px_-10px_rgba(15,26,42,0.22)]">
        {SOURCES.map((s) => (
          <li key={s.text} className="flex items-center gap-2 py-[2.5px] text-[12px] leading-[18px] text-ink">
            <span className={`h-2 w-2 rounded-full ${s.dot}`} />
            {s.text}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function Home() {
  return (
    <>
      <header className="h-[72px] border-b border-border bg-page">
        <div className="mx-auto flex h-full max-w-[1200px] items-center px-6 min-[1260px]:px-0">
          <Link href="/" aria-label="Alpha Analyst home" className={`flex items-center gap-[10.5px] ${FOCUS}`}>
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-navy font-serif text-[19px] leading-none text-[#e8d5a8]">α</span>
            <span className="text-[16px] font-semibold text-ink">Alpha Analyst</span>
          </Link>
          <nav aria-label="Primary" className="ml-[22.5px] hidden items-center md:flex">
            {NAV.map((n) => (
              <a
                key={n.label}
                href={n.href}
                {...("external" in n ? EXT : {})}
                className={`rounded px-4 py-1 text-[14.2px] text-body ${FOCUS}`}
              >
                {n.label}
              </a>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-[25px]">
            {/* ponytail: no auth until S7 */}
            <a href="#" className={`rounded px-2 text-[14.5px] font-semibold text-ink ${FOCUS}`}>Sign in</a>
            <Link href="/workspace" className={`flex h-[42px] items-center rounded-lg bg-navy px-[18px] text-[14.5px] font-semibold text-white ${FOCUS}`}>
              Try a live run
            </Link>
          </div>
        </div>
      </header>

      <main>
        <section className="relative mx-auto h-auto max-w-[1200px] px-6 pt-[111px] pb-16 min-[1260px]:h-[704px] min-[1260px]:px-0 min-[1260px]:pb-0">
          <div>
            <span className="inline-flex h-[30px] items-center gap-2 rounded-full border border-[#efe1bf] bg-gold-tint px-3 text-[12.3px] font-medium text-gold">
              <span className="h-[6px] w-[6px] rounded-full bg-gold" />
              Multi-agent equity research
            </span>
            <h1 className="mt-[30px] max-w-[560px] font-serif text-[56.2px] leading-[67px] font-normal tracking-[-0.042em] text-ink">
              Investment memos where every claim shows its source.
            </h1>
            <p className="mt-[25px] max-w-[560px] text-[18.4px] leading-[30.4px] text-body">
              Specialist agents gather SEC filings, market data, news and sentiment into one hashed evidence snapshot. A writer drafts, a reviewer checks each claim against that evidence, and nothing ships below the grounding gate.
            </p>
            <div className="mt-[29px] flex gap-3">
              <Link href="/workspace" className={`flex h-[52px] items-center gap-2.5 rounded-[10px] bg-navy px-[25.5px] text-[15px] font-semibold text-white ${FOCUS}`}>
                Analyze a ticker <Arrow />
              </Link>
              <Link href="/memos/sample" className={`flex h-[52px] items-center rounded-[10px] border border-border bg-surface px-6 text-[15px] font-medium text-ink ${FOCUS}`}>
                Read a sample memo
              </Link>
            </div>
          </div>
          <div className="mt-12 min-[1260px]:absolute min-[1260px]:top-[96px] min-[1260px]:left-[664px] min-[1260px]:mt-0 min-[1260px]:w-[534px]">
            <MemoCard />
          </div>
        </section>

        <section aria-label="Built on" className="border-y border-border bg-[#fafaf7]">
          <div className="mx-auto flex h-[87px] max-w-[1200px] items-center gap-12 px-6 min-[1260px]:px-0">
            <span className="text-[13px] text-muted">Built on</span>
            <ul className="flex flex-1 items-center justify-between text-[17.5px] font-semibold text-[#5a6474]">
              {STACK.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </div>
        </section>

        <section id="how-it-works" className="mx-auto max-w-[1200px] scroll-mt-4 px-6 pt-[112px] pb-[103px] min-[1260px]:px-0">
          <div className="flex items-end justify-between gap-10">
            <div>
              <p className="text-[12.3px] font-semibold tracking-[0.14em] text-gold">HOW IT WORKS</p>
              <h2 className="mt-[14px] max-w-[700px] font-serif text-[38.8px] leading-[49px] font-normal tracking-[-0.02em] text-ink">
                Four specialists, one sealed record, and a reviewer that says no.
              </h2>
            </div>
            <p className="mb-[1px] hidden w-[400px] text-[15.2px] leading-[26.3px] text-body md:block">
              Agents are split by the evidence they own, not to save wall-clock time. That keeps each one small, testable and replaceable.
            </p>
          </div>
          <ol className="mt-[56px] grid gap-y-8 border-t-2 border-navy pt-[26px] sm:grid-cols-2 lg:grid-cols-4">
            {STEPS.map((s) => (
              <li key={s.n} className="pr-9">
                <span className="font-mono text-[13px] text-gold">{s.n}</span>
                <h3 className="mt-[14px] text-[19.4px] leading-7 font-semibold text-ink">{s.title}</h3>
                <p className="mt-[12px] text-[14.5px] leading-[24.8px] text-body">{s.body}</p>
                <p className="mt-[13px] font-mono text-[12px] leading-[15px] text-navy">{s.tag}</p>
              </li>
            ))}
          </ol>
        </section>

        <section id="evidence" className="scroll-mt-4 bg-navy">
          <div className="mx-auto flex h-[388px] max-w-[1200px] items-start justify-between gap-10 px-6 pt-[86px] min-[1260px]:px-0">
            <div className="w-[420px]">
              <p className="text-[12.3px] font-semibold tracking-[0.14em] text-[#e8d5a8]">MEASURED, NOT CLAIMED</p>
              <h2 className="mt-2 font-serif text-[34.2px] leading-[44.6px] font-normal tracking-[-0.02em] text-white">Grounding is scored on every run.</h2>
              <p className="mt-[18px] max-w-[390px] text-[14.4px] leading-[25px] text-[#c3cde0]">
                Figures below are candidate baselines on live evidence. By policy they can’t be approved until replayed against frozen snapshots.
              </p>
            </div>
            <dl className="mt-[29px] grid h-[158px] w-[700px] grid-cols-3 divide-x divide-[#2a4775] rounded-[14px] border border-[#2a4775]">
              <div className="px-[29px] pt-[35px]">
                <dd className="font-mono text-[43px] leading-[48px] font-semibold text-white">0.904</dd>
                <dt className="mt-3 text-[14px] leading-[17px] text-[#c3cde0]">
                  grounded claim rate <span className="font-mono text-[13px] text-[#8fa0bd]">± 0.062</span>
                </dt>
              </div>
              <div className="px-[29px] pt-[35px]">
                <dd className="font-mono text-[43px] leading-[48px] font-semibold text-white">0.925</dd>
                <dt className="mt-3 text-[14px] leading-[17px] text-[#c3cde0]">
                  citation coverage <span className="font-mono text-[13px] text-[#8fa0bd]">± 0.049</span>
                </dt>
              </div>
              <div className="px-[29px] pt-[35px]">
                <dd className="font-mono text-[43px] leading-[48px] font-semibold text-[#e8d5a8]">1 hash</dd>
                <dt className="mt-3 text-[14px] leading-[17px] text-[#c3cde0]">to replay any run exactly</dt>
              </div>
            </dl>
          </div>
        </section>

        <section className="mx-auto max-w-[1200px] px-6 pt-[112px] min-[1260px]:px-0">
          <h2 className="max-w-[640px] font-serif text-[38.8px] leading-[49px] font-normal tracking-[-0.02em] text-ink">
            Built the way a research desk would audit it.
          </h2>
          <ul className="mt-[46px] grid gap-5 md:grid-cols-2">
            {FEATURES.map((f) => (
              <li key={f.title} className="flex h-[147px] gap-5 rounded-2xl border border-border bg-surface px-8 pt-8">
                <span aria-hidden="true" className="flex h-11 w-11 shrink-0 items-center justify-center rounded-[10px] bg-[#e8edf5] text-[12px] text-navy">
                  {f.icon}
                </span>
                <div>
                  <h3 className="text-[18.3px] leading-6 font-semibold text-ink">{f.title}</h3>
                  <p className="mt-[6px] text-[14.3px] leading-[25px] text-body">{f.body}</p>
                </div>
              </li>
            ))}
          </ul>
        </section>

        <section className="mx-auto mt-[115px] h-[197px] max-w-[1200px] overflow-hidden px-6 min-[1260px]:px-0">
          <div className="flex min-h-[240px] items-start justify-between gap-8 rounded-[20px] bg-ink px-[72px] pt-[63px]">
            <div>
              <h2 className="font-serif text-[34.3px] leading-[48px] font-normal whitespace-nowrap text-white">Watch a memo get built — and checked.</h2>
              <p className="mt-[9px] text-[15px] leading-6 text-[#b0bacb]">Enter a ticker. See every agent, every source and every revision.</p>
            </div>
            <div className="mt-[14px] flex shrink-0 gap-4">
              <Link href="/workspace" className={`flex h-[52px] items-center rounded-[10px] bg-[#e8d5a8] px-[24px] text-[15px] font-semibold text-ink ${FOCUS}`}>
                Start a live run
              </Link>
              <Link href="/runs" className={`flex h-[52px] items-center rounded-[10px] border border-[#2a3a52] px-6 text-[15px] font-medium text-white ${FOCUS}`}>
                View evaluation
              </Link>
            </div>
          </div>
        </section>
      </main>
    </>
  );
}
