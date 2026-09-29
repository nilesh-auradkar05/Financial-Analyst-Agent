import Link from "next/link";
import type { BarTone, MemoSegment, TraceTone, WorkspaceRun } from "../_data";
import { AgentGraph } from "./AgentGraph";

const TRACE_COLOR: Record<TraceTone, string> = {
  green: "text-green",
  amber: "text-[#8a4b06]",
  navy: "text-navy",
  body: "text-body",
  blue: "text-[#2952cc]",
};
const BAR_COLOR: Record<BarTone, string> = {
  green: "bg-green",
  amber: "bg-amber",
  navy: "bg-navy",
  blue: "bg-[#2952cc]",
};
const CHANGE_COLOR = { up: "text-green", down: "text-red", flat: "text-muted" } as const;

const CARD = "rounded-[14px] border border-border bg-surface";
const EYEBROW = "font-sans text-[11px] font-semibold uppercase leading-none tracking-[0.08em] text-body";

function Legend() {
  const items: [string, string][] = [
    ["Completed", "bg-green"],
    ["Running", "bg-[#2952cc]"],
    ["Degraded", "bg-amber"],
    ["Queued", "border border-[#9ca3af] bg-transparent"],
  ];
  return (
    <ul className="m-0 flex list-none items-center gap-[16.5px] p-0 text-[12px] text-body">
      {items.map(([label, cls]) => (
        <li key={label} className="flex items-center gap-[5px]">
          <span aria-hidden className={`size-2 rounded-[2px] ${cls}`} />
          {label}
        </li>
      ))}
    </ul>
  );
}

function Stat({ value, label, suffix, warn }: { value: string; label: string; suffix?: string | null; warn?: boolean }) {
  return (
    <div>
      <div className={`text-[22px] font-medium leading-[26px] ${warn ? "text-[#8a4b06]" : "text-ink"}`}>
        {value}
        {suffix && <span className="text-[15px] font-normal text-muted">{suffix}</span>}
      </div>
      <div className="mt-[2px] text-[12.5px] leading-[18px] text-muted">{label}</div>
    </div>
  );
}

function Meter({ name, value, gate }: { name: string; value: number | null; gate: number }) {
  return (
    <div>
      <div className="flex items-baseline justify-between font-mono text-[12px] tracking-[0.2px] leading-[16px]">
        <span className="text-body">{name}</span>
        <span className="font-medium text-green">{value == null ? "—" : value.toFixed(2)}</span>
      </div>
      <div
        role="meter"
        aria-label={name}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={value ?? undefined}
        aria-valuetext={value == null ? "not available" : `${value.toFixed(2)}, publish gate ${gate}`}
        className="relative mt-[6.5px] h-[6px] rounded-full bg-[#efede7]"
      >
        <div className="h-full rounded-full bg-green" style={{ width: `${(value ?? 0) * 100}%` }} />
        <span aria-hidden className="absolute top-1/2 h-[12.5px] w-[1.5px] -translate-y-1/2 bg-ink" style={{ left: `${gate * 100}%` }} />
      </div>
    </div>
  );
}

function Memo({ segs }: { segs: MemoSegment[] }) {
  return (
    <p className="mt-[13px] text-[13px] leading-[22px] tracking-[-0.2px] text-[#3a4556]">
      {segs.map((s, i) => {
        if (s.kind === "text") return <span key={i}>{s.text}</span>;
        if (s.kind === "cursor")
          return <span key={i} aria-hidden className="ml-[3px] inline-block h-[14px] w-[3px] translate-y-[2px] bg-[#2952cc]" />;
        return (
          <span
            key={i}
            className={`rounded-[5px] px-[5px] py-[1px] font-mono text-[11px] tracking-[-0.4px] ${s.tone === "sec" ? "bg-[#e8edf5] text-navy" : "bg-gold-tint text-[#8a4b06]"}`}
          >
            {s.label}
          </span>
        );
      })}
    </p>
  );
}

export function WorkspaceView({ run }: { run: WorkspaceRun }) {
  const { evidence: ev, grounding: g, time: t } = run;
  return (
    <main className="mx-auto grid w-[1440px] max-w-full grid-cols-[992px_380px] gap-x-5 px-6 pt-6">
      <div className="flex flex-col">
        <section aria-label="Run summary" className="relative h-[97px]">
          <div className="flex items-center gap-3.5 pt-[9px]">
            <h1 className="m-0 font-serif text-[33px] font-normal leading-[40px] tracking-[-0.005em] text-ink">{run.company}</h1>
            <span className="rounded-[6px] border border-border bg-surface px-2 py-[3px] font-mono text-[12px] tracking-[0.2px] leading-[15px] text-body">{run.ticker}</span>
          </div>
          <p className="m-0 mt-[8.5px] flex items-center text-[13px] leading-[20px] text-muted">
            <span className="mr-[14px] font-mono text-[13px] tracking-[-0.2px] font-medium text-ink">{run.price}</span>
            <span className={`font-mono text-[13px] tracking-[-0.2px] ${CHANGE_COLOR[run.changeTone]}`}>{run.change}</span>
            <span aria-hidden className="mx-4 h-[13px] w-px bg-border" />
            <span>
              Run <span className="font-mono text-[13px] tracking-[-0.2px] text-body">{run.runId}</span>
            </span>
            <span aria-hidden className="mx-4 h-[13px] w-px bg-border" />
            <span>
              Started {run.started} · {run.elapsed} elapsed
            </span>
          </p>
          <span className="absolute right-0 top-[44px] flex h-[31.5px] items-center gap-2 rounded-[8px] bg-[#e7edfc] px-3 text-[12px] font-semibold text-[#2952cc]">
            <span aria-hidden className="size-[7px] rounded-full bg-[#2952cc]" />
            {run.pill}
          </span>
        </section>

        <section aria-label="Agent graph" className={`mt-0 overflow-hidden ${CARD}`}>
          <header className="flex h-[46px] items-center border-b border-border px-5">
            <h2 className="m-0 text-[14px] font-semibold tracking-[-0.01em] text-ink">Agent graph</h2>
            <span className="ml-3 font-mono text-[11px] tracking-[-0.4px] text-muted">{run.graphSubtitle}</span>
            <div className="ml-auto">
              <Legend />
            </div>
          </header>
          <AgentGraph nodes={run.graph} reviseLabel={run.reviseLabel} gateLabel={run.gateLabel} />
        </section>

        <div className="mt-[23px] grid grid-cols-3 gap-4">
          <section aria-label="Evidence" className={`h-[239px] px-5 pt-5 ${CARD}`}>
            <div className="flex items-center justify-between">
              <h3 className={`m-0 ${EYEBROW}`}>Evidence</h3>
              <span className="font-mono text-[12px] leading-none tracking-[0.2px] text-muted">{ev.typesLabel}</span>
            </div>
            <div className="mt-[16px] grid grid-cols-2 content-start gap-x-4 gap-y-[13px]">
              <Stat value={ev.sec} label="SEC chunks" />
              <Stat value={ev.newsCount} suffix={ev.newsOf} warn={ev.newsWarn} label="News articles" />
              <Stat value={ev.priceBars} label="Price bars" />
              <Stat value={ev.sentiment} label="Sentiment scores" />
            </div>
          </section>

          <section aria-label="Grounding" className={`h-[239px] px-5 pt-5 ${CARD}`}>
            <div className="-mt-px flex items-center justify-between">
              <h3 className={`m-0 ${EYEBROW}`}>Grounding · live</h3>
              <span className="rounded-[6px] bg-amber-tint px-2 py-[2px] text-[10.5px] font-medium leading-[14px] text-[#8a4b06]">{g.badge}</span>
            </div>
            <div className="mt-[14px] flex flex-col gap-[13px]">
              <Meter name="grounded_claim_rate" value={g.grounded} gate={g.gate} />
              <Meter name="citation_coverage" value={g.citation} gate={g.gate} />
            </div>
            <p className="m-0 mt-[13px] text-[11px] leading-[16px] text-muted">Tick = publish gate · live evidence can’t be approved</p>
          </section>

          <section aria-label="Time by node" className={`h-[239px] px-5 pt-5 ${CARD}`}>
            <div className="flex items-center justify-between">
              <h3 className={`m-0 ${EYEBROW}`}>Time by node</h3>
              <span className="font-mono text-[12px] leading-none tracking-[0.2px] text-muted">{t.cap}</span>
            </div>
            <div role="img" aria-label="Bar chart of time spent per node" className="mt-[12px] flex h-[62.5px] items-end gap-1">
              {t.bars.map((b, i) => (
                <div key={i} className={`flex-1 rounded-t-[2px] ${BAR_COLOR[b.tone]}`} style={{ height: `${b.h * 100}%` }} />
              ))}
            </div>
            <div className="mt-[14px] flex items-baseline justify-between text-[12.5px] leading-[16px] text-body">
              <span>
                <span className="font-mono text-[12px] tracking-[0.2px] text-ink">{t.tokens}</span> tokens
              </span>
              <span>
                <span className="font-mono text-[12px] tracking-[0.2px] text-ink">{t.cost}</span> so far
              </span>
            </div>
          </section>
        </div>
      </div>

      <aside className="flex flex-col gap-[21px]" aria-label="Run details">
        <section aria-label="Trace" className={`h-[521.5px] overflow-hidden ${CARD}`}>
          <header className="flex h-[47px] items-center justify-between border-b border-border px-[18px]">
            <h2 className="m-0 text-[13px] font-semibold text-ink">Trace</h2>
            {run.streaming && (
              <span className="flex items-center gap-1.5 text-[12px] font-medium text-[#2952cc]">
                <span aria-hidden className="size-[6px] rounded-full bg-[#2952cc]" />
                Streaming
              </span>
            )}
          </header>
          {run.trace.length === 0 ? (
            <p className="m-0 px-[18px] py-4 text-[13px] text-muted">{run.traceNote}</p>
          ) : (
            <ol className="m-0 list-none p-0">
              {run.trace.map((r) => (
                <li key={r.time} className="grid grid-cols-[90px_80px_1fr] border-b border-[#f0eee9] px-[18px] py-[7.5px] font-mono text-[11px] tracking-[-0.4px] leading-[16px]">
                  <time className="text-[#8b93a1]">{r.time}</time>
                  <span className={`font-medium ${TRACE_COLOR[r.tone]}`}>{r.agent}</span>
                  <span className="text-[#3a4556]">{r.message}</span>
                </li>
              ))}
            </ol>
          )}
        </section>

        <section aria-label="Memo draft" className={`h-[377.5px] px-[19.5px] pt-5 ${CARD}`}>
          <header className="flex h-4 items-center justify-between">
            <h2 className="m-0 text-[13px] font-semibold text-ink">
              Memo draft <span className="text-[11px] font-normal text-muted">{run.memo.version}</span>
            </h2>
            <Link href={run.memo.href} className="text-[12.5px] font-medium text-navy focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-navy">
              Open reader →
            </Link>
          </header>
          {run.memo.headline && (
            <p className="m-0 mt-[12px] font-serif text-[20px] font-medium leading-[27.5px] text-ink">{run.memo.headline}</p>
          )}
          <Memo segs={run.memo.body} />
        </section>
      </aside>
    </main>
  );
}
