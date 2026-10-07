import Link from "next/link";
import { AppHeader, Avatar } from "@/components/AppHeader";
import { recentJobs } from "@/lib/jobs.server";
import { Filters } from "./_components/Filters";
import { GATE, GROUNDED_BARS, OUTCOMES, RUNS, STATUSES, SUMMARY, summarize, toRunRow, type RunStatus } from "./_data";

const BLUE = { bg: "#e7edfc", fg: "#2952cc" };
const PILL: Record<RunStatus, { bg: string; fg: string }> = {
  degraded: { bg: "#fbf0dd", fg: "#8a4b06" },
  completed: { bg: "#e3f2ec", fg: "#0b6146" },
  evidence_missing: { bg: "#fce9e7", fg: "#9f1d12" },
  failed: { bg: "#fce9e7", fg: "#9f1d12" },
  pending: BLUE,
  running: BLUE,
};

const CARD = "rounded-[14px] border border-border bg-surface px-5 pt-[15px]";
const TH = "px-6 py-0 text-[11px] font-semibold uppercase tracking-[0.075em] text-body";
const LINK = "font-mono text-[12.4px] text-navy underline underline-offset-2 focus-visible:outline-2 focus-visible:outline-navy";
const fmt2 = (n: number | null) => (n === null ? "—" : n.toFixed(2));

export const metadata = { title: "Runs · Alpha Analyst" };

export default async function Page({ searchParams }: { searchParams: Promise<{ status?: string | string[] }> }) {
  const { status: raw } = await searchParams;
  const requested = Array.isArray(raw) ? raw[0] : raw;
  const status = STATUSES.find((s) => s === requested) ?? "";

  // This browser's own runs; the design sample only when there are none.
  const mine = (await recentJobs()).map(toRunRow);
  const sample = mine.length === 0;
  const all = sample ? RUNS : mine;
  const rows = status ? all.filter((r) => r.status === status) : all;
  const summary = sample
    ? { ...SUMMARY, outcomes: OUTCOMES, bars: GROUNDED_BARS, gateTop: 4 }
    : { ...summarize(mine), gateTop: 50 * (1 - GATE) };
  const totalW = summary.outcomes.reduce((a, o) => a + o.weight, 0);

  return (
    <>
      <AppHeader active="runs">
        <Avatar />
      </AppHeader>
      <main className="mx-auto w-full max-w-[1440px] px-10 pt-[33px] pb-8">
        <div className="flex items-end justify-between">
          <div>
            <h1 className="font-serif text-[33.5px] leading-[1.1] font-medium text-ink">Runs</h1>
            <p className="mt-[8.5px] text-[14px] text-muted">
              {sample ? (
                <span className="rounded-[6px] bg-amber-tint px-2 py-[3px] text-[12px] font-semibold text-[#8a4b06]">
                  Sample data · runs you start from this browser will appear here
                </span>
              ) : (
                "Analyses started from this browser. Model, cost and snapshot id are not reported by the API yet."
              )}
            </p>
          </div>
          <div className="mb-1 flex gap-[9.5px]">
            <Filters status={status} options={STATUSES} />
          </div>
        </div>

        <section aria-label="Run summary" className="mt-[21.5px] grid grid-cols-4 gap-4">
          {[
            { label: "grounded_claim_rate", ...summary.groundedClaimRate },
            { label: "citation_coverage", ...summary.citationCoverage },
          ].map((m) => (
            <div key={m.label} className={`${CARD} h-[126px]`}>
              <div className="font-mono text-[12.6px] text-body">{m.label}</div>
              <div className="mt-[9px] flex items-baseline gap-2">
                <span className="font-mono text-[29.5px] leading-[1.2] font-medium text-ink">{m.value}</span>
                <span className="font-mono text-[12.7px] text-muted">{m.pm}</span>
              </div>
              <span className="mt-[10px] inline-block rounded bg-amber-tint px-2 py-[1.5px] text-[10.5px] font-medium text-[#8a4b06]">
                {summary.tag}
              </span>
            </div>
          ))}

          <div className={`${CARD} h-[126px]`}>
            <div className="text-[11.85px] text-body">Run outcomes</div>
            <div
              role="img"
              aria-label={`Run outcomes: ${summary.outcomes.map((o) => o.status).join(", ")} proportions`}
              className="mt-[19px] flex h-2 gap-[1.5px] overflow-hidden rounded-full"
            >
              {summary.outcomes.map((o) => (
                <div key={o.status} style={{ flexGrow: o.weight / totalW, background: o.color }} />
              ))}
            </div>
            <ul className="mt-[14px] flex flex-wrap gap-x-[14px] gap-y-1 text-[12px] text-body">
              {summary.outcomes.map((o) => (
                <li key={o.status} className="flex items-center gap-[5.5px]">
                  <span aria-hidden className="h-[7px] w-[7px] rounded-[2px]" style={{ background: o.color }} />
                  {o.status}
                </li>
              ))}
            </ul>
          </div>

          <div className={`${CARD} h-[126px]`}>
            <div className="text-[11.9px] text-body">Grounded rate · last 12 runs</div>
            <div
              role="img"
              aria-label={`Grounded rate for the last ${summary.bars.length} runs; red bars fall below the dashed ${GATE} line`}
              className="relative mt-[15.5px] flex h-[50px] items-end gap-[4.6px]"
            >
              {summary.bars.map((b, i) => (
                <div
                  key={i}
                  className="max-w-[40px] flex-1 rounded-[3px]"
                  style={{ height: b.h, background: b.bad ? "#b42318" : "#14305a" }}
                />
              ))}
              <div aria-hidden className="absolute inset-x-0 border-t border-dashed border-gold" style={{ top: summary.gateTop }} />
            </div>
          </div>
        </section>

        <section aria-label="Runs" className="mt-[23px] min-h-[541.5px] overflow-hidden rounded-[14px] border border-border bg-surface">
          <table className="w-full table-fixed border-collapse text-[12.5px]">
            <colgroup>
              {[126, 106, 156, 318, 208, 116, 116, 106, 106].map((w, i) => (
                <col key={i} style={{ width: w }} />
              ))}
            </colgroup>
            <thead className="bg-[#fafaf7]">
              <tr className="h-[39.5px] border-b border-border text-left">
                <th scope="col" className={TH}>Run</th>
                <th scope="col" className={TH}>Ticker</th>
                <th scope="col" className={TH}>{sample ? "Snapshot" : "Memo"}</th>
                <th scope="col" className={TH}>Model</th>
                <th scope="col" className={TH}>Status</th>
                <th scope="col" className={`${TH} text-right`}>Grounded</th>
                <th scope="col" className={`${TH} text-right`}>Coverage</th>
                <th scope="col" className={`${TH} text-right`}>Latency</th>
                <th scope="col" className={`${TH} text-right`}>Cost</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.jobId ?? r.id} className="h-[50.75px] border-b border-[#f0eee8]">
                  <td className="px-6">
                    {r.jobId ? (
                      <Link href={`/workspace?job=${r.jobId}`} className={LINK}>{r.id}</Link>
                    ) : (
                      <span className="font-mono text-[12.4px] text-[#5b6575]">{r.id}</span>
                    )}
                  </td>
                  <td className="px-6 font-semibold text-ink">{r.ticker}</td>
                  <td className="px-6 font-mono text-[12.6px] text-[#5b6575]">
                    {r.jobId ? <Link href={`/memos/${r.jobId}`} className={LINK}>Open memo</Link> : (r.snapshot ?? "—")}
                  </td>
                  <td className="px-6 text-body">{r.model ?? "—"}</td>
                  <td className="px-6">
                    <span
                      className="inline-flex h-5 items-center gap-[6px] rounded px-2 text-[11.3px] font-medium"
                      style={{ background: PILL[r.status].bg, color: PILL[r.status].fg }}
                    >
                      <span aria-hidden className="h-[6px] w-[6px] rounded-full bg-current" />
                      {r.status}
                    </span>
                  </td>
                  <td className="px-6 text-right font-mono text-[#2b3645]">{fmt2(r.grounded)}</td>
                  <td className="px-6 text-right font-mono text-[#2b3645]">{fmt2(r.coverage)}</td>
                  <td className="px-6 text-right font-mono text-[#5b6575]">{r.latencyS === null ? "—" : `${r.latencyS.toFixed(1)}s`}</td>
                  <td className="px-6 text-right font-mono text-[#5b6575]">{r.costUsd === null ? "—" : `$${r.costUsd.toFixed(3)}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length === 0 && <p className="px-6 py-5 text-[13px] text-muted">No runs match this filter.</p>}
        </section>
      </main>
    </>
  );
}
