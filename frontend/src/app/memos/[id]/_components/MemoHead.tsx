import Link from "next/link";
import type { MemoView, Tone } from "../_data";

const TONE: Record<Tone, string> = {
  green: "text-green",
  amber: "text-[#8a4b06]",
  ink: "text-ink",
  navy: "text-navy",
};

/** Breadcrumb row, title, subtitle and the 4-cell stat strip (server component). */
export function MemoHead({ view }: { view: MemoView }) {
  return (
    <header>
      <div className="flex items-center justify-between text-[12px] leading-[20px]">
        <div className="flex items-center">
          <Link
            href={view.runHref}
            className="ml-[2px] flex items-center gap-[7px] font-medium text-navy focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-navy"
          >
            <svg aria-hidden width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
              <path d="M6.5 1.5 3 5l3.5 3.5" />
            </svg>
            {view.runLabel}
          </Link>
          <span aria-hidden className="mx-[11.5px] h-[14px] w-px bg-border" />
          <span className="text-[12px] font-semibold uppercase tracking-[0.09em] text-gold">Investment memo</span>
          {view.draft && (
            <>
              <span aria-hidden className="mx-[11.5px] h-[14px] w-px bg-border" />
              <span className="text-[12px] text-muted">{view.draft}</span>
            </>
          )}
        </div>
        {view.verified && (
          <p className={`flex items-center gap-[6px] font-medium ${view.verified.ok ? "text-green" : "text-[#8a4b06]"}`}>
            <svg aria-hidden width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round">
              <path d="m2.5 7.5 3 3 6-6.5" />
            </svg>
            {view.verified.text}
          </p>
        )}
      </div>

      <h1 className="mt-[28px] font-serif text-[48px] font-normal leading-[56px] tracking-[-0.02em] text-ink">{view.title}</h1>
      {view.subtitle && (
        <p className="mt-[16.5px] font-serif text-[22.5px] italic leading-[30px] text-body">{view.subtitle}</p>
      )}

      <dl className="mt-[29.5px] grid grid-cols-4 overflow-hidden rounded-xl border border-border bg-surface">
        {view.stats.map((s, i) => (
          <div key={s.label} className={`h-[67px] px-[18px] pt-[11.5px] ${i < 3 ? "border-r border-[#efede7]" : ""}`}>
            <dt className="text-[11px] uppercase leading-[16px] tracking-[0.06em] text-muted">{s.label}</dt>
            <dd className={`mt-[1px] text-[15px] leading-[24px] ${s.mono ? "font-mono text-[14px] font-medium" : "font-semibold"} ${TONE[s.tone]}`}>
              {s.value}
            </dd>
          </div>
        ))}
      </dl>
    </header>
  );
}
