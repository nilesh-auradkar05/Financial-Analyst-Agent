"use client";

import { useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { Disclaimer } from "@/components/Disclaimer";
import { isDisclaimer, parseInline } from "../_map";
import { GROUP_LABEL, GROUP_ORDER, type Block, type CitationView, type Group, type MemoView } from "../_data";

const FOCUS = "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-navy";

function chipTone(label: string, selected: boolean): string {
  if (selected) return "bg-navy text-white";
  if (label.startsWith("M")) return "bg-[#e3f2ec] text-[#0b6146]";
  if (label.startsWith("N")) return "bg-gold-tint text-[#7a5418]";
  return "bg-[#e8edf5] text-navy";
}

function Text({
  text,
  byId,
  selectedId,
  onPick,
}: {
  text: string;
  byId: Map<string, CitationView>;
  selectedId: string | null;
  onPick: (id: string) => void;
}) {
  const nodes = useMemo(() => parseInline(text, new Set(byId.keys())), [text, byId]);
  return (
    <>
      {nodes.map((n, i) => {
        if (n.t === "text") return n.text;
        const c = byId.get(n.id)!;
        const sel = n.id === selectedId;
        return (
          <button
            key={i}
            type="button"
            onClick={() => onPick(n.id)}
            aria-pressed={sel}
            aria-label={`Citation ${c.label}: ${c.title}`}
            className={`mx-px inline-block rounded px-[6px] py-[2px] align-[2px] font-mono text-[12px] font-medium leading-[14px] ${chipTone(c.label, sel)} ${FOCUS}`}
          >
            {c.label}
          </button>
        );
      })}
    </>
  );
}

function Blocks({ blocks, ...rest }: { blocks: Block[] } & Omit<Parameters<typeof Text>[0], "text">) {
  return blocks.map((b, i) =>
    b.kind === "p" ? (
      <p
        key={i}
        className={
          isDisclaimer(b.text)
            ? "font-sans text-[16px] font-bold leading-[26px] text-red"
            : "font-serif text-[18.06px] leading-[32px] text-[#1f2a3a]"
        }
      >
        <Text text={b.text} {...rest} />
      </p>
    ) : (
      <ul key={i} className="m-0 list-disc space-y-[6px] pl-[20px] font-serif -mt-px text-[17.3px] leading-[32px] text-[#1f2a3a] marker:text-[#1f2a3a]">
        {b.items.map((it, j) => (
          <li key={j} className="pl-[2px]">
            <Text text={it} {...rest} />
          </li>
        ))}
      </ul>
    ),
  );
}

function Score({ c, size = 12 }: { c: CitationView; size?: 11 | 12 }) {
  const t = size === 11 ? "text-[11px]" : "text-[12px]";
  if (c.score === null) return <span className={`font-mono ${t} text-muted`}>—</span>;
  if (c.score === "exact") return <span className={`font-mono ${t} font-medium text-green`}>exact</span>;
  return (
    <span className={`font-mono ${t} font-medium ${c.score < 0.6 ? "text-[#8a4b06]" : "text-green"}`}>
      {c.score.toFixed(2)}
    </span>
  );
}

function Chip({ c, sel }: { c: CitationView; sel?: boolean }) {
  return (
    <span className={`inline-block rounded px-[7px] py-[2px] font-mono text-[12px] font-semibold leading-[14px] ${chipTone(c.label, !!sel)}`}>
      {c.label}
    </span>
  );
}

export function Reader({ view, head }: { view: MemoView; head: ReactNode }) {
  const byId = useMemo(() => new Map(view.citations.map((c) => [c.id, c])), [view.citations]);
  const [selectedId, setSelectedId] = useState<string | null>(view.initialId);
  const [tab, setTab] = useState<Group>(byId.get(view.initialId ?? "")?.group ?? "sec");
  // The list is unfiltered until a tab is clicked (matches the design's initial state).
  const [filtered, setFiltered] = useState(false);
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});

  const selected = selectedId ? (byId.get(selectedId) ?? null) : null;
  const list = view.citations.filter((c) => c.id !== selectedId && (!filtered || c.group === tab));

  function pick(id: string) {
    const c = byId.get(id);
    if (!c) return;
    setSelectedId(id);
    setTab(c.group);
  }

  function onTabKey(e: KeyboardEvent<HTMLButtonElement>, i: number) {
    const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    const j = e.key === "Home" ? 0 : e.key === "End" ? GROUP_ORDER.length - 1 : d ? (i + d + GROUP_ORDER.length) % GROUP_ORDER.length : -1;
    if (j < 0) return;
    e.preventDefault();
    setTab(GROUP_ORDER[j]);
    setFiltered(true);
    tabRefs.current[GROUP_ORDER[j]]?.focus();
  }

  const textProps = { byId, selectedId, onPick: pick };
  const dash = "—";

  return (
    <div className="flex items-start [text-rendering:geometricPrecision] [&_button]:[text-rendering:geometricPrecision]">
      <main className="min-w-0 flex-1 px-6 pt-[52px] pb-16">
        <article className="mx-auto max-w-[720px]">
          {head}
          {view.sections.map((s, i) => (
            <section key={i} aria-labelledby={s.heading ? `sec-${i}` : undefined} className="mt-[29px] first-of-type:mt-[27px]">
              {s.heading && (
                <h2 id={`sec-${i}`} className="mb-[8.5px] text-[12px] font-semibold uppercase leading-[16px] tracking-[0.11em] text-navy">
                  {s.heading}
                </h2>
              )}
              <div className="space-y-[6px]">
                <Blocks blocks={s.blocks} {...textProps} />
              </div>
            </section>
          ))}
          <Disclaimer className="mt-[32px] border-t-2 border-red pt-[14px] text-[17px] leading-[26px]" />
          {view.callout && (
            <aside
              aria-label="Reviewer note"
              className="mt-[26.5px] flex gap-[14px] rounded-xl border border-[#ead9b6] bg-[#fbf6ec] pl-[20px] pr-[16px] py-[16px]"
            >
              <span aria-hidden className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#f0e2c4] text-[#6b5230]">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12a9 9 0 0 0-15.5-6.2L3 8" />
                  <path d="M3 3v5h5" />
                  <path d="M3 12a9 9 0 0 0 15.5 6.2L21 16" />
                  <path d="M16 16h5v5" />
                </svg>
              </span>
              <div>
                <h2 className="text-[14px] font-semibold leading-[20px] text-[#5c3f12]">{view.callout.title}</h2>
                <p className="mt-[2.5px] text-[13px] leading-[21px] text-[#6b5230]">{view.callout.body}</p>
              </div>
            </aside>
          )}
        </article>
      </main>

      <aside
        aria-label="Evidence"
        className="sticky top-14 h-[calc(100dvh-3.5rem)] w-[457px] shrink-0 self-start overflow-y-auto border-l border-border bg-surface print:hidden"
      >
        <div className="flex h-[61px] items-center justify-between border-b border-border pt-[3px] px-6">
          <h2 className="text-[15px] font-semibold text-ink">Evidence</h2>
          <span className="font-mono text-[11px] text-muted">
            {view.totalCitations} citations · {view.sourceCount} sources
          </span>
        </div>

        <div className="px-6 pt-[15px]">
          <div role="tablist" aria-label="Evidence sources" className="flex gap-[5.5px]">
            {GROUP_ORDER.map((g, i) => {
              const on = g === tab;
              return (
                <button
                  key={g}
                  ref={(el) => {
                    tabRefs.current[g] = el;
                  }}
                  role="tab"
                  type="button"
                  id={`tab-${g}`}
                  aria-selected={on}
                  aria-controls="evidence-panel"
                  tabIndex={on ? 0 : -1}
                  onClick={() => {
                    setTab(g);
                    setFiltered(true);
                  }}
                  onKeyDown={(e) => onTabKey(e, i)}
                  className={`h-8 rounded-lg border px-[12px] text-[12px] ${FOCUS} ${
                    on ? "border-navy bg-navy font-medium text-white" : "border-[#dcd9d0] bg-surface text-body"
                  }`}
                >
                  {GROUP_LABEL[g]} {view.counts[g]}
                </button>
              );
            })}
          </div>

          <div role="tabpanel" id="evidence-panel" aria-labelledby={`tab-${tab}`} tabIndex={-1}>
            {selected ? (
              <div className="mt-[20px] rounded-xl border border-[#c9d3e3] bg-[#f7f9fc] px-[18px] pt-[19px] pb-[14.5px]" aria-live="polite">
                <div className="flex items-center gap-2">
                  <Chip c={selected} sel />
                  <h3 className="text-[13px] font-semibold text-navy">{selected.title}</h3>
                  {selected.meta && <span className="ml-auto text-[12px] text-muted">{selected.meta}</span>}
                </div>
                <blockquote className="mt-[11.5px] font-serif text-[16.5px] leading-[26.5px] text-ink">
                  {selected.quote ? (
                    selected.quote.map((q, i) =>
                      q.hl ? (
                        <mark key={i} className="rounded-[3px] bg-[#fcefc7] px-[3px] text-ink">
                          {q.text}
                        </mark>
                      ) : (
                        q.text
                      ),
                    )
                  ) : (
                    <span className="text-muted">Source text is not exposed by the API.</span>
                  )}
                </blockquote>

                {selected.score !== null && (
                  <div className="mt-[12.5px]">
                    <div className="flex items-baseline justify-between text-[12px] text-body">
                      <span>Grounding · cosine</span>
                      <Score c={selected} />
                    </div>
                    {typeof selected.score === "number" && (
                      <div
                        role="meter"
                        aria-label="Grounding cosine"
                        aria-valuemin={0}
                        aria-valuemax={1}
                        aria-valuenow={selected.score}
                        className="relative mt-[4.5px] h-[5px] rounded-full bg-[#e4e8ef]"
                      >
                        <div className="h-full rounded-full bg-green" style={{ width: `${selected.score * 100}%` }} />
                        {selected.threshold !== null && (
                          <span
                            aria-hidden
                            className="absolute -top-[3px] h-[11px] w-[1.5px] bg-ink"
                            style={{ left: `${selected.threshold * 100}%` }}
                          />
                        )}
                      </div>
                    )}
                    <div className="mt-[6.5px] flex justify-between text-[11px] text-muted">
                      <span>
                        {selected.threshold !== null ? `threshold ${selected.threshold.toFixed(2)}` : "threshold —"} ·{" "}
                        {selected.embedder ?? dash}
                      </span>
                      <span>numbers matched {selected.numbersMatched ?? dash}</span>
                    </div>
                  </div>
                )}

                <dl className="mt-[11.5px] grid grid-cols-[100px_1fr] gap-y-[3px] border-t border-[#e4e8ef] pt-[12px] text-[12px] leading-[18px]">
                  {(
                    [
                      ["chunk_id", selected.chunkId],
                      ["extractor", selected.extractor],
                      ["retrieved", selected.retrieved],
                    ] as const
                  ).map(([k, v]) => (
                    <div key={k} className="contents">
                      <dt className="text-muted">{k}</dt>
                      <dd className="font-mono text-ink">{v ?? dash}</dd>
                    </div>
                  ))}
                </dl>

                {selected.url && selected.linkLabel && (
                  <a
                    href={selected.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`mt-[12px] inline-block text-[13px] leading-[20px] font-semibold text-navy ${FOCUS}`}
                  >
                    {selected.linkLabel} →
                  </a>
                )}
              </div>
            ) : (
              <p className="mt-[20px] text-[14px] text-muted">No citations available.</p>
            )}

            <h3 className="mt-[19.5px] mb-[7px] text-[11px] font-semibold uppercase leading-[16px] tracking-[0.08em] text-body">
              All citations
            </h3>
            {list.length === 0 && <p className="py-3 text-[13px] text-muted">No other citations in this group.</p>}
            <ul className="pb-6">
              {list.map((c) => (
                <li key={c.id} className="border-b border-[#efede7]">
                  <button
                    type="button"
                    onClick={() => pick(c.id)}
                    className={`grid h-[39px] w-full grid-cols-[42px_1fr_auto] items-center text-left ${FOCUS}`}
                  >
                    <span className="font-mono text-[11px] font-medium text-navy">{c.label}</span>
                    <span className="truncate text-[13px] text-[#2b3645]">{c.title}</span>
                    <Score c={c} size={11} />
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </aside>
      <style>{"@media print{header{display:none!important}}"}</style>
    </div>
  );
}
