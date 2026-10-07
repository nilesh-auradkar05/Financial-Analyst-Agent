import Link from "next/link";
import { AppHeader, Avatar } from "@/components/AppHeader";
import { getJob } from "@/lib/api.server";
import { resolveJobId } from "@/lib/jobs.server";
import { groupOf, safeHttpUrl } from "../memos/[id]/_map";
import { GROUP_LABEL } from "../memos/[id]/_data";

export const metadata = { title: "Evidence · Alpha Analyst" };

const TH = "px-5 py-[10px] text-left text-[11px] font-semibold uppercase tracking-[0.075em] text-body";
const CARD = "mt-4 overflow-hidden rounded-[14px] border border-border bg-surface";

function Source({ title, url }: { title: string; url?: string | null }) {
  const href = safeHttpUrl(url);
  return href ? (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-navy underline underline-offset-2">
      {title}
    </a>
  ) : (
    title
  );
}

/** Evidence the writer was given for one run: the citation registry and the news articles. */
export default async function Page({ searchParams }: { searchParams: Promise<{ job?: string | string[] }> }) {
  const jobId = await resolveJobId((await searchParams).job);
  const job = jobId ? await getJob(jobId).catch(() => null) : null;
  const r = job?.result ?? null;

  return (
    <>
      <AppHeader active="evidence">
        <Avatar />
      </AppHeader>
      <main className="mx-auto w-full max-w-[1100px] px-10 pt-[33px] pb-12">
        <h1 className="font-serif text-[33.5px] leading-[1.1] font-medium text-ink">Evidence</h1>
        {!r || !job ? (
          <p className="mt-3 text-[14px] text-body">
            {!jobId
              ? "No runs yet. Start an analysis from the Workspace to see the evidence it gathered."
              : !job
                ? "This run is no longer available from the analysis service."
                : `Evidence is not ready yet · status ${job.status}.`}
          </p>
        ) : (
          <>
            <p className="mt-[8.5px] text-[14px] text-muted">
              {r.company_name || r.ticker} <span className="font-mono text-body">{r.ticker}</span> ·{" "}
              <Link href={`/workspace?job=${job.job_id}`} className="text-navy underline underline-offset-2">
                run {job.job_id.slice(0, 8)}
              </Link>{" "}
              ·{" "}
              <Link href={`/memos/${job.job_id}`} className="text-navy underline underline-offset-2">
                open memo
              </Link>
            </p>
            {r.missing.length > 0 && (
              <p role="status" className="mt-3 text-[14px] font-semibold text-[#8a4b06]">
                Not available for this run: {r.missing.join(", ")}.
              </p>
            )}

            <section aria-label="Cited sources" className={CARD}>
              <table className="w-full border-collapse text-[13px]">
                <caption className="border-b border-border px-5 py-3 text-left text-[14px] font-semibold text-ink">
                  Sources given to the memo writer ({r.citations.length})
                </caption>
                <thead className="bg-[#fafaf7]">
                  <tr>
                    <th scope="col" className={`${TH} w-[70px]`}>Ref</th>
                    <th scope="col" className={`${TH} w-[110px]`}>Type</th>
                    <th scope="col" className={TH}>Source</th>
                    <th scope="col" className={`${TH} w-[250px]`}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {r.citations.map((c) => (
                    <tr key={c.index} className="border-t border-[#f0eee8] align-top">
                      <td className="px-5 py-[10px] font-mono text-navy">[{c.index}]</td>
                      <td className="px-5 py-[10px] text-body">{GROUP_LABEL[groupOf(c.source_type)]}</td>
                      <td className="px-5 py-[10px] text-ink"><Source title={c.title} url={c.url} /></td>
                      <td className="px-5 py-[10px] font-mono text-[12px] text-muted">{c.date ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>

            <section aria-label="News articles" className={CARD}>
              <h2 className="border-b border-border px-5 py-3 text-[14px] font-semibold text-ink">
                News articles ({r.news_articles.length})
                {r.sentiment && (
                  <span className="ml-3 text-[12.5px] font-normal text-muted">
                    FinBERT: {r.sentiment.overall_sentiment} · {r.sentiment.positive_count} positive · {r.sentiment.neutral_count} neutral ·{" "}
                    {r.sentiment.negative_count} negative
                  </span>
                )}
              </h2>
              <ul className="m-0 list-none p-0">
                {r.news_articles.map((a, i) => (
                  <li key={i} className="border-t border-[#f0eee8] px-5 py-3 first:border-t-0">
                    <p className="text-[13.5px] font-medium text-ink"><Source title={a.title} url={a.url} /></p>
                    <p className="mt-[2px] text-[12px] text-muted">
                      {a.source}
                      {a.published_date ? ` · ${a.published_date}` : ""}
                    </p>
                    <p className="mt-1 text-[13px] leading-[20px] text-body">{a.snippet}</p>
                  </li>
                ))}
              </ul>
            </section>
            <p className="mt-4 text-[12.5px] text-muted">
              SEC chunk text, price history and retrieval scores are not returned by the API yet, so they cannot be shown here.
            </p>
          </>
        )}
      </main>
    </>
  );
}
