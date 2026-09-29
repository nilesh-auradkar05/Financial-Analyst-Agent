import { notFound } from "next/navigation";
import { AppHeader } from "@/components/AppHeader";
import { UpstreamError, getJob } from "@/lib/api.server";
import { isJobId } from "@/lib/validate";
import { sample } from "./_data";
import { toMemoView } from "./_map";
import { MemoScreen } from "./MemoScreen";

export const metadata = { title: "Memo · Alpha Analyst" };

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (id === "sample") return <MemoScreen view={sample} />;
  if (!isJobId(id)) notFound();

  let job;
  try {
    job = await getJob(id);
  } catch (e) {
    if (e instanceof UpstreamError && e.status === 404) notFound();
    return (
      <>
        <AppHeader active="memos" />
        <main className="mx-auto max-w-[720px] px-6 pt-16">
          <h1 className="font-serif text-[32px] text-ink">Memo unavailable</h1>
          <p className="mt-3 text-body">We couldn&rsquo;t load this memo right now. Please try again shortly.</p>
        </main>
      </>
    );
  }
  if (!job.result) {
    return (
      <>
        <AppHeader active="memos" />
        <main className="mx-auto max-w-[720px] px-6 pt-16">
          <h1 className="font-serif text-[32px] text-ink">
            {job.status === "failed" ? "This run did not produce a memo" : "Memo not ready yet"}
          </h1>
          <p className="mt-3 text-body">Status: {job.status}.</p>
        </main>
      </>
    );
  }
  return <MemoScreen view={toMemoView(job.result)} />;
}
