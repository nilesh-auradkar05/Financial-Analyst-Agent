"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { isTerminalJobStatus, type JobPollResponse } from "@/lib/api-types";
import { toWorkspaceRun } from "../_data";
import { WorkspaceView } from "./WorkspaceView";

const POLL_MS = 2000;

/** Polls /api/jobs/{id} every 2s until the job is terminal, rendering real fields only. */
export function LiveWorkspace({ jobId }: { jobId: string }) {
  const [job, setJob] = useState<JobPollResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [gone, setGone] = useState(false);

  useEffect(() => {
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      let done = false;
      try {
        const res = await fetch(`/api/jobs/${jobId}`, { signal: ctrl.signal, cache: "no-store" });
        if (res.ok) {
          const j = (await res.json()) as JobPollResponse;
          setJob(j);
          setError(null);
          done = isTerminalJobStatus(j.status);
        } else if (res.status === 400 || res.status === 404) {
          setGone(true);
          done = true;
        } else {
          setError("Job service unavailable; retrying…");
        }
      } catch {
        if (ctrl.signal.aborted) return;
        setError("Job service unavailable; retrying…");
      }
      if (!done && !ctrl.signal.aborted) timer = setTimeout(tick, POLL_MS);
    };
    void tick();
    return () => {
      ctrl.abort();
      clearTimeout(timer);
    };
  }, [jobId]);

  if (gone) {
    return (
      <main className="mx-auto max-w-[720px] px-6 pt-16">
        <h1 className="font-serif text-[32px] text-ink">This run is no longer available</h1>
        <p className="mt-3 text-body">
          The analysis service has no record of it, most likely because its data was reset.{" "}
          <Link href="/workspace" className="font-medium text-navy underline underline-offset-2">
            Open the latest available run
          </Link>{" "}
          or start a new analysis above.
        </p>
      </main>
    );
  }

  const run = job
    ? toWorkspaceRun(job)
    : {
        ...toWorkspaceRun({ job_id: jobId, ticker: "—", status: "pending", started_at: "" }),
        company: "Loading run…",
      };
  return (
    <>
      {error && (
        <p role="alert" className="mx-auto w-[1440px] max-w-full px-6 pt-3 text-[13px] text-red">
          {error}
        </p>
      )}
      <WorkspaceView run={run} />
    </>
  );
}
