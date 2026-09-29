"use client";

import { useRouter } from "next/navigation";

const BTN =
  "h-[36.5px] rounded-lg border border-[#dcd9d0] bg-surface px-[13px] text-[13px] text-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-navy";

export function Filters({ status, options }: { status: string; options: readonly string[] }) {
  const router = useRouter();
  return (
    <>
      {/* ponytail: window filter is presentational until the registry endpoint supports date ranges */}
      <button type="button" disabled className={`${BTN} disabled:opacity-100`}>
        Last 7 days
      </button>
      <select
        aria-label="Filter by status"
        value={status}
        onChange={(e) => router.replace(e.target.value ? `/runs?status=${e.target.value}` : "/runs")}
        className={`${BTN} w-[96.5px] font-sans cursor-pointer appearance-none text-center [text-align-last:center]`}
      >
        <option value="">All statuses</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
      {/* ponytail: no compare endpoint yet; link is inert */}
      <a
        href="/runs"
        className="flex h-[36.5px] items-center rounded-lg bg-navy px-[13px] text-[13px] font-medium text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-navy"
      >
        Compare runs
      </a>
    </>
  );
}
