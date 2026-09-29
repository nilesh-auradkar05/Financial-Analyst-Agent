"use client";

import { useEffect, useId, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { isTicker, normalizeTicker } from "@/lib/validate";
import type { JobAcceptedResponse } from "@/lib/api-types";

/** Header search + "New analysis": validates a ticker, POSTs /api/analyze, opens /workspace?job=<id>. */
export function SearchNew() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const errId = useId();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        inputRef.current?.focus();
        inputRef.current?.select();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  async function submit(e?: React.FormEvent) {
    e?.preventDefault();
    if (busy) return;
    if (!value.trim()) {
      inputRef.current?.focus();
      return;
    }
    if (!isTicker(value)) {
      setError("Enter a valid ticker, e.g. NWSC or BRK.B.");
      return;
    }
    setError(null);
    setBusy(true);
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ticker: normalizeTicker(value) }),
      });
      if (res.status === 400) {
        setError("That ticker was rejected as invalid.");
      } else if (!res.ok) {
        setError("Analysis service is unavailable. Try again shortly.");
      } else {
        const job = (await res.json()) as JobAcceptedResponse;
        router.push(`/workspace?job=${encodeURIComponent(job.job_id)}`);
      }
    } catch {
      setError("Analysis service is unavailable. Try again shortly.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form role="search" onSubmit={submit} className="relative flex items-center gap-[28px]" noValidate>
      <label className="flex h-9 w-[320px] items-center gap-[9.5px] rounded-[9px] border border-border bg-[#fbfaf7] px-3 focus-within:outline-2 focus-within:outline-offset-1 focus-within:outline-navy">
        <span className="sr-only">Analyze a ticker</span>
        <svg aria-hidden width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="#6b7280" strokeWidth="1.3" strokeLinecap="round">
          <circle cx="7" cy="7" r="4.6" />
          <path d="M10.5 10.5L14 14" />
        </svg>
        <input
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Analyze a ticker…"
          autoComplete="off"
          spellCheck={false}
          aria-invalid={error ? true : undefined}
          aria-describedby={error ? errId : undefined}
          className="min-w-0 flex-1 bg-transparent text-[13px] text-ink outline-none placeholder:text-muted"
        />
        <kbd className="rounded-[5px] border border-border bg-surface px-1.5 py-[3px] font-mono text-[10px] text-muted">⌘K</kbd>
      </label>
      <button
        type="submit"
        disabled={busy}
        className="flex h-9 w-[129px] items-center justify-center gap-2 rounded-[8px] bg-navy text-[13px] font-medium text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-navy disabled:opacity-70"
      >
        <svg aria-hidden width="12" height="12" viewBox="0 0 14 14" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round">
          <path d="M7 1.5v11M1.5 7h11" />
        </svg>
        New analysis
      </button>
      {error && (
        <p
          id={errId}
          role="alert"
          className="absolute right-0 top-full z-40 mt-2 w-[320px] rounded-md border border-[#f0c9c4] bg-surface px-3 py-2 text-[12.5px] text-red shadow-sm"
        >
          {error}
        </p>
      )}
    </form>
  );
}
