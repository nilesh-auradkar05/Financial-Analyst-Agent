"use client";

import { useEffect, useId, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { isTicker, normalizeTicker } from "@/lib/validate";
import type { JobAcceptedResponse } from "@/lib/api-types";

const TURNSTILE_SCRIPT = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";

type TurnstileApi = {
  render(container: HTMLElement, options: { sitekey: string; appearance: "interaction-only"; callback(token: string): void; "error-callback"(): void }): string;
  remove(widgetId: string): void;
};

/** Resolves with Cloudflare's Turnstile API; the script is added from our own nonced bundle, which `strict-dynamic` trusts. */
function loadTurnstile(): Promise<TurnstileApi> {
  const ready = () => (window as unknown as { turnstile?: TurnstileApi }).turnstile;
  return new Promise((resolve, reject) => {
    const api = ready();
    if (api) return resolve(api);
    const script = document.createElement("script");
    script.src = TURNSTILE_SCRIPT;
    script.async = true;
    script.onload = () => {
      const loaded = ready();
      return loaded ? resolve(loaded) : reject(new Error("turnstile unavailable"));
    };
    script.onerror = () => reject(new Error("turnstile unavailable"));
    document.head.appendChild(script);
  });
}

/**
 * Header search + "New analysis": validates a ticker, POSTs /api/analyze, opens /workspace?job=<id>.
 * A visitor without a session (401) first passes a Turnstile challenge, which `/api/session` turns into the session cookie.
 */
export function SearchNew({ siteKey }: { siteKey: string | undefined }) {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [challenge, setChallenge] = useState(false);
  const challengeRef = useRef<HTMLDivElement>(null);
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

  useEffect(() => {
    const container = challengeRef.current;
    if (!challenge || !container || !siteKey) return;
    let widgetId: string | undefined;
    let cancelled = false;
    const fail = () => {
      setChallenge(false);
      setBusy(false);
      setError("Could not verify your browser. Try again.");
    };
    loadTurnstile()
      .then((turnstile) => {
        if (cancelled) return;
        widgetId = turnstile.render(container, {
          sitekey: siteKey,
          appearance: "interaction-only",
          callback: (token) => {
            fetch("/api/session", {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify({ turnstile_token: token }),
            })
              .then((res) => {
                if (!res.ok) return fail();
                setChallenge(false);
                void submit();
              })
              .catch(fail);
          },
          "error-callback": fail,
        });
      })
      .catch(fail);
    return () => {
      cancelled = true;
      if (widgetId) (window as unknown as { turnstile: TurnstileApi }).turnstile.remove(widgetId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- submit is re-created each render; only a new challenge should re-run this.
  }, [challenge, siteKey]);

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
      if (res.status === 401 && siteKey && !challenge) {
        setChallenge(true);
        return;
      }
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
      {challenge && <div ref={challengeRef} className="absolute right-0 top-full z-40 mt-2" />}
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
