"use client";

import { useState } from "react";

const BTN =
  "inline-flex [text-rendering:geometricPrecision] h-9 items-center rounded-lg border px-[14px] text-[13px] font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-navy";
const GHOST = `${BTN} border-[#dcd9d0] bg-surface text-ink enabled:hover:bg-page`;

/** Right-hand header slot: Replay snapshot / Export PDF / Share memo. */
export function HeaderActions() {
  const [status, setStatus] = useState("");

  async function share() {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setStatus("Link copied to clipboard");
    } catch {
      setStatus("Could not copy the link");
    }
    setTimeout(() => setStatus(""), 3000);
  }

  return (
    <div className="flex items-center gap-[28.5px] print:hidden">
      {/* ponytail: replay needs a GET /jobs/{id}/snapshot/replay endpoint the API does not expose yet. */}
      <button
        type="button"
        disabled
        title="Replay isn't exposed by the API yet"
        className={`${GHOST} disabled:cursor-not-allowed`}
      >
        Replay snapshot
      </button>
      <button type="button" onClick={() => window.print()} className={GHOST}>
        Export PDF
      </button>
      <button
        type="button"
        onClick={share}
        className={`${BTN} border-navy bg-navy text-white hover:bg-[#1b3d70]`}
      >
        Share memo
      </button>
      <span role="status" aria-live="polite" className="sr-only">
        {status}
      </span>
    </div>
  );
}
