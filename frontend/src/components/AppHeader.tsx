import Link from "next/link";
import { recentJobIds } from "@/lib/jobs.server";
import { Logo } from "./Logo";

const NAV = [
  { key: "workspace", label: "Workspace", href: "/workspace" },
  { key: "memos", label: "Memos", href: "/memos/sample" },
  { key: "runs", label: "Runs", href: "/runs" },
  { key: "evidence", label: "Evidence", href: "/evidence" },
] as const;

export type NavKey = (typeof NAV)[number]["key"];

/** App nav (56px incl. bottom border). `children` fills the right-hand slot. */
export async function AppHeader({ active, children }: { active?: NavKey; children?: React.ReactNode }) {
  // Memos opens this browser's latest run; the sample only when there is none.
  const latest = (await recentJobIds())[0];
  const hrefOf = (n: (typeof NAV)[number]) => (n.key === "memos" && latest ? `/memos/${latest}` : n.href);
  return (
    <header className="sticky top-0 z-30 flex h-14 items-stretch border-b border-border bg-surface px-6">
      <Link href="/" className="mr-[27.5px] flex items-center" aria-label="Alpha Analyst home">
        <Logo />
      </Link>
      <nav className="flex items-stretch gap-1" aria-label="Primary">
        {NAV.map((n) => (
          <Link
            key={n.key}
            href={hrefOf(n)}
            aria-current={n.key === active ? "page" : undefined}
            className={`relative flex items-center px-3 text-[14px] ${
              n.key === active
                ? "font-medium text-ink after:absolute after:inset-x-0 after:-bottom-px after:h-[2px] after:bg-navy"
                : "text-body"
            }`}
          >
            {n.label}
          </Link>
        ))}
      </nav>
      <div className="ml-auto flex items-center gap-3">
        {children}
      </div>
    </header>
  );
}

/** "NA" user avatar; place last in the header slot on screens that show it (not the memo reader). */
export function Avatar() {
  return (
    <span className="flex h-[30px] w-[30px] items-center justify-center rounded-full border border-[#e8dcc0] bg-gold-tint text-[12px] font-semibold text-gold">
      NA
    </span>
  );
}
