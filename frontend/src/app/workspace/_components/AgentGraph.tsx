import type { GraphNodeData, NodeStatus } from "../_data";

/** Canvas is 990 x 512 CSS px; nodes are absolutely positioned (left, top). */
const POS: Record<GraphNodeData["id"], { x: number; y: number; w: number }> = {
  orchestrator: { x: 406.75, y: 9, w: 176.5 },
  market: { x: 65.25, y: 118.5, w: 176.5 },
  sec: { x: 293.25, y: 118.5, w: 176.5 },
  news: { x: 521.25, y: 118.5, w: 176.5 },
  sentiment: { x: 749.25, y: 118.5, w: 176.5 },
  snapshot: { x: 383, y: 229, w: 224 },
  writer: { x: 406.75, y: 338.5, w: 176.5 },
  verifier: { x: 406.75, y: 449, w: 176.5 },
  publish: { x: 749.25, y: 449, w: 176.5 },
};

const DOT: Record<NodeStatus, string> = {
  completed: "bg-green",
  running: "bg-[#2952cc]",
  degraded: "bg-amber",
  queued: "border border-[#9ca3af] bg-transparent",
};

function Dot({ status, sealed }: { status: NodeStatus; sealed?: boolean }) {
  return (
    <span
      aria-hidden
      className={`size-[7px] shrink-0 rounded-full ${sealed ? "bg-[#6ee7b7]" : DOT[status]}`}
    />
  );
}

const STATUS_LABEL: Record<NodeStatus, string> = {
  completed: "completed",
  running: "running",
  degraded: "degraded",
  queued: "queued",
};

function Node({ n }: { n: GraphNodeData }) {
  const p = POS[n.id];
  const sealed = n.variant === "sealed";
  const gate = n.variant === "gate";
  const box = sealed
    ? "border border-navy bg-navy text-white shadow-[0_1px_2px_rgba(15,26,42,0.12)]"
    : n.status === "degraded"
      ? "border border-[#f0dcae] bg-amber-tint shadow-[0_1px_2px_rgba(15,26,42,0.04)]"
      : n.status === "running"
        ? "border-[1.5px] border-[#2952cc] bg-surface shadow-[0_0_0_4px_rgba(42,79,199,0.14)]"
        : gate
          ? "border border-dashed border-[#c9c7bf] bg-[#fbfaf7]"
          : "border border-border bg-surface shadow-[0_1px_2px_rgba(15,26,42,0.04)]";
  const titleCls = sealed ? "text-white" : gate ? "text-body" : "text-ink";
  const badgeCls = sealed
    ? "text-[#e8d5a8]"
    : n.status === "degraded"
      ? "text-[#8a4b06]"
      : n.status === "running"
        ? "text-[#2952cc]"
        : "text-muted";
  const metaCls = sealed ? "text-[#c3cee3]" : n.status === "degraded" ? "text-[#8a4b06]" : "text-muted";
  return (
    <div
      role="group"
      aria-label={`${n.title}, ${STATUS_LABEL[n.status]}`}
      className={`absolute flex h-[55.5px] flex-col justify-center rounded-lg px-[12.5px] ${box}`}
      style={{ left: p.x, top: p.y, width: p.w }}
    >
      <div className="flex items-center gap-[7px]">
        <Dot status={n.status} sealed={sealed} />
        <span className={`text-[13px] font-semibold leading-[19px] tracking-[-0.03em] ${titleCls}`}>{n.title}</span>
        {n.badge && (
          <span className={`ml-auto font-mono text-[10px] ${badgeCls}`}>{n.badge}</span>
        )}
      </div>
      {n.variant === "progress" ? (
        <div className="mt-[3px] h-[4px] rounded-full bg-[#e7edfc]" role="progressbar" aria-label="Claims checked" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round((n.progress ?? 0) * 100)}>
          <div className="h-full rounded-full bg-[#2952cc]" style={{ width: `${(n.progress ?? 0) * 100}%` }} />
        </div>
      ) : (
        <p className={`mt-[3px] font-mono text-[10.5px] leading-[13.5px] tracking-[0.3px] ${metaCls} ${sealed ? "" : ""}`}>{n.meta}</p>
      )}
    </div>
  );
}

const G = "#c4c8cf";
function Arrow({ x, y, color = G }: { x: number; y: number; color?: string }) {
  return <polygon points={`${x - 5},${y - 11} ${x + 5},${y - 11} ${x},${y}`} fill={color} />;
}

export function AgentGraph({
  nodes,
  reviseLabel,
  gateLabel,
}: {
  nodes: GraphNodeData[];
  reviseLabel: string;
  gateLabel: string;
}) {
  return (
    <figure
      aria-label="Agent graph: orchestrator fans out to four evidence agents, which feed a sealed evidence snapshot, then the memo writer and verifier with a revise loop, then publish."
      className="relative m-0 h-[511px] overflow-hidden bg-[#fbfaf7] [background-image:radial-gradient(circle_at_center,#e6e4dd_0.9px,transparent_1.6px)] [background-position:-0.5px_-7.5px] [background-size:20px_20px]"
    >
      <svg
        aria-hidden
        className="absolute inset-0"
        width="990"
        height="512"
        viewBox="10 108 1980 1024"
        fill="none"
      >
        <g stroke={G} strokeWidth="3">
          {[316, 773, 1229, 1685].map((x) => (
            <path key={`o${x}`} d={`M1000 238 C1000 ${x === 316 || x === 1685 ? 296 : 302}, ${x} 262, ${x} 336`} />
          ))}
          {[316, 773, 1229, 1685].map((x) => (
            <path key={`m${x}`} d={`M${x} 456 C${x} 540, 1000 470, 1000 558`} />
          ))}
          <path d="M1000 677 V775" />
        </g>
        {[316, 773, 1229, 1685].map((x) => (
          <Arrow key={`a${x}`} x={x} y={345} />
        ))}
        <Arrow x={1000} y={566} />
        <Arrow x={1000} y={785} />
        <path d="M1000 896 V992" stroke="#2952cc" strokeWidth="3" />
        <Arrow x={1000} y={1006} color="#2952cc" />
        <path d="M822 1060 C686 1060, 686 840, 812 840" stroke="#9a6b1f" strokeWidth="3" strokeDasharray="10 8" />
        <polygon points="810,833 810,847 824,840" fill="#9a6b1f" />
        <path d="M1183 1061 H1508" stroke="#c9c7bf" strokeWidth="3" strokeDasharray="1 7" strokeLinecap="round" />
      </svg>
      <span className="absolute -translate-y-1/2 whitespace-pre font-mono text-[10px] text-gold" style={{ left: 291.5, top: 421.5 }}>
        {reviseLabel}
      </span>
      <span className="absolute -translate-y-1/2 font-mono text-[10px] text-muted" style={{ left: 617.5, top: 466 }}>
        {gateLabel}
      </span>
      {nodes.map((n) => (
        <Node key={n.id} n={n} />
      ))}
      <figcaption className="sr-only">Agent graph: {nodes.map((n) => `${n.title} ${STATUS_LABEL[n.status]}`).join(", ")}</figcaption>
    </figure>
  );
}
