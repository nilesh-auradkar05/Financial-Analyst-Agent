import type { GraphNodeData, NodeStatus } from "../_data";

const DOT: Record<NodeStatus, string> = {
  completed: "bg-green",
  running: "bg-[#2952cc]",
  degraded: "bg-amber",
  queued: "border border-[#9ca3af] bg-transparent",
};
const BOX: Record<NodeStatus, string> = {
  completed: "border border-border bg-surface",
  queued: "border border-dashed border-[#c9c7bf] bg-[#fbfaf7]",
  degraded: "border border-[#f0dcae] bg-amber-tint",
  running: "border-[1.5px] border-[#2952cc] bg-surface shadow-[0_0_0_4px_rgba(42,79,199,0.14)]",
};
const TEXT: Record<NodeStatus, string> = { completed: "text-muted", queued: "text-muted", degraded: "text-[#8a4b06]", running: "text-[#2952cc]" };

function Node({ n }: { n: GraphNodeData }) {
  return (
    <div role="group" aria-label={`${n.title}, ${n.status}`} className={`w-[200px] rounded-lg px-[12.5px] py-[9px] ${BOX[n.status]}`}>
      <div className="flex items-center gap-[7px]">
        <span aria-hidden className={`size-[7px] shrink-0 rounded-full ${DOT[n.status]}`} />
        <span className="text-[13px] font-semibold leading-[19px] tracking-[-0.03em] text-ink">{n.title}</span>
        {n.badge && <span className={`ml-auto font-mono text-[10px] ${TEXT[n.status]}`}>{n.badge}</span>}
      </div>
      <p className={`mt-[3px] font-mono text-[10.5px] leading-[13.5px] tracking-[0.3px] ${TEXT[n.status]}`}>{n.meta}</p>
    </div>
  );
}

const Arrow = () => <span aria-hidden className="text-[14px] leading-none text-[#9ca3af]">↓</span>;

/** The workflow as built: three evidence nodes in parallel, then sentiment → memo writer → verifier. */
export function AgentGraph({ nodes }: { nodes: GraphNodeData[] }) {
  return (
    <figure
      aria-label="Agent graph: news, market data and SEC filings run in parallel, then sentiment, memo writer and verifier."
      className="m-0 flex flex-col items-center gap-[10px] bg-[#fbfaf7] px-5 py-7 [background-image:radial-gradient(circle_at_center,#e6e4dd_0.9px,transparent_1.6px)] [background-size:20px_20px]"
    >
      <div className="flex flex-wrap justify-center gap-4">
        {nodes.slice(0, 3).map((n) => (
          <Node key={n.id} n={n} />
        ))}
      </div>
      {nodes.slice(3).map((n) => (
        <div key={n.id} className="flex flex-col items-center gap-[10px]">
          <Arrow />
          <Node n={n} />
        </div>
      ))}
      <figcaption className="sr-only">Agent graph: {nodes.map((n) => `${n.title} ${n.status}`).join(", ")}</figcaption>
    </figure>
  );
}
