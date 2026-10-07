/** Fixed, deliberately loud notice shown wherever memo content appears. */
export function Disclaimer({ className = "" }: { className?: string }) {
  return (
    <p role="note" className={`font-sans font-bold text-red ${className}`}>
      Disclaimer: AI-generated analysis for informational purposes only. This is not investment advice. Verify every
      claim against the cited sources before making any investment decision.
    </p>
  );
}
