/** Navy rounded tile with "α" plus the "Alpha Analyst" wordmark. */
export function Logo() {
  return (
    <span className="flex items-center gap-2.5">
      <span className="flex h-7 w-7 items-center justify-center rounded-[7px] bg-navy font-serif text-[17px] leading-none text-[#e8d5a8]">
        α
      </span>
      <span className="text-[14.3px] font-semibold text-ink">Alpha Analyst</span>
    </span>
  );
}
