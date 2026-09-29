import { AppHeader } from "@/components/AppHeader";
import { HeaderActions } from "./_components/HeaderActions";
import { MemoHead } from "./_components/MemoHead";
import { Reader } from "./_components/Reader";
import type { MemoView } from "./_data";

export function MemoScreen({ view }: { view: MemoView }) {
  return (
    <>
      <AppHeader active="memos">
        <HeaderActions />
      </AppHeader>
      <Reader view={view} head={<MemoHead view={view} />} />
    </>
  );
}
