import { AppHeader, Avatar } from "@/components/AppHeader";
import { resolveJobId } from "@/lib/jobs.server";
import { FIXTURE_RUN } from "./_data";
import { LiveWorkspace } from "./_components/LiveWorkspace";
import { SearchNew } from "./_components/SearchNew";
import { WorkspaceView } from "./_components/WorkspaceView";

export const metadata = { title: "Workspace · Alpha Analyst" };

export default async function Page({ searchParams }: { searchParams: Promise<{ [key: string]: string | string[] | undefined }> }) {
  // No ?job= opens this browser's latest run; the sample shows only when there is none.
  const jobId = await resolveJobId((await searchParams).job);
  return (
    <>
      <AppHeader active="workspace">
        <div className="flex items-center gap-[30px]">
          <SearchNew />
          <Avatar />
        </div>
      </AppHeader>
      {jobId ? <LiveWorkspace key={jobId} jobId={jobId} /> : <WorkspaceView run={FIXTURE_RUN} />}
    </>
  );
}
