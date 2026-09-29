import { AppHeader, Avatar } from "@/components/AppHeader";
import { isJobId } from "@/lib/validate";
import { FIXTURE_RUN } from "./_data";
import { LiveWorkspace } from "./_components/LiveWorkspace";
import { SearchNew } from "./_components/SearchNew";
import { WorkspaceView } from "./_components/WorkspaceView";

export const metadata = { title: "Workspace · Alpha Analyst" };

export default async function Page({ searchParams }: { searchParams: Promise<{ [key: string]: string | string[] | undefined }> }) {
  const job = (await searchParams).job;
  const jobId = typeof job === "string" && isJobId(job) ? job : null;
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
