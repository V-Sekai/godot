"use client";

import { VSekaiMark } from "~/components/vsekai-mark";
import { useReturnIntent } from "~/hooks/return-intent";
import { restoreReturnIntent } from "~/hooks/return-intent/common";
import { useAdminStatus } from "./data";

export default function AdminStatusPage() {
	const adminStatus = useAdminStatus();
	const { returnIntent } = useReturnIntent();

	if (adminStatus?.status?.is_admin == 'false')
		restoreReturnIntent(returnIntent?.href || "/");

	return (
		<div className="flex h-full grow items-center justify-center">
			<div className="flex w-[32rem] flex-col gap-4 overflow-hidden rounded-xl border border-tertiary-300 bg-tertiary-50 p-8">
				<span className="text-xl">
					<VSekaiMark className="inline size-5" /> Admin Access Status
				</span>
				<p className="opacity-75">
					Status: {" "}
					<span className="font-medium">{adminStatus?.status?.is_admin}</span>
					You have admin panel access.
				</p>
				<div className="flex gap-2">
				</div>
			</div>
		</div>
	);
}
