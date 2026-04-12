import { twMerge } from "tailwind-merge";

import type { FC } from "react";
import type { api } from "~/api";

const statusMetadata: Record<
	api.UserStatus,
	{ message: string; className: string }
> = {
	away: {
		className: "bg-yellow-500",
		message: "Away"
	},
	busy: {
		className: "bg-red-500",
		message: "Busy"
	},
	invisible: {
		className: "bg-gray-500",
		message: "Offline"
	},
	offline: {
		className: "bg-gray-500",
		message: "Offline"
	},
	online: {
		className: "bg-green-500",
		message: "Online"
	}
};

export const StatusBadge: FC<{
	user: Pick<api.User, "status" | "status_message">;
}> = ({ user: { status, status_message } }) => {
	if (!status || ["offline", "invisible"].includes(status)) return null;
	const { message: defaultStatusMessage, className: statusClassName } =
		statusMetadata[status];

	return (
		<div className="flex w-28 max-w-fit items-center gap-2 overflow-hidden rounded-xl border border-tertiary-300 bg-tertiary-50 pr-3 transition-all">
			<div
				className={twMerge(
					"aspect-square size-6 shrink-0 rounded-full",
					statusClassName
				)}
			/>
			<span className="whitespace-nowrap text-sm">
				{status_message || defaultStatusMessage}
			</span>
		</div>
	);
};
