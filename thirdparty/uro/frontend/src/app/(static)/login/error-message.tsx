import { TriangleAlert } from "lucide-react";
import { twMerge } from "tailwind-merge";
import { capitalize, trim } from "@ariesclark/extensions";

import type { FC } from "react";

function normalizeMessage(value: string) {
	return capitalize(trim(value, "[.!?]"));
}

export const ErrorMessage: FC<{
	message?: string | null;
	className?: string;
}> = ({ message, className }) => {
	return (
		<div
			className={twMerge(
				"max-h-48 overflow-hidden overflow-y-auto whitespace-break-spaces rounded-md border border-red-500/5 bg-red-500/5 px-4 text-sm text-red-500 transition-all",
				className,
				message ? "py-2 opacity-100" : "h-0 py-0 opacity-0"
			)}
		>
			<TriangleAlert className="mr-2 inline-block size-4 shrink-0" />
			{message && normalizeMessage(message)}
		</div>
	);
};
