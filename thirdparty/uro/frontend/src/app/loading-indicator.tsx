"use client";

import {
	useIsFetching,
	useIsMutating,
	useIsRestoring
} from "@tanstack/react-query";

import type { FC } from "react";

export const LoadingIndicator: FC = () => {
	const fetching = useIsFetching();
	const mutating = useIsMutating();
	const restoring = useIsRestoring();

	if (!fetching || !mutating || !restoring) return null;

	return (
		<div className="pointer-events-none fixed inset-x-0 top-0 z-50 h-1">
			<div className="size-full bg-red-500 bg-gradient-to-r" />
		</div>
	);
};
