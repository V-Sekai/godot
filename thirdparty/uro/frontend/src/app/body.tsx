"use client";

import { Prompt } from "next/font/google";
import { twMerge } from "tailwind-merge";
import { QueryClientProvider } from "@tanstack/react-query";

import { ThemeOverride } from "~/hooks/theme";
import { getQueryClient } from "~/query";
import { EventProvider } from "~/hooks/event";

import type { FC, PropsWithChildren } from "react";

const prompt = Prompt({
	subsets: ["latin"],
	weight: ["400", "500"]
});

export const Body: FC<PropsWithChildren> = ({ children }) => {
	return (
		<EventProvider>
			<ThemeOverride>
				<body
					className={twMerge(
						"flex min-h-svh flex-col bg-tertiary-100 font-normal",
						prompt.className
					)}
				>
					{children}
				</body>
			</ThemeOverride>
		</EventProvider>
	);
};

export const QueryProvider: FC<PropsWithChildren> = ({ children }) => {
	return (
		<QueryClientProvider client={getQueryClient()}>
			{children}
		</QueryClientProvider>
	);
};
