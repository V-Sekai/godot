import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { HydrationBoundary } from "@tanstack/react-query";
import { PublicEnvScript } from 'next-runtime-env';

import { getTheme } from "~/hooks/theme/server";
import { dehydrateAll, getQueryClient } from "~/query";
import { getOptionalSession } from "~/data/session";

import { Body, QueryProvider } from "./body";
import { LoadingIndicator } from "./loading-indicator";

import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
	description: "Your virtual reality platform, on your game engine.",
	title: "V-Sekai"
};

export default async function RootLayout({
	children
}: Readonly<{
	children: React.ReactNode;
}>) {
	const session = await getOptionalSession();

	const queryClient = getQueryClient();

	queryClient.setQueryData(["theme"], getTheme());
	queryClient.setQueryData(["session"], session);

	if (session)
		queryClient.setQueryData(["users", session.user.username], session.user);

	return (
		<html lang="en">
			<head>
				<PublicEnvScript />
			</head>
			<QueryProvider>
				<HydrationBoundary state={dehydrateAll(queryClient)}>
					<Body>
						<ReactQueryDevtools />
						<LoadingIndicator />
						{children}
					</Body>
				</HydrationBoundary>
			</QueryProvider>
		</html>
	);
}
