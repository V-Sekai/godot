"use client";

import { redirect, useSearchParams } from "next/navigation";

import { withSuspense } from "~/hooks/with-suspense";

import { useOptionalSession } from "../../hooks/session";

import type { PropsWithChildren } from "react";

function GuestLayout({ children }: PropsWithChildren) {
	const session = useOptionalSession();

	const to = useSearchParams().get("to") || "/";
	if (session) redirect(to);

	return children;
}

export default withSuspense(GuestLayout);
