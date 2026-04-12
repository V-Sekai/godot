"use client";

import { useSession } from "../../hooks/session";

import type { PropsWithChildren } from "react";

export default function AuthenticatedLayout({ children }: PropsWithChildren) {
	useSession();
	return children;
}
