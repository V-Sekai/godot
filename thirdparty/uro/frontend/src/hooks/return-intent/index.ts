import { useMemo } from "react";
import { useRouter } from "next/navigation";

import { getFirstPartyOrigins, getServerEnv } from "~/environment";

import { useLocation } from "../location";

function minimizeHref(href: URL | string) {
	const origin = getServerEnv()?.origin || "";
	const url = new URL(href.toString(), origin);
	return url.origin === origin ? url.href.replace(origin, "") : url.href;
}

export function useReturnIntent() {
	const router = useRouter();

	const { current, searchParams } = useLocation();
	const _returnIntent = searchParams.get("ri");

	return useMemo(() => {
		const origin = getServerEnv()?.origin || "";
		const firstPartyOrigins = getFirstPartyOrigins();

		let returnIntent = _returnIntent ? new URL(_returnIntent, origin) : null;
		if (returnIntent && !firstPartyOrigins.has(returnIntent.origin))
			returnIntent = null;

		return {
			restoreReturnIntent: (fallback: string = "/") =>
				router.push(returnIntent?.toString() || fallback),
			returnIntent,
			withReturnIntent: (pathname: string) => {
				const url = new URL(pathname, origin);
				url.searchParams.set(
					"ri",
					minimizeHref(returnIntent ? returnIntent.href : current)
				);

				return url;
			}
		};
	}, [router, _returnIntent, current]);
}
