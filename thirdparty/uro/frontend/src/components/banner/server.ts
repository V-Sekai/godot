import "server-only";

import { cookies } from "next/headers";

import { unknownBannerState, type BannerState } from ".";

export function getBannerState(id?: string): BannerState {
	if (!id) return unknownBannerState;

	const cookie = cookies().get(`banner-${id}`)?.value;
	return {
		...((cookie ? JSON.parse(cookie) : unknownBannerState) as BannerState),
		id
	};
}
