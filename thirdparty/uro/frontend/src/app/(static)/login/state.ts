import { cookies } from "next/headers";
import { ms } from "@ariesclark/extensions";

import type { ProviderID } from "~/api";

export interface OAuth2State {
	p: ProviderID;
	ri: string;
}

export function setOAuth2State(nonce: string, data: OAuth2State) {
	cookies().set(`oauth2.${nonce}`, JSON.stringify(data), {
		httpOnly: true,
		maxAge: Math.floor(ms("30m") / 1000),
		secure: true
	});
}

export function deleteOAuth2State(nonce: string) {
	cookies().delete(`oauth2.${nonce}`);
}

export function getOAuth2State(nonce: string): OAuth2State | null {
	const data = cookies().get(`oauth2.${nonce}`)?.value;
	if (!data) return null;

	return JSON.parse(data) as OAuth2State;
}
