"use server";

import { redirect } from "next/navigation";

import {
	loginWithProvider as _loginWithProvider,
	type ProviderID
} from "~/api";

import { setOAuth2State, type OAuth2State } from "./state";

export async function loginWithProvider(
	providerId: ProviderID,
	state: Omit<OAuth2State, "p">
) {
	const { data } = await _loginWithProvider({
		path: { provider: providerId }
	});

	if (!data) return;

	setOAuth2State(data.state, { p: providerId, ...state });
	redirect(data.url);
}
