import { redirect } from "next/navigation";

import { restoreReturnIntent } from "~/hooks/return-intent/common";

import { LoginForm } from "./form";
import { getOAuth2State } from "./state";

export default async function LoginPage({
	searchParams: { state: nonce }
}: {
	searchParams: Partial<Record<string, Array<string> | string>>;
}) {
	if (nonce && typeof nonce !== "string")
		redirect(
			`/login?error=invalid_code&error_description=${encodeURIComponent("Invalid callback request, try again.")}`
		);

	if (nonce) {
		const { ri } = getOAuth2State(nonce) || {};
		restoreReturnIntent(ri || "/");
	}

	return (
		<div className="flex h-full grow items-center justify-center">
			<LoginForm />
		</div>
	);
}
