import { decode } from "jsonwebtoken";

import { api } from "~/api";
import { VSekaiMark } from "~/components/vsekai-mark";

export default async function ConfirmEmailPage({
	params: { token }
}: {
	params: { token: string };
}) {
	const user_id = decode(token, { json: true })?.sub;

	const {
		response: { status }
	} = user_id
		? await api.confirmEmail({
				body: { token: token },
				path: { user_id }
			})
		: { response: { status: 400 } };

	return (
		<div className="flex h-full grow items-center justify-center">
			<div className="flex w-[32rem] flex-col gap-4 overflow-hidden rounded-xl border border-tertiary-300 bg-tertiary-50 p-8">
				{status === 200 ? (
					<>
						<span className="text-xl">
							<VSekaiMark className="inline size-5" /> Email successfully
							confirmed
						</span>
						<p className="opacity-75">
							You can now close this tab, or continue using the application.
						</p>
					</>
				) : (
					<>
						<span className="text-xl">
							<VSekaiMark className="inline size-5" /> Email confirmation failed
						</span>
						<p className="opacity-75">
							We were unable to confirm your email address because this link is
							either invalid or has already expired. Please try again or request
							a new confirmation email.
						</p>
					</>
				)}
			</div>
		</div>
	);
}
