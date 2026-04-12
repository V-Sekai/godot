"use client";

import { VSekaiMark } from "~/components/vsekai-mark";
import { useReturnIntent } from "~/hooks/return-intent";
import { restoreReturnIntent } from "~/hooks/return-intent/common";
import { useSession } from "~/hooks/session";

import { ResendButton } from "./resend";
import { ChangeEmailButton } from "./change-email";

export default function ConfirmEmailPage() {
	const session = useSession();
	const { returnIntent } = useReturnIntent();

	if (session.user.email_confirmed_at)
		restoreReturnIntent(returnIntent?.href || "/");

	return (
		<div className="flex h-full grow items-center justify-center">
			<div className="flex w-[32rem] flex-col gap-4 overflow-hidden rounded-xl border border-tertiary-300 bg-tertiary-50 p-8">
				<span className="text-xl">
					<VSekaiMark className="inline size-5" /> Confirm your Email Address
				</span>
				<p className="opacity-75">
					We&apos;ve sent you an email to{" "}
					<span className="font-medium">{session.user.email}</span> with a link
					to confirm your account. Please check your inbox and click the link to
					continue.
				</p>
				<div className="flex gap-2">
					<ResendButton />
					<ChangeEmailButton />
				</div>
			</div>
		</div>
	);
}
