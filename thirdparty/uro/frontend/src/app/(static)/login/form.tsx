"use client";

import { useQueryClient } from "@tanstack/react-query";

import { type LoginCredentials, api } from "~/api";
import { useLocation } from "~/hooks/location";
import { Input } from "~/components/input";
import { HorizontalSeparator } from "~/components/horizontal-separator";
import { useReturnIntent } from "~/hooks/return-intent";
import { InlineLink } from "~/components/link";
import { VSekaiMark } from "~/components/vsekai-mark";
import { FormButton, FormErrorMessage, MutationForm } from "~/hooks/form";

import { OAuth2ButtonGroup } from "./oauth2-button";

import type { FC } from "react";

export const LoginForm: FC = () => {
	const { searchParams } = useLocation();
	const { withReturnIntent, restoreReturnIntent } = useReturnIntent();
	const queryClient = useQueryClient();

	const email = searchParams.get("email");
	const error = searchParams.has("error")
		? {
				message:
					searchParams.get("error_description") || searchParams.get("error")
			}
		: null;

	return (
		<MutationForm
			className="grid w-[32rem] overflow-hidden rounded-xl border border-tertiary-300 bg-tertiary-50"
			mutationKey={["login"]}
			defaultVariables={{
				password: "",
				username_or_email: email || ""
			}}
			mutationFn={async (body: LoginCredentials) => {
				const { data, error } = await api.login({ body });

				if (error || !data) throw error;
				return data;
			}}
			onSuccess={async (session) => {
				await queryClient.setQueryData(["session"], session);
				restoreReturnIntent();
			}}
		>
			{({ fields: { username_or_email, password } }) => (
				<div className="flex flex-col gap-8 p-8">
					<span className="text-xl">
						<VSekaiMark className="inline size-5" /> Login
					</span>
					<OAuth2ButtonGroup />
					<HorizontalSeparator>or</HorizontalSeparator>
					<div className="flex flex-col gap-4">
						<Input {...username_or_email} label="Email or Username" />
						<Input {...password} label="Password" type="password" />
					</div>
					<div className="flex flex-col gap-4">
						<FormErrorMessage messageOverride={error?.message} />
						<FormButton size="large">Continue</FormButton>
						<p className="opacity-75">
							Don&apos;t have an account yet?{" "}
							<InlineLink href={withReturnIntent("/sign-up")}>
								Create one
							</InlineLink>
							.
						</p>
					</div>
				</div>
			)}
		</MutationForm>
	);
};
