"use client";

import { useDeferredValue, type FC } from "react";
import { Check } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";

import { api } from "~/api";
import { Input, type InputProps } from "~/components/input";
import { HorizontalSeparator } from "~/components/horizontal-separator";
import { useReturnIntent } from "~/hooks/return-intent";
import { InlineLink } from "~/components/link";
import { VSekaiMark } from "~/components/vsekai-mark";
import { useUser } from "~/app/(public)/user/data";
import { FormButton, FormErrorMessage, MutationForm } from "~/hooks/form";
import { Captcha } from "~/components/captcha";

import { OAuth2ButtonGroup } from "../login/oauth2-button";

function transformUsername(username: string): string {
	return username.replaceAll(/\s/g, "").replaceAll(/_+/g, "_").toLowerCase();
}

const UsernameInput: FC<Omit<InputProps<string>, "label">> = ({
	value = "",
	onChange,
	errors = [],
	...props
}) => {
	const deferredValue = useDeferredValue(value);
	const available = !useUser(deferredValue);

	return (
		<Input
			{...props}
			autoComplete="username"
			errors={[...(available ? [] : ["was already taken"]), ...errors]}
			label="Username"
			maxLength={16}
			value={value}
			endContent={
				available &&
				value.length >= 3 && (
					<Check className="size-5 shrink-0 text-green-500" />
				)
			}
			onChange={(value) => onChange?.(transformUsername(value))}
		/>
	);
};

export const SignUpForm: FC = () => {
	const { withReturnIntent } = useReturnIntent();
	const queryClient = useQueryClient();

	return (
		<MutationForm
			className="grid w-[32rem] overflow-hidden rounded-xl border border-tertiary-300 bg-tertiary-50"
			mutationKey={["signup"]}
			defaultVariables={{
				captcha: "",
				display_name: "",
				email: "",
				password: "",
				username: ""
			}}
			mutationFn={async (body) => {
				const { data, error } = await api.signup({
					body
				});

				if (error || !data) throw error;

				queryClient.setQueryData(["session"], data);
			}}
			onSettled={async () => {
				await queryClient.invalidateQueries({
					predicate: ({ queryKey }) => queryKey[0] === "users"
				});
			}}
		>
			{({ fields: { display_name, username, email, password, captcha } }) => (
				<div className="flex flex-col gap-8 p-8">
					<span className="text-xl">
						<VSekaiMark className="inline size-5" /> Create an Account
					</span>
					<OAuth2ButtonGroup />
					<HorizontalSeparator>or</HorizontalSeparator>
					<div className="flex flex-col gap-4">
						<Input
							{...display_name}
							autoComplete="name"
							label="Display Name"
							maxLength={32}
							onChange={(displayName) => {
								display_name.onChange(displayName);

								const newUsername = transformUsername(displayName);

								if (
									username.value !== "" &&
									newUsername !== "" &&
									// Trim the last character because this value is always one re-render behind.
									username.value !== newUsername.slice(0, username.value.length)
								)
									return;

								username.onChange(newUsername);
							}}
						/>
						<UsernameInput {...username} />
						<Input {...email} autoComplete="email" label="Email" type="email" />
						<Input
							{...password}
							autoComplete="new-password"
							label="Password"
							type="password"
						/>
						<div className="mx-auto">
							<Captcha {...captcha} />
						</div>
					</div>
					<div className="flex flex-col gap-4">
						<FormErrorMessage />
						<FormButton size="large">Continue</FormButton>
						<p className="opacity-75">
							Already have an account?{" "}
							<InlineLink href={withReturnIntent("/login")}>Sign In</InlineLink>
							.
						</p>
					</div>
				</div>
			)}
		</MutationForm>
	);
};
