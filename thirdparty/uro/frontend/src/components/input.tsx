"use client";

import { tv, type VariantProps } from "tailwind-variants";
import { Eye, EyeOff } from "lucide-react";
import { twMerge } from "tailwind-merge";
import { useEffect, useState, type Dispatch, type ReactNode } from "react";

import { ErrorMessage } from "~/app/(static)/login/error-message";

import { Button } from "./button";

const tvInput = tv({
	defaultVariants: {
		disabled: false,
		multiline: false,
		status: "idle"
	},
	slots: {
		base: "group relative flex w-full cursor-text flex-col rounded-xl border border-tertiary-300 bg-tertiary-50 outline-current transition-all focus-within:bg-tertiary-100 focus-within:outline",
		content: "relative flex gap-2 px-4 pb-3",
		header:
			"pointer-events-none flex justify-between px-4 pt-3 text-sm opacity-75 transition-all",
		input: "w-full bg-transparent outline-none"
	},
	variants: {
		disabled: {
			true: {
				base: "cursor-default opacity-50"
			}
		},
		multiline: {
			true: {}
		},
		status: {
			error: {
				base: "border-red-500/50"
			},
			idle: {},
			success: {}
		}
	}
});

export type InputAutocomplete = "off" | (string & {});

export interface InputProps<T extends string>
	extends VariantProps<typeof tvInput> {
	value?: T;
	onChange?: Dispatch<T>;
	name?: string;
	defaultValue?: T;
	label: ReactNode;
	autoComplete?: InputAutocomplete;
	maxLength?: number;
	errors?: Array<string>;
	type?: "text" | "password" | "email";
	rows?: number;
	endContent?: ReactNode;
}

export function Input<T extends string>({
	value,
	onChange,
	label,
	defaultValue,
	endContent,
	name,
	type = "text",
	errors = [],
	autoComplete = "off",
	maxLength,
	status,
	multiline = false,
	disabled = false,
	rows = multiline ? 3 : 1,
	...tvProps
}: InputProps<T>) {
	const originalType = type;

	if (!autoComplete || autoComplete === "off") {
		if (type === "password") autoComplete = "current-password";
		if (type === "email") autoComplete = "email";
	}

	const [masked, setMasked] = useState(type === "password");
	const MaskedIcon = masked ? EyeOff : Eye;

	useEffect(() => setMasked(type === "password"), [type]);
	if (!masked && type === "password") type = "text";

	status = errors.length > 0 ? "error" : status;

	const {
		base,
		header: headerClassName,
		content,
		input
	} = tvInput({ disabled, multiline, status, ...tvProps });

	const InputComponent = multiline ? "textarea" : "input";

	return (
		<label className={base()}>
			{label && (
				<div className={headerClassName()}>
					{label}
					{value && maxLength && (
						<span className="opacity-75">
							<span
								className={twMerge(value.length >= maxLength && "text-red-500")}
							>
								{value.length}
							</span>
							/{maxLength}
						</span>
					)}
				</div>
			)}
			<div className={content()}>
				<InputComponent
					autoComplete={autoComplete}
					className={input()}
					defaultValue={defaultValue}
					disabled={disabled}
					maxLength={maxLength}
					name={name}
					rows={rows}
					type={masked ? "password" : type}
					value={value}
					onChange={({ currentTarget: { value } }) => {
						onChange?.(value as T);
					}}
				/>
				{endContent}
				{originalType === "password" && (
					<Button
						iconOnly
						className="aspect-square rounded-full !p-0 opacity-75 hover:opacity-100"
						size="small"
						type="ghost"
						onClick={() => setMasked((masked) => !masked)}
					>
						<MaskedIcon className="size-5" />
					</Button>
				)}
			</div>
			<ErrorMessage
				className={twMerge(
					"overflow-hidden rounded-b-xl rounded-t-none border-0 border-t transition-all"
				)}
				message={
					status === "error" && errors.length > 0
						? `${(name || label)?.toString().replaceAll("_", " ")} ${errors[0]}`.trim()
						: null
				}
			/>
		</label>
	);
}
