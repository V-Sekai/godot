"use client";

import { useMutation } from "@tanstack/react-query";
import { Check } from "lucide-react";

import { api } from "~/api";
import { Button } from "~/components/button";

import type { FC } from "react";

export const ResendButton: FC = () => {
	const { mutate, isPending, status } = useMutation({
		mutationFn: async () => {
			const { error } = await api.resendConfirmationEmail({
				path: { user_id: "me" }
			});

			if (error) throw error;
		}
	});
	const handleClick = () => {
		mutate();
	};

	return (
		<Button
			className="w-fit"
			disabled={status === "success"}
			pending={isPending}
			onClick={handleClick}
		>
			{status === "success" ? (
				<>
					Sent <Check className="size-4 shrink-0" />
				</>
			) : (
				"Resend"
			)}
		</Button>
	);
};
