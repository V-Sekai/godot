"use client";

import Image from "next/image";
import { useState, type FC, type PropsWithChildren } from "react";

import { Input } from "~/components/input";
import {
	Dialog,
	DialogContent,
	DialogHeader,
	DialogTitle
} from "~/components/dialog";
import VSekaiRed from "~/assets/v-sekai-red.png";
import { FormButton, FormErrorMessage, MutationForm } from "~/hooks/form";
import { api } from "~/api";

import { invalidateUser, useUser } from "../data";

import { StatusBadge } from "./status-badge";

export const EditProfile: FC<PropsWithChildren<{ userId: string }>> = ({
	userId,
	children
}) => {
	const user = useUser(userId);

	const [dialogOpen, setDialogOpen] = useState(false);

	if (!user) return children;
	const { icon, banner } = user;

	return (
		<Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
			{children}
			<MutationForm
				asChild
				defaultVariables={{
					biography: user.biography,
					display_name: user.display_name
				}}
				mutationFn={async ({ display_name, biography }) => {
					const { data, error } = await api.updateUser({
						body: { biography, display_name },
						path: { user_id: "me" }
					});

					if (error || !data) throw error;
					return data;
				}}
				onSuccess={(user) => {
					invalidateUser(user);
					setDialogOpen(false);
				}}
			>
				{({ fields: { display_name, biography } }) => (
					<DialogContent asChild>
						<form>
							<DialogHeader>
								<DialogTitle>Edit profile</DialogTitle>
								<FormButton type="light">Save</FormButton>
							</DialogHeader>
							<div className="-mx-6 flex flex-col">
								<div
									className="aspect-[16/4] w-full bg-cover bg-center"
									style={{
										backgroundImage: banner ? `url(${banner})` : undefined
									}}
								/>
								<div className="-mt-12 flex items-center gap-4 px-6">
									<Image
										priority
										alt={`${user.display_name}'s profile picture`}
										className="size-24 shrink-0 rounded-xl"
										height={144}
										src={icon || VSekaiRed.src}
										width={144}
									/>
									<StatusBadge user={user} />
								</div>
							</div>
							<Input {...display_name} label="Display Name" maxLength={24} />
							<Input
								{...biography}
								multiline
								label="Biography"
								maxLength={256}
							/>
							<FormErrorMessage />
						</form>
					</DialogContent>
				)}
			</MutationForm>
		</Dialog>
	);
};
