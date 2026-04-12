import { useState, type FC } from "react";

import { api } from "~/api";
import { Button } from "~/components/button";
import { useSession } from "~/hooks/session";
import {
	Dialog,
	DialogContent,
	DialogHeader,
	DialogTitle,
	DialogTrigger
} from "~/components/dialog";
import { Input } from "~/components/input";
import { FormButton, FormErrorMessage, MutationForm } from "~/hooks/form";
import { invalidateUser } from "~/app/(public)/user/data";

export const ChangeEmailButton: FC = () => {
	const { user } = useSession();
	const [dialogOpen, setDialogOpen] = useState(false);

	return (
		<Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
			<DialogTrigger asChild>
				<Button className="w-fit" type="ghost">
					Change Email
				</Button>
			</DialogTrigger>
			<MutationForm
				asChild
				defaultVariables={{
					current_password: "",
					email: user.email || ""
				}}
				mutationFn={async ({ email, current_password }) => {
					const { data, error } = await api.updateEmail({
						body: {
							current_password,
							email
						},
						path: {
							user_id: "me"
						}
					});

					if (error || !data) throw error;
					return data;
				}}
				onSuccess={(user) => {
					invalidateUser(user);
					setDialogOpen(false);
				}}
			>
				{({ fields: { email, current_password } }) => (
					<DialogContent asChild>
						<form>
							<DialogHeader>
								<DialogTitle>Change Email</DialogTitle>
								<FormButton className="w-fit" type="light">
									Save
								</FormButton>
							</DialogHeader>
							<Input {...email} label="Email Address" />
							<Input
								{...current_password}
								label="Current Password"
								type="password"
							/>
							<FormErrorMessage />
						</form>
					</DialogContent>
				)}
			</MutationForm>
		</Dialog>
	);
};
