"use client";

import { LogIn, PaintBucket } from "lucide-react";

import { InlineLink } from "~/components/link";
import { logout, useOptionalSession } from "~/hooks/session";
import { Button } from "~/components/button";
import { useTheme } from "~/hooks/theme";
import { useReturnIntent } from "~/hooks/return-intent";

import type { FC } from "react";

export const HeaderUserNavigation: FC = () => {
	const session = useOptionalSession();

	const { toggle: toggleTheme } = useTheme();
	const { withReturnIntent } = useReturnIntent();

	return (
		<div className="flex items-center gap-2 text-sm">
			{session ? (
				<>
					<span className="mr-2">
						Logged in as{" "}
						<InlineLink
							className="font-medium"
							href={`/@${session.user.username}`}
						>
							{session.user.display_name}
						</InlineLink>
					</span>
					<Button onClick={logout}>Sign out</Button>
				</>
			) : (
				<>
					<Button href={withReturnIntent("/login")} type="ghost">
						Sign In
						<LogIn className="size-4" />
					</Button>
					<Button href={withReturnIntent("/sign-up")}>Create an Account</Button>
				</>
			)}
			<Button iconOnly type="ghost" onClick={toggleTheme}>
				<PaintBucket className="size-4" />
			</Button>
		</div>
	);
};
