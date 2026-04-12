"use client";

import { LogIn } from "lucide-react";
import { notFound } from "next/navigation";

import { VSekaiMark } from "~/components/vsekai-mark";
import { Button } from "~/components/button";
import { useOptionalSession } from "~/hooks/session";
import { Banner } from "~/components/banner";
import { useReturnIntent } from "~/hooks/return-intent";

import { useUser } from "../data";

import { UserProfile } from "./profile";

export default function UserPage({
	params: { userId }
}: {
	params: { userId: string };
}) {
	const { withReturnIntent } = useReturnIntent();
	const session = useOptionalSession();

	const user = useUser(userId);
	if (!user) return notFound();

	return (
		<>
			{!session && (
				<Banner
					actions={
						<>
							<Button href={withReturnIntent("/login")} type="ghost">
								Sign In <LogIn className="size-4" />
							</Button>
							<Button href={withReturnIntent("/sign-up")}>
								Create an Account
							</Button>
						</>
					}
				>
					<VSekaiMark className="inline size-4" /> V-Sekai, a completely free
					and open source social VR platform.
					{/* <div className="absolute inset-y-0 right-0 my-auto hidden h-fit gap-2 md:flex">
						<Button href={withReturnIntent("/login")} type="ghost">
							Sign In <LogIn className="size-4" />
						</Button>
						<Button href={withReturnIntent("/sign-up")}>
							Create an Account
						</Button>
					</div> */}
				</Banner>
			)}
			<UserProfile userId={userId} />
		</>
	);
}
