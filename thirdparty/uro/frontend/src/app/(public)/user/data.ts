import { skipToken, useSuspenseQuery } from "@tanstack/react-query";

import {
	type api,
	getUser,
	type User,
	type LooseUserKey,
	friendStatus
} from "~/api";
import { useOptionalSession } from "~/hooks/session";
import { getQueryClient } from "~/query";

export function useUser(userId: LooseUserKey) {
	const session = useOptionalSession();

	const { data: user = null } = useSuspenseQuery({
		queryFn:
			userId === "me" ||
			userId === session?.user.id ||
			userId === session?.user.username
				? () => session?.user || null
				: async ({ queryKey: [, userId], signal }) => {
						if (!userId) return null;

						const { data, error, response } = await getUser({
							path: { user_id: userId },
							signal
						});

						if (response.status === 404) return null;
						if (error) throw error;
						return data;
					},
		queryKey: ["users", userId]
	});

	if (!user) return null;

	return {
		...user,
		banner: "https://unsplash.it/1600/900/?random"
	};
}

export const friendshipQueryKey = (userId: LooseUserKey) => [
	"users",
	userId,
	"friendship"
];

export function useFriendship(userId: LooseUserKey) {
	const session = useOptionalSession();

	const { data: data = null } = useSuspenseQuery({
		queryFn:
			session &&
			!(
				userId === "me" ||
				userId === session?.user.id ||
				userId === session?.user.username
			)
				? async ({ queryKey: [, userId], signal }) => {
						if (!userId) return null;

						const { data, error, response } = await friendStatus({
							path: { user_id: userId },
							signal
						});

						if (response.status === 404) return null;
						if (error) throw error;
						return data;
					}
				: () => null,
		queryKey: friendshipQueryKey(userId)
	});

	return data;
}

export function invalidateUser(user: User) {
	const queryClient = getQueryClient();
	const session = queryClient.getQueryData<api.Session>(["session"]);

	queryClient.setQueryData(["users", user.id], user);
	queryClient.setQueryData(["users", user.username], user);

	if (session && session.user.id === user.id)
		queryClient.setQueryData(["session"], (previous: api.Session) => ({
			...previous,
			user
		}));
}
