import { skipToken, useSuspenseQuery } from "@tanstack/react-query";

import {
	type api,
	getAdminStatus
} from "~/api";
import { useOptionalSession } from "~/hooks/session";
import { getQueryClient } from "~/query";

export function useAdminStatus() {
	const session = useOptionalSession();

	const { data: admin_status = null } = useSuspenseQuery({
		queryFn:
			async ({ signal }) => {
					const { data, error, response } = await getAdminStatus({
						signal
					});

					if (response.status === 404) return null;
					if (error) throw error;
					return data;
				},
		queryKey: ["adminStatus"]
	});

	if (!admin_status) return null;

	return {
		...admin_status,
	};
}
