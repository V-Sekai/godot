import {
	QueryClient,
	defaultShouldDehydrateQuery,
	dehydrate as _dehydrate,
	hashKey,
	type QueryKey
} from "@tanstack/react-query";
import { cache } from "react";

let _queryClient: QueryClient | null;

export const getQueryClient = cache(() => {
	if (_queryClient && typeof window !== "undefined") return _queryClient;

	return (_queryClient = new QueryClient({
		defaultOptions: {
			dehydrate: {
				shouldDehydrateQuery: (query) =>
					defaultShouldDehydrateQuery(query) || query.state.status === "pending"
			},
			queries: {
				staleTime: 60 * 1000
			}
		}
	}));
});

export function optimisticMutation<T>(queryKey: QueryKey) {
	const queryClient = getQueryClient();

	return async (newValue: T) => {
		await queryClient.cancelQueries({ queryKey });

		const previousValue = queryClient.getQueryData(queryKey);
		queryClient.setQueryData(queryKey, newValue);

		return previousValue;
	};
}

export function dehydrateAll(queryClient: QueryClient) {
	const dehydratedState = _dehydrate(queryClient, {
		shouldDehydrateMutation: () => false,
		shouldDehydrateQuery: () => true
	});

	return dehydratedState;
}

export function dehydrateMany(
	queryClient: QueryClient,
	queryKeys: Array<QueryKey>
) {
	const queryHashs = new Set(queryKeys.map((queryKey) => hashKey(queryKey)));

	const dehydratedState = _dehydrate(queryClient, {
		shouldDehydrateMutation: () => false,
		shouldDehydrateQuery: (options) => queryHashs.has(options.queryHash)
	});

	return dehydratedState;
}

export function dehydrateSpecific(
	queryClient: QueryClient,
	queryKey: QueryKey
) {
	return dehydrateMany(queryClient, [queryKey]);
}
