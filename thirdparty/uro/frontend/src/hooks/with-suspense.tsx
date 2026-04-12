import {
	Suspense,
	type ComponentProps,
	type JSXElementConstructor
} from "react";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function withSuspense<T extends JSXElementConstructor<any>>(
	Component: T,
	options: Parameters<typeof Suspense>[0] = {}
) {
	return function WithSuspense(props: ComponentProps<T>) {
		return (
			<Suspense {...options}>
				<Component {...props} />
			</Suspense>
		);
	};
}
