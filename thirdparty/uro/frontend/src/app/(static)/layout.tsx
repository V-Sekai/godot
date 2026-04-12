import { Header } from "./header";

import type { PropsWithChildren } from "react";

export default function StaticLayout({ children }: PropsWithChildren) {
	return (
		<>
			<Header />
			{children}
		</>
	);
}
