import { twMerge } from "tailwind-merge";

import type { ComponentProps, FC } from "react";

export const Section: FC<ComponentProps<"section">> = ({
	className,
	children,
	...props
}) => (
	<section
		className={twMerge("flex flex-col gap-4 pb-8", className)}
		{...props}
	>
		{children}
	</section>
);

export const SectionTitle: FC<ComponentProps<"h2">> = ({
	className,
	children
}) => (
	<h2
		className={twMerge(
			"flex items-center gap-4 py-4 text-xl font-medium",
			className
		)}
	>
		<div className="size-5 rounded-full bg-blue-500" />
		{children}
	</h2>
);
