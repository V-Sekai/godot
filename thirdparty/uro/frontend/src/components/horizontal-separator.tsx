import { forwardRef, type ComponentProps } from "react";
import { twMerge } from "tailwind-merge";

export const HorizontalSeparator = forwardRef<
	HTMLHRElement,
	ComponentProps<"hr">
>(({ children, className, ...props }, reference) => {
	if (!children)
		return (
			<hr
				className={twMerge("w-full border-tertiary-300", className)}
				ref={reference}
				{...props}
			/>
		);

	return (
		<div className={twMerge("flex items-center gap-4", className)}>
			<hr className="w-full border-tertiary-300" />
			<span className="shrink-0 text-sm opacity-75">{children}</span>
			<hr className="w-full border-tertiary-300" />
		</div>
	);
});

HorizontalSeparator.displayName = "HorizontalSeparator";
