import LinkPrimitive from "next/link";
import { twMerge } from "tailwind-merge";
import {
	type ComponentRef,
	forwardRef,
	type ComponentProps,
	type FC,
	useMemo
} from "react";

import { dataAttribute } from "~/element";
import { getFirstPartyOrigins, getServerEnv } from "~/environment";

export const Link = forwardRef<
	ComponentRef<typeof LinkPrimitive>,
	ComponentProps<typeof LinkPrimitive>
>(({ href: _href, children, className, ...props }, reference) => {
	const { href, external } = useMemo(() => {
		const origin = getServerEnv()?.origin || "";
		const firstPartyOrigins = getFirstPartyOrigins();
		const url = new URL(_href.toString(), origin);
		const href =
			url.origin === origin ? url.href.replace(origin, "") : url.href;

		const external = !firstPartyOrigins.has(url.origin);

		return { external, href };
	}, [_href]);

	return (
		<LinkPrimitive
			data-external={dataAttribute(external)}
			href={href}
			ref={reference}
			target={dataAttribute(external && "_blank")}
			className={twMerge(
				"outline-offset-2 outline-current transition-all focus-visible:outline",
				className
			)}
			{...props}
		>
			{children}
		</LinkPrimitive>
	);
});

Link.displayName = "Link";

export const InlineLink: FC<
	Omit<ComponentProps<typeof Link>, "href"> & { href: URL | string }
> = ({ children, className, ...props }) => {
	return (
		<Link
			className={twMerge(
				"text-blue-500 transition-all hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-500",
				className
			)}
			{...props}
		>
			{children}
		</Link>
	);
};
