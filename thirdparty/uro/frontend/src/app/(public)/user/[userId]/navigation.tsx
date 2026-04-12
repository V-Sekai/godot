"use client";

import { Home, LogOut, PaintBucket } from "lucide-react";
import { twMerge } from "tailwind-merge";

import { logout, useOptionalSession } from "~/hooks/session";
import { Link } from "~/components/link";
import { useTheme } from "~/hooks/theme";

import type { ComponentProps, FC, PropsWithChildren } from "react";

export const NavigationItem: FC<
	PropsWithChildren<{
		href?: string;
		onClick?: () => void;
		Icon?: FC<ComponentProps<"svg">>;
		className?: string;
		invert?: boolean;
	}>
> = ({ href, onClick, Icon, className, invert = false, children }) => {
	const Component = href ? Link : "div";

	return (
		<Component
			href={href!}
			className={twMerge(
				"flex cursor-pointer gap-4 px-4 py-4 first:rounded-t-xl last:rounded-b-xl md:px-6",
				invert ? "text-white" : "hover:bg-tertiary-200",
				className
			)}
			onClick={onClick}
		>
			{Icon && <Icon className="size-6 shrink-0" />}
			<span className="hidden select-none xl:block">{children}</span>
		</Component>
	);
};

export const Navigation: FC = () => {
	const session = useOptionalSession();
	const theme = useTheme();

	return (
		<div className="flex shrink-0 flex-col gap-3 xl:w-72">
			<div className="flex flex-col rounded-xl border border-tertiary-300 bg-tertiary-50">
				<NavigationItem href="/" Icon={Home}>
					Home
				</NavigationItem>
				<NavigationItem href="/" Icon={Home}>
					Home
				</NavigationItem>
				<NavigationItem href="/" Icon={Home}>
					Home
				</NavigationItem>
				<NavigationItem href="/" Icon={Home}>
					Home
				</NavigationItem>
			</div>
			<div className="flex flex-col rounded-xl border border-tertiary-300 bg-tertiary-50">
				<NavigationItem Icon={PaintBucket} onClick={theme.toggle}>
					Theme
				</NavigationItem>
				{session && (
					<NavigationItem Icon={LogOut} onClick={logout}>
						Logout
					</NavigationItem>
				)}
			</div>
			<div className="hidden xl:block">
				<p className="text-xs leading-snug opacity-75">
					V-Sekai v0, written & maintained by the V-Sekai Team, licensed under
					the MIT License.
				</p>
			</div>
		</div>
	);
};
