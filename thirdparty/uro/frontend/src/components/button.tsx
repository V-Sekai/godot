/* eslint-disable @typescript-eslint/no-explicit-any */

"use client";

import {
	forwardRef,
	useState,
	type ButtonHTMLAttributes,
	type ForwardedRef,
	type MouseEventHandler,
	type PropsWithChildren
} from "react";
import { tv, type VariantProps } from "tailwind-variants";

import { dataAttribute } from "~/element";

import { Link } from "./link";

export const button = tv({
	base: [
		"group flex items-center justify-center whitespace-nowrap transition-all",
		"outline-offset-2 outline-current focus-visible:outline",
		"rounded-xl group-data-[button-group]:rounded-none group-data-[button-group]:first:rounded-l-xl group-data-[button-group]:last:rounded-r-xl",
		"border border-transparent group-data-[button-group]:border-x-0 group-data-[button-group]:first:border-l group-data-[button-group]:last:border-r",
		"data-[pressed]:scale-[.97] group-data-[button-group]:first:origin-right group-data-[button-group]:last:origin-left"
	],
	defaultVariants: {
		disabled: false,
		iconOnly: false,
		pending: false,
		size: "medium",
		type: "primary"
	},
	variants: {
		disabled: {
			true: "pointer-events-none opacity-50"
		},
		iconOnly: {
			true: "!p-2"
		},
		pending: {
			true: "pointer-events-none animate-pulse"
		},
		size: {
			large: "gap-3 px-6 py-3 text-base",
			medium: "gap-2 px-4 py-1 text-base",
			small: "gap-2 px-2 py-1 text-sm"
		},
		type: {
			ghost: "text-secondary-100",
			light:
				"border-tertiary-300/20 bg-tertiary-100/5 text-secondary-100 backdrop-blur",
			primary: "bg-blue-500 text-white hover:bg-blue-600"
		}
	}
});

export type ButtonProps = PropsWithChildren<
	VariantProps<typeof button> & {
		href?: string | URL;
		onClick?: MouseEventHandler<HTMLElement>;
		action?: () => void;
		className?: string;
		actionType?: ButtonHTMLAttributes<HTMLButtonElement>["type"];
	}
>;

export const Button = forwardRef<HTMLElement, ButtonProps>(
	(
		{
			onClick,
			action,
			href,
			actionType = "button",
			disabled,
			children,
			...tvProps
		},
		reference
	) => {
		const [pressed, setPressed] = useState(false);

		const Component = href ? Link : "button";

		return (
			<Component
				className={button({ disabled, ...tvProps })}
				data-pressed={dataAttribute(pressed)}
				disabled={disabled}
				data-button=""
				href={href!}
				ref={reference as ForwardedRef<any>}
				type={actionType}
				onClick={(event) => {
					if (disabled) return;

					onClick?.(event);
					action?.();
				}}
				onPointerDown={() => setPressed(true)}
				onPointerOut={() => setPressed(false)}
				onPointerUp={() => setPressed(false)}
			>
				{children}
			</Component>
		);
	}
);

Button.displayName = "Button";

export const buttonGroup = tv({
	base: "group flex rounded-xl",
	defaultVariants: {
		direction: "horizontal"
	},
	variants: {
		direction: {
			horizontal: "flex-row",
			vertical: "flex-col"
		}
	}
});

export type ButtonGroupProps = { className?: string } & PropsWithChildren<
	VariantProps<typeof buttonGroup>
>;

export const ButtonGroup = forwardRef<HTMLDivElement, ButtonGroupProps>(
	({ children, ...tvProps }, reference) => {
		return (
			<div
				className={buttonGroup(tvProps)}
				data-button-group=""
				ref={reference}
			>
				{children}
			</div>
		);
	}
);

ButtonGroup.displayName = "Button.Group";
