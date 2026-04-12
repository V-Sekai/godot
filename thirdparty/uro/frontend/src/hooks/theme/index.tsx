"use client";

import { useMutation, useSuspenseQuery } from "@tanstack/react-query";
import { useCallback, type FC, type PropsWithChildren } from "react";
import { Slot } from "@radix-ui/react-slot";

import { optimisticMutation } from "~/query";

import { setTheme } from "./actions";

import type { Theme } from "./common";

export function useTheme() {
	const { data: theme } = useSuspenseQuery<Theme>({
		queryKey: ["theme"]
	});

	const { mutate: set } = useMutation({
		mutationFn: setTheme,
		mutationKey: ["theme"],
		onMutate: optimisticMutation(["theme"])
	});

	const toggle = useCallback(
		() => set(theme === "light" ? "dark" : "light"),
		[theme, set]
	);

	return { set, theme, toggle };
}

export const ThemeOverride: FC<PropsWithChildren<{ theme?: Theme }>> = ({
	theme: themeOverride,
	children
}) => {
	const { theme: _theme } = useTheme();
	const theme = themeOverride || _theme;

	return <Slot className={theme}>{children}</Slot>;
};

export * from "./common";
