import "server-only";

import { cache } from "react";
import { cookies } from "next/headers";

import { defaultTheme, themes, type Theme } from "./common";

export const getTheme = cache((): Theme => {
	const theme = cookies().get("theme")?.value as Theme | undefined;

	if (!theme || !themes.includes(theme)) return defaultTheme;
	return theme;
});
