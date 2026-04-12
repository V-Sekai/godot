export const themes = ["light", "dark"] as const;
export type Theme = (typeof themes)[number];

export const defaultTheme = "light" as Theme;
