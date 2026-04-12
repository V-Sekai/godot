"use server";

import { cookies } from "next/headers";
import { revalidatePath } from "next/cache";

import { themes, type Theme } from "./common";

export async function setTheme(theme: Theme) {
	if (!themes.includes(theme)) return;

	cookies().set("theme", theme);
	revalidatePath("/");
}
