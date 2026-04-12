export function dataAttribute(
	condition: string | boolean | undefined
): string | undefined {
	if (!condition) return undefined;

	if (typeof condition === "boolean") return "";
	return condition;
}
