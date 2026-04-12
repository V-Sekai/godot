import { session } from "~/api";

export async function getOptionalSession() {
	const {
		data,
		error,
		response: { status }
	} = await session();

	if (status === 401) return null;
	if (error) throw error;

	return data || null;
}
