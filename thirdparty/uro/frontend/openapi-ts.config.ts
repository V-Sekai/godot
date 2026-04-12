import { defineConfig } from "@hey-api/openapi-ts";

export default defineConfig({
	client: "@hey-api/client-fetch",
	input: "src/__generated/openapi.json",
	output: "src/__generated/api",
	schemas: false
});
