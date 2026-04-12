import { configs, config } from "@ariesclark/eslint-config";

import nextjs from "@ariesclark/eslint-config/nextjs";
import tailwindcss from "@ariesclark/eslint-config/tailwindcss";

export default config({
	extends: [...configs.recommended, ...nextjs, ...tailwindcss],
	ignores: ["out/**", ".next/**"],
	settings: {
		react: {
			version: "18"
		}
	}
});
