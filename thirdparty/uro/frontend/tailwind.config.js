/**
 * @type {import("tailwindcss").Config}
 */
export default {
	content: [
		"./src/components/**/*.{js,ts,jsx,tsx,mdx}",
		"./src/app/**/*.{js,ts,jsx,tsx,mdx}"
	],
	darkMode: "class",
	plugins: [require("tailwindcss-animate")],
	theme: {
		extend: {
			colors: {
				secondary: {
					0: "rgb(var(--secondary-0))",
					100: "rgb(var(--secondary-100))"
				},
				tertiary: {
					0: "rgb(var(--tertiary-0))",
					100: "rgb(var(--tertiary-100))",
					200: "rgb(var(--tertiary-200))",
					300: "rgb(var(--tertiary-300))",
					400: "rgb(var(--tertiary-400))",
					50: "rgb(var(--tertiary-50))",
					500: "rgb(var(--tertiary-500))",
					600: "rgb(var(--tertiary-600))",
					700: "rgb(var(--tertiary-700))",
					800: "rgb(var(--tertiary-800))",
					900: "rgb(var(--tertiary-900))",
					950: "rgb(var(--tertiary-950))"
				}
			},
			opacity: {
				1: "1%",
				2: "2%",
				3: "3%",
				4: "4%"
			}
		},
		fontWeight: {
			medium: "500",
			normal: "400"
		}
	}
};
