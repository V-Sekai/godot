/** @type {import('next').NextConfig} */
const nextConfig = {
	eslint: {
		ignoreDuringBuilds: true
	},
	experimental: {
		// reactCompiler: true,
		// turbotrace: {}
	},
	images: {
		unoptimized: true
	},
	output: "standalone",
	rewrites: async () => {
		return {
			afterFiles: [
				{
					destination: "/user/:username/:path*",
					source: "/@:username/:path*"
				}
			]
		};
	}
};

export default nextConfig;
