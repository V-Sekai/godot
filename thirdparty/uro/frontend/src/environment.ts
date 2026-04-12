import { env } from 'next-runtime-env';

function environment<T>(value: unknown, name: string): T {
	if (!value) throw new Error(`Missing environment variable: ${name}.`);
	return value as T;
}

export const development = process.env.NODE_ENV === "development";

if (development) {
	process.env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0";
}

export const urls = {
	discord: "https://discord.gg/7BQDHesck8",
	github: "https://github.com/v-sekai",
	twitter: "https://twitter.com/vsekaiofficial"
};

// Runtime fetch from API
interface serverEnvType {
	origin: string;
	apiOrigin: string;
	turnstileSiteKey: string
};

let envCache: serverEnvType | null = null;


export const getServerEnv = (): serverEnvType | null => {
  if (envCache) {
    // console.log("Using cached env:", envCache);
    return envCache;
  }

  const isServer = typeof window === "undefined";
  const apiOriginKey = isServer ? "API_ORIGIN" : "NEXT_PUBLIC_API_ORIGIN";

  try {
    const serverEnv = {
      origin: environment<string>(env("NEXT_PUBLIC_ORIGIN"), "NEXT_PUBLIC_ORIGIN"),
      apiOrigin: environment<string>(env(apiOriginKey), apiOriginKey),
      turnstileSiteKey: environment<string>(env("NEXT_PUBLIC_TURNSTILE_SITEKEY"), "NEXT_PUBLIC_TURNSTILE_SITEKEY"),
    };
    envCache = serverEnv;
    // console.log(`Fetched environment on ${isServer ? "server" : "client"} side.`, serverEnv);
    return serverEnv;
  } catch (error) {
    console.error("Error fetching server environment:", String(error));
  }

  return null;
};

/**
 * A set of first-party origins, these are given special treatment in the
 * application, such as in OAuth2 redirection & opening links in a new tab.
 * Add to set below if required.
 */
const firstPartyOrigins: Set<string> = new Set([]);

export const getFirstPartyOrigins = (): Set<string> => {
  const appOrigin = getServerEnv()?.origin;
  let origins = firstPartyOrigins;

  // Ensure firstPartyOrigins includes the fetched origin
  if (appOrigin) {
    origins.add(appOrigin);
  }

  // console.log("First-party origins:", [...origins]);
  return origins;
};
