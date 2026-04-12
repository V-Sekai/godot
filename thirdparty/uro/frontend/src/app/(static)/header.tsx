import { SiDiscord } from "@icons-pack/react-simple-icons";

import { VSekaiMark } from "~/components/vsekai-mark";
import { urls } from "~/environment";
import { Banner, ClosableBannerAction } from "~/components/banner";
import { getBannerState } from "~/components/banner/server";
import { Link } from "~/components/link";

import { HeaderUserNavigation } from "./header-navigation";

export async function Header() {
	return (
		<header className="flex flex-col gap-4">
			<Banner
				actions={<ClosableBannerAction />}
				href={urls.discord}
				state={getBannerState("interested-in-vsekai")}
			>
				<SiDiscord className="inline size-4" /> Interested in{" "}
				<span className="font-medium">V-Sekai</span> or{" "}
				<span className="font-medium">#GodotVR</span> development? Join the
				V-Sekai Discord Server!
			</Banner>
			<div className="mx-auto flex w-full max-w-screen-xl justify-between gap-8 p-4">
				<Link className="flex h-fit shrink-0 items-center gap-4" href="/">
					<VSekaiMark className="size-8 shrink-0" />
					<span className="hidden text-xl font-medium sm:inline">V-Sekai</span>
				</Link>
				<nav className="flex flex-col items-end gap-2">
					<HeaderUserNavigation />
					<div className="flex gap-4">
						<Link href="/about">About</Link>
						<Link href="/download">Download</Link>
					</div>
				</nav>
			</div>
		</header>
	);
}
