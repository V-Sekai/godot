import { InlineLink } from "~/components/link";
import { urls } from "~/environment";
import { VSekaiMark } from "~/components/vsekai-mark";

import { Footer } from "../footer";

import { Section, SectionTitle } from "./section";

export default function LandingPage() {
	return (
		<main className="mx-auto flex w-full max-w-screen-lg flex-col lg:pt-16">
			<Section className="py-16 pb-32 text-lg lg:items-center lg:text-center">
				<h1 className="mx-auto flex w-full max-w-screen-md items-center gap-2 px-4 py-6 text-3xl font-medium lg:justify-center">
					<VSekaiMark className="size-12 shrink-0" />
					<span>V-Sekai</span>
				</h1>
				<div className="mx-auto flex max-w-screen-md flex-col gap-4 px-4">
					<p className="text-balance italic">
						Your virtual reality platform, on your game engine.
					</p>
					<p>
						Imagine a completely{" "}
						<InlineLink href={urls.github} title="V-Sekai GitHub page">
							free and open source
						</InlineLink>{" "}
						social VR platform, utilizing the community and the power of{" "}
						<InlineLink href="https://godotengine.org/">
							Godot Engine
						</InlineLink>{" "}
						4.
					</p>
					<p>
						V-Sekai, A completely community-run, self-hostable{" "}
						<InlineLink href="https://choosealicense.com/licenses/mit/">
							MIT open source
						</InlineLink>{" "}
						<br className="hidden lg:inline" />
						software stack built with Godot Engine 4.
					</p>
					<p>
						<InlineLink href={urls.discord}>
							Join our Discord community
						</InlineLink>
					</p>
					<p className="text-balance">
						Are you a developer? Clone or fork our
						<br /> code on the{" "}
						<InlineLink href={urls.github}>
							V-Sekai GitHub organization
						</InlineLink>
						.
					</p>
				</div>
			</Section>
			<div className="mx-auto flex w-full max-w-screen-md flex-col gap-4 px-4 text-lg">
				<Section>
					<SectionTitle>About us</SectionTitle>
					<p>
						V-Sekai is a community of enthusiasts building a next-generation
						social and gaming VR platform powered by the{" "}
						<InlineLink href="https://godotengine.org/">
							Godot Engine 4.0
						</InlineLink>
						. Please{" "}
						<InlineLink href={urls.discord}>join our Discord server</InlineLink>{" "}
						to receive updates and join the community.
					</p>
				</Section>
				<Section>
					<SectionTitle>Why Godot Engine?</SectionTitle>
					<p>
						We felt there was a need for a remixable ecosystem focused on VR.
						Social VR should be a truly community-driven experience. For this to
						be realised, we believe the entire software stack should be
						available, moddable, and tweakable from the source up, which is why
						we encourage you to{" "}
						<InlineLink href={urls.github}>
							tinker with our code on GitHub
						</InlineLink>{" "}
						and a lot of our projects focus on the MIT-Licensed{" "}
						<InlineLink href="https://godotengine.org/">
							Godot Engine
						</InlineLink>
						.
					</p>
					<p>
						As such, the license of Godot Engine is compatible with our goals
						and should allow for fully custom-built solutions at the engine
						level. Additionally, we&apos;re able to make use of the hard work of
						thousands of Godot contributors to keep the technology stack
						powering V-Sekai innovative and competitive amongst industry leading
						engines.
					</p>
					<p>
						An astute reader might be wondering, why not use a more developed,{" "}
						<em>commercial</em> engine, perhaps even one whose name begins with
						&quot;U&quot;? It&apos;s a good question, and there are other
						projects which may go this route. By synergizing with the dozens of
						core Godot developers and thousands of individual contributors,
						building upon the rising star Godot Engine 4.0 offers V-Sekai a
						unique opportunity to create a truly community owned, fully free
						ecosystem that won&apos;t be forever in a corporate shadow. We hope
						to follow a shining role model,{" "}
						<InlineLink href="https://blender.org">Blender</InlineLink>, which
						has grown over two decades to become an industry leader.
					</p>
				</Section>
				<Section>
					<SectionTitle>A unique community</SectionTitle>
					<p>
						V-Sekai is the first fully-open source social VR platform running on
						the Godot Engine. It is designed as a living virtual space where you
						can meet people and interact in a virtual space, while leveraging
						the fast-growing Godot game engine to let users create any content
						they want and have it be immediately accessible and shareable to a
						growing community.
					</p>
				</Section>
				<Section>
					<SectionTitle>Why V-Sekai?</SectionTitle>
					<p>
						We believe social VR should be a truly community-driven experience.
						For this to be realised, we believe the entire software stack should
						be available, moddable, and tweakable from the source up, in order
						to ensure V-Sekai develops into the best platform it can possibly
						be.
					</p>
				</Section>
				<Section>
					<SectionTitle>Who are we?</SectionTitle>
					<p>
						We are a small group of developers and VR enthusiasts who felt there
						was a need for a project like this. Many of us already love the
						existing offerings out there, but wanted our own self-sustainable
						platform which could survive on the backs of its community, rather
						than being tied to the life of a company and a small group of
						individuals with exclusive control over its direction.
					</p>
				</Section>
				<Section>
					<SectionTitle>What do we offer?</SectionTitle>
					<p>
						We have built a framework for the Godot game engine which makes the
						distribution of custom avatars and worlds in a networked environment
						possible. Our current plan is to offer hosting via our own web
						server, as well as individual server hosting and social functions,
						and in the future plan to broaden what we can offer. We have already
						been tackling many of the technical challenges required to make such
						a project possible on the Godot game engine, including:
					</p>
					<ul className="list-inside list-disc">
						<li>Spatialised audio and VOIP support </li>
						<li>Native VRM support</li>
						<li>Our own custom networking stack</li>
						<li>
							A security-focused sandboxed scripting environment backed by WASM
						</li>
						<li>Customisable servers with their own individual game rules</li>
					</ul>
				</Section>
				<Section>
					<SectionTitle>Alright, when can we get it?</SectionTitle>
					<p>
						At the moment, Godot Engine 4.0 is in early beta, and there is
						nothing usable outside of development purposes. We encourage you to
						connect with us on Discord, check out the code, or try the Godot
						Engine 4.0 beta.
					</p>
					<p>
						We want to ensure that our formal release is as polished as it can
						be. Please{" "}
						<InlineLink href="https://discord.gg/7BQDHesck8">
							join our Discord server
						</InlineLink>{" "}
						to learn more and stay connected.
					</p>
				</Section>
			</div>
			<Footer />
		</main>
	);
}
