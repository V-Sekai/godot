"use client";

import { Mail } from "lucide-react";
import { SiDiscord, SiX } from "@icons-pack/react-simple-icons";

import { InlineLink } from "~/components/link";
import { urls } from "~/environment";
import { Section, SectionTitle } from "~/app/(static)/section";
import { Footer } from "~/app/footer";

export default function AboutPage() {
	return (
		<main className="mx-auto flex w-full max-w-screen-lg flex-col pt-8 lg:pt-16">
			<div className="mx-auto flex w-full max-w-screen-md flex-col gap-4 px-4 text-lg">
				<Section>
					<SectionTitle>About V-Sekai</SectionTitle>
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
						<li>Spatialised audio and VOIP support</li>
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
						At the moment, we have functional prototypes, but are still deep in
						development. We have slowly been offering download keys to a small
						group of private volunteering testers. We want to ensure that our
						formal release is as polished as it can be.
					</p>
					<p>
						<InlineLink href="/sign-up">Register now</InlineLink> to reserve
						your name and to receive updates on future beta participation. You
						can also follow us on{" "}
						<InlineLink href={urls.twitter}>Twitter</InlineLink> for updates on
						the project.
					</p>
					<p>
						If you&apos;re interested in developing for Godot Engine, chatting
						about anything or learning more, contact any of us directly.
						We&apos;d love to hear from you:
					</p>
					<ul className="list-inside list-disc">
						<li>
							Join the V-Sekai Discord:{" "}
							<InlineLink href={urls.discord}>
								V-Sekai Community Discord Server
							</InlineLink>
						</li>
						<li>
							Follow us on Twitter:{" "}
							<InlineLink href={urls.twitter}>@VSekaiOfficial</InlineLink>
						</li>
					</ul>
					<div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3">
						{[
							{
								discord: {
									id: "293462873508937728",
									name: "lyuma"
								},
								email: "xn.lyuma@gmail.com",
								twitter: "Lyuma2d"
							},
							{
								discord: {
									id: "286756637446766602",
									name: "saracenone"
								},
								email: "saracenone@gmail.com",
								twitter: "SaracenGameDev"
							},
							{
								discord: {
									id: "399349882432782356",
									name: "mmmaellon"
								},
								email: "mmmaellon@gmail.com",
								twitter: "MMMaellon"
							}
						].map(({ discord, twitter, email }) => (
							<div
								className="relative flex flex-col gap-2 overflow-hidden rounded-lg border border-tertiary-200 p-4 hover:bg-tertiary-50"
								key={discord.id}
							>
								<div
									className="absolute right-0 top-0 size-4 bg-blue-500"
									style={{ clipPath: "polygon(100% 0, 100% 100%, 0 0)" }}
								/>
								<InlineLink
									className="before:absolute before:inset-0"
									href={`https://twitter.com/${twitter}`}
								>
									<SiX className="mr-2 inline-block size-4" />
									{twitter}
								</InlineLink>
								<div className="flex flex-col text-sm">
									<InlineLink href={`mailto:${email}`}>
										<Mail className="mr-2 inline-block size-4" />
										{email}
									</InlineLink>
									<InlineLink href={`https://discord.com/users/${discord.id}`}>
										<SiDiscord className="mr-2 inline-block size-4" />
										{discord.name}
									</InlineLink>
								</div>
							</div>
						))}
					</div>
				</Section>
			</div>
			<Footer />
		</main>
	);
}
