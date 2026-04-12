"use client";

import { ExternalLink } from "lucide-react";
import { SiDiscord, SiGithub } from "@icons-pack/react-simple-icons";

import { useReturnIntent } from "~/hooks/return-intent";
import { Button } from "~/components/button";

import { loginWithProvider } from "./actions";

import type { ComponentProps, FC } from "react";
import type { ProviderID } from "~/api";

export const providerMetadata: Record<
	ProviderID,
	{
		name: string;
		Icon: FC<ComponentProps<"svg">>;
	}
> = {
	discord: {
		Icon: SiDiscord,
		name: "Discord"
	},
	github: {
		Icon: SiGithub,
		name: "GitHub"
	}
};

export const OAuth2Button: FC<{ providerId: ProviderID }> = ({
	providerId
}) => {
	const { name, Icon } = providerMetadata[providerId] || {
		Icon: ExternalLink,
		name: providerId
	};

	const { returnIntent } = useReturnIntent();

	return (
		<Button
			size="large"
			type="light"
			onClick={async () => {
				await loginWithProvider(providerId, { ri: returnIntent?.href || "/" });
			}}
		>
			<Icon className="size-5 shrink-0" />
			{name}
		</Button>
	);
};

export const OAuth2ButtonGroup: FC = () => {
	return (
		<div className="grid grid-cols-3 gap-4">
			{Object.keys(providerMetadata).map((providerId) => (
				<OAuth2Button key={providerId} providerId={providerId} />
			))}
		</div>
	);
};
