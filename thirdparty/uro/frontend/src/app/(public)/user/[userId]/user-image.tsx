import Image, { type ImageProps } from "next/image";
import { twMerge } from "tailwind-merge";

import VSekaiRed from "~/assets/v-sekai-red.png";

import type { FC } from "react";
import type { User } from "~/api";

export const UserImage: FC<
	Omit<ImageProps, "src" | "alt"> & {
		user: Pick<User, "display_name" | "icon">;
	}
> = ({ user: { display_name, icon }, className, ...props }) => {
	return (
		<Image
			{...props}
			alt={`${display_name}'s profile picture`}
			className={twMerge("shrink-0 rounded-xl", className)}
			src={icon || VSekaiRed.src}
		/>
	);
};
