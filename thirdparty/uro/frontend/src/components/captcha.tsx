import { useLatest } from "@ariesclark/react-hooks";
import { useMutationState } from "@tanstack/react-query";
import dynamic from "next/dynamic";
import Script from "next/script";
import {
	Suspense,
	use,
	useEffect,
	useMemo,
	useRef,
	type Dispatch,
	type FC
} from "react";
import { twMerge } from "tailwind-merge";

import { getServerEnv } from "~/environment";
import { MutationFormContext } from "~/hooks/form";
import { useTheme } from "~/hooks/theme";

interface CaptchaProps {
	onChange?: Dispatch<string>;
}

const CaptchaContent: FC<
	CaptchaProps & { promise: Promise<Turnstile.Turnstile> }
> = ({ onChange: _onChange, promise }) => {
	const turnstile = use(promise);
	const turnstileSiteKey = getServerEnv()?.turnstileSiteKey || "";

	const { theme } = useTheme();
	const reference = useRef<HTMLDivElement>(null);
	const onChange = useLatest(_onChange);

	useEffect(() => {
		const { current: element } = reference;
		if (!element) return;

		turnstile.render(element, {
			callback: (value) => onChange.current?.(value),
			sitekey: turnstileSiteKey,
			theme
		});

		return () => turnstile.remove(element);
	}, [onChange, turnstile, theme]);

	const { status: formStatus } = use(MutationFormContext) || {};

	useEffect(() => {
		const { current: element } = reference;
		if (!element || formStatus === "pending") return;

		turnstile.reset(element);
	}, [formStatus]);

	return (
		<div
			className={twMerge(
				"relative h-[64px] w-[300px] overflow-hidden rounded-xl border border-tertiary-300",
				theme === "dark" ? "bg-[#222]" : "bg-[#fafafa]"
			)}
		>
			<div className="absolute -inset-px" ref={reference} />
		</div>
	);
};

const CaptchaContentSkeleton: FC = () => {
	const { theme } = useTheme();

	return (
		<div
			className={twMerge(
				"relative h-[64px] w-[300px] animate-pulse overflow-hidden rounded-xl border border-tertiary-300",
				theme === "dark" ? "bg-[#222]" : "bg-[#fafafa]"
			)}
		/>
	);
};

const _Captcha: FC<CaptchaProps> = (props) => {
	const { promise, resolve } = useMemo(
		() => Promise.withResolvers<Turnstile.Turnstile>(),
		[]
	);

	return (
		<>
			<Script
				async
				defer
				src="https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit"
				onReady={() => resolve(turnstile)}
			/>
			<Suspense fallback={<CaptchaContentSkeleton />}>
				<CaptchaContent {...props} promise={promise} />
			</Suspense>
		</>
	);
};

export const Captcha = dynamic(() => Promise.resolve(_Captcha), {
	loading: () => <CaptchaContentSkeleton />,
	ssr: false
});
