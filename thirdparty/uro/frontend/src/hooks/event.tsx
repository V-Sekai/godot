import {
	createContext,
	use,
	useCallback,
	useEffect,
	useId,
	useState,
	type FC,
	type PropsWithChildren
} from "react";

const EventContext = createContext(
	{} as {
		subscribers: Record<string, string>;
		listen: (id: string, event: string) => () => void;
	}
);

export function useEvent(event: string, _callback: () => void) {
	const { listen } = use(EventContext);
	const id = useId();

	useEffect(() => listen(id, event), [id, event, listen]);
}

export const EventProvider: FC<PropsWithChildren> = ({ children }) => {
	const [subscribers, setSubscribers] = useState<Record<string, string>>({});
	// console.log(subscribers);

	const listen = useCallback((id: string, event: string) => {
		setSubscribers((previous) => ({ ...previous, [id]: event }));
		return () =>
			setSubscribers((previous) => {
				const { [id]: _, ...rest } = previous;
				return rest;
			});
	}, []);

	return (
		<EventContext.Provider value={{ listen, subscribers }}>
			{children}
		</EventContext.Provider>
	);
};
