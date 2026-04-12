import { useCallback, useEffect, useState } from "react";

export function useLocalStorage<T>(
	key: string,
	{ initial, fallback }: { initial: T; fallback?: T }
) {
	const [value, setValue] = useState(initial);

	useEffect(() => {
		const raw = localStorage.getItem(key);
		setValue(raw ? (JSON.parse(raw) as T) : fallback ?? initial);
	}, [key, fallback, initial]);

	const set = useCallback(
		(newValue: T) => {
			window.localStorage.setItem(key, JSON.stringify(newValue));
			setValue(newValue);
		},
		[key]
	);

	return [value, set] as const;
}
