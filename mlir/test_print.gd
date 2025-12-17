# Test print function with vcall

extends Node

func print_hello() -> int:
	print("Hello from print!")
	return 0

func print_number(n: int) -> int:
	print(n)
	return 0

func print_string(s: String) -> int:
	print(s)
	return 0
