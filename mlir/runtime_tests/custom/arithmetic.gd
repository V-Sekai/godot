# Arithmetic operations

extends Node

func subtract(a: int, b: int) -> int:
	return a - b

func divide(x: float, y: float) -> float:
	return x / y

func calculate_area(width: float, height: float) -> float:
	return width * height

func max_value(a: int, b: int) -> int:
	if a > b:
		return a
	else:
		return b
