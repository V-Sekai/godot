extends Node

# Sample GDScript functions for StableHLO conversion testing

# Simple arithmetic function
func add_numbers(a: int, b: int) -> int:
	return a + b

# Function with variables and arithmetic
func calculate_area(width: float, height: float) -> float:
	var area = width * height
	return area

# Function with conditional logic
func max_value(x: int, y: int) -> int:
	if x > y:
		return x
	else:
		return y

# Function with loop (basic counting)
func sum_range(n: int) -> int:
	var sum = 0
	var i = 1
	while i <= n:
		sum = sum + i
		i = i + 1
	return sum

# Simple boolean logic
func is_even(n: int) -> bool:
	return n % 2 == 0

# Function with multiple operations
func process_data(data: Array) -> int:
	var result = 0
	for item in data:
		if item > 0:
			result = result + item
	return result