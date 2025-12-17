# Comprehensive GDScript AST Elements for StableHLO Conversion
# This file contains examples of all AST elements that can be converted to StableHLO

extends Node

# =============================================================================
# 1. RETURN STATEMENTS
# =============================================================================

# Simple return
func return_int() -> int:
	return 42

# Return with expression
func return_add(a: int, b: int) -> int:
	return a + b

# Return with variable
func return_variable() -> int:
	var x = 10
	return x

# =============================================================================
# 2. ASSIGNMENTS
# =============================================================================

# Variable assignment
func assign_variable() -> int:
	var x = 5
	var y = 10
	return x + y

# Assignment with expression
func assign_expression(a: int) -> int:
	var result = a * 2 + 1
	return result

# Multiple assignments
func multiple_assignments() -> int:
	var x = 1
	var y = 2
	var z = 3
	return x + y + z

# =============================================================================
# 3. CONSTANT ASSIGNMENTS
# =============================================================================

# Null assignment (conceptual)
func assign_null_concept() -> Variant:
	var x = null
	return x

# Boolean assignments
func assign_booleans() -> bool:
	var t = true
	var f = false
	return t and not f

# =============================================================================
# 4. ARITHMETIC OPERATIONS
# =============================================================================

# Addition
func arithmetic_add(a: int, b: int) -> int:
	return a + b

# Subtraction
func arithmetic_subtract(a: int, b: int) -> int:
	return a - b

# Multiplication
func arithmetic_multiply(a: int, b: int) -> int:
	return a * b

# Division
func arithmetic_divide(a: float, b: float) -> float:
	return a / b

# Complex arithmetic
func complex_arithmetic(a: int, b: int, c: int) -> int:
	return a + b * c - (a / 2)

# =============================================================================
# 5. COMPARISON OPERATIONS
# =============================================================================

# Greater than
func compare_greater(a: int, b: int) -> bool:
	return a > b

# Less than
func compare_less(a: int, b: int) -> bool:
	return a < b

# Greater equal
func compare_greater_equal(a: int, b: int) -> bool:
	return a >= b

# Less equal
func compare_less_equal(a: int, b: int) -> bool:
	return a <= b

# Equal
func compare_equal(a: int, b: int) -> bool:
	return a == b

# Not equal
func compare_not_equal(a: int, b: int) -> bool:
	return a != b

# =============================================================================
# 6. CONDITIONAL LOGIC
# =============================================================================

# Simple if-else
func conditional_simple(x: int) -> int:
	if x > 10:
		return 100
	else:
		return 0

# If without else
func conditional_if_only(x: int) -> int:
	if x > 5:
		return 1
	return 0

# Nested conditionals
func conditional_nested(x: int) -> int:
	if x > 10:
		if x > 20:
			return 200
		else:
			return 100
	else:
		return 0

# =============================================================================
# 7. MEMBER ACCESS (limited support)
# =============================================================================

# Simple member access (if supported)
func member_access_example() -> int:
	var obj = {"key": 42}
	return obj.key

# =============================================================================
# 8. FUNCTION CALLS (limited support)
# =============================================================================

# Built-in function calls
func builtin_call_example() -> float:
	return sin(1.57) # Approximately 1.0

# Custom function calls (if supported)
func custom_call_example() -> int:
	return abs(-5)

# =============================================================================
# 9. COMPLEX EXPRESSIONS
# =============================================================================

# Mixed operations
func mixed_operations(a: float, b: float) -> float:
	var result = a * b + (a - b) / 2.0
	return result

# Boolean logic with arithmetic
func boolean_arithmetic(x: int, y: int) -> int:
	if x > y and x > 0:
		return x * 2
	else:
		return y * 2

# =============================================================================
# 10. EDGE CASES AND LIMITATIONS
# =============================================================================

# Large numbers
func large_numbers() -> int:
	return 1000000 + 2000000

# Floating point precision
func float_precision() -> float:
	return 1.0 / 3.0 * 3.0

# Zero operations
func zero_operations() -> int:
	return 0 + 0 * 5 - 0

# Identity operations
func identity_operations(x: int) -> int:
	return x + 0 * 5 / 1