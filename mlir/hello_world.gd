# GDScript Hello World that can be converted to StableHLO
# This demonstrates working with string constants

extends Node

# Hello World function using string constant
# The string "Hello, World!" is stored as a constant
# Note: String operations like .length() may not be fully supported yet
# This version uses the string as a constant value
func hello_world() -> String:
	# Return the "Hello, World!" string constant
	# In StableHLO, this would be converted to a byte buffer tensor
	return "Hello, World!"
