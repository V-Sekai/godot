# GDScript Hello World that can be converted to StableHLO
# This demonstrates printing "Hello, World!" using byte buffers and godot_syscalls

extends Node

# Simple hello world function that prints a message
# Note: In practice, print() would be converted to a custom_call to godot_syscall_print
func hello_world() -> int:
	# The string "Hello, World!\n" would be converted to a byte buffer tensor
	# and passed to godot_syscall_print via stablehlo.custom_call
	print("Hello, World!")
	return 0

# Alternative version showing explicit byte buffer creation
func hello_world_with_buffer() -> int:
	# Create byte buffer from string
	var message = "Hello, World!\n"
	var buffer = message.to_utf8_buffer()
	
	# Call syscall to print (this would be converted to stablehlo.custom_call)
	# In the sandbox module, this would invoke godot_syscall_print
	godot_syscall_print(buffer)
	
	return 0
