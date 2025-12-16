#!/usr/bin/env -S godot --headless --script
# Test ELF64 files using Godot Sandbox
# Usage: godot --headless --script tools/test_elf64_with_sandbox.gd <elf_file> [function_name] [args...]

@tool
extends SceneTree

func _init():
	# Parse arguments
	var args = OS.get_cmdline_args()
	var script_name = "test_elf64_with_sandbox.gd"
	var script_index = -1
	for i in range(args.size()):
		if args[i].ends_with(script_name):
			script_index = i
			break
	
	var elf_path = ""
	var function_name = ""
	var function_args = []
	
	if script_index >= 0:
		var i = script_index + 1
		while i < args.size():
			var arg = args[i]
			if not arg.begins_with("--"):
				if elf_path.is_empty():
					elf_path = arg
				elif function_name.is_empty():
					function_name = arg
				else:
					function_args.append(arg)
			i += 1
	
	if elf_path.is_empty():
		print("Usage: godot --headless --script tools/test_elf64_with_sandbox.gd <elf_file> [function_name] [args...]")
		quit(1)
		return
	
	# Check if Sandbox module is available
	if not ClassDB.class_exists("Sandbox"):
		print("Error: Sandbox module is not available.")
		print("Note: Sandbox module cannot be built with ASAN enabled.")
		print("To test ELF files with Sandbox, build without ASAN:")
		print("  scons target=editor use_asan=no")
		print("")
		print("ELF file compiled successfully: ", elf_path)
		print("File size: ", FileAccess.get_file_as_bytes(elf_path).size(), " bytes")
		quit(1)
		return
	
	# Create a root node to hold the sandbox
	var root = Node.new()
	get_root().add_child(root)
	
	# Create ELFScript and load from file path
	var elf_script = ClassDB.instantiate("ELFScript")
	elf_script.set_file(elf_path)
	
	# Create a Sandbox node to execute the ELF
	var sandbox = ClassDB.instantiate("Sandbox")
	root.add_child(sandbox)
	sandbox.set_program(elf_script)
	
	# Wait for sandbox to initialize - use call_deferred to process next frame
	call_deferred("_execute_test", sandbox, elf_script, function_name, function_args)

func _execute_test(sandbox, elf_script, function_name, function_args):
	# Wait one more frame for full initialization
	await process_frame
	
	# Check if sandbox has the function
	if not function_name.is_empty():
		if not sandbox.has_function(function_name):
			print("Error: Function '", function_name, "' not found in ELF")
			print("Available functions:")
			var functions = elf_script.function_names
			for fn_name in functions:
				print("  - ", fn_name)
			quit(1)
			return
		
		# Convert string args to integers if they look like numbers
		var converted_args = []
		for arg in function_args:
			if arg.is_valid_int():
				converted_args.append(arg.to_int())
			elif arg.is_valid_float():
				converted_args.append(arg.to_float())
			else:
				converted_args.append(arg)
		
		print("Calling function: ", function_name, " with args: ", converted_args)
		var callable = sandbox.vmcallable(function_name, converted_args)
		var result = callable.call()
		print("Result: ", result)
		quit(0)
	else:
		print("ELF loaded successfully. Available functions:")
		# List available functions
		var functions = elf_script.function_names
		for fn_name in functions:
			print("  - ", fn_name)
		quit(0)
