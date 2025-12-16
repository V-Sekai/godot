#!/usr/bin/env -S godot --headless --script
# Test ELF64 files using Godot Sandbox
# Usage: godot --headless --script tools/test_elf64_with_sandbox.gd <elf_file> [function_name] [args...]

@tool
extends SceneTree

func _init():
	var args = OS.get_cmdline_args()
	
	# Find the script name in arguments
	var script_name = "test_elf64_with_sandbox.gd"
	var script_index = -1
	for i in range(args.size()):
		if args[i].ends_with(script_name):
			script_index = i
			break
	
	# Parse arguments
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
	
	# Load ELF file
	var file = FileAccess.open(elf_path, FileAccess.READ)
	if file == null:
		print("Error: Cannot open ELF file: ", elf_path)
		quit(1)
		return
	
	var elf_data = file.get_buffer(file.get_length())
	file.close()
	
	if elf_data.is_empty():
		print("Error: ELF file is empty: ", elf_path)
		quit(1)
		return
	
	# Create ELFScript and load from file path
	# ELFScript loads from file paths directly
	var elf_script = ELFScript.new()
	elf_script.set_file(elf_path)
	
	# Create a Sandbox node to execute the ELF
	var sandbox = Sandbox.new()
	sandbox.set_program(elf_script)
	
	# Wait for initialization
	await get_tree().process_frame
	
	# If function name provided, call it
	if not function_name.is_empty():
		print("Calling function: ", function_name, " with args: ", function_args)
		var callable = sandbox.vmcallable(function_name, function_args)
		var result = callable.call()
		print("Result: ", result)
	else:
		print("ELF loaded successfully. Available functions:")
		# List available functions if possible
		var functions = elf_script.function_names
		for func in functions:
			print("  - ", func)
	
	quit(0)
