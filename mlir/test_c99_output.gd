@tool
extends EditorScript

func _run():
	print("=".repeat(80))
	print("GDScript to C99 Output Test")
	print("=".repeat(80))
	
	# Test with add_numbers.gd
	var script_path = "res://mlir/runtime_tests/custom/add_numbers.gd"
	var script = load(script_path) as GDScript
	if not script:
		print("ERROR: Failed to load script: ", script_path)
		return
	
	# Reload to ensure compiled
	var err = script.reload()
	if err != OK:
		print("ERROR: Failed to reload script: ", err)
		return
	
	# Get functions
	var functions = script.get_member_functions()
	print("\nFound ", functions.size(), " functions")
	
	for func_name in functions:
		print("\n" + "-".repeat(80))
		print("Function: ", func_name)
		print("-".repeat(80))
		
		# Check if can convert
		var can_convert = GDScriptToC99.can_convert_script(script, func_name)
		print("Can convert: ", can_convert)
		
		if can_convert:
			# Generate C99
			var c99_code = GDScriptToC99.generate_c99_from_script(script, func_name)
			print("\nGenerated C99 code:")
			print("=".repeat(80))
			print(c99_code)
			print("=".repeat(80))
			
			# Write to file
			var output_file = "res://mlir/runtime_tests/custom/" + func_name + ".c99"
			var file = FileAccess.open(output_file, FileAccess.WRITE)
			if file:
				file.store_string(c99_code)
				file.close()
				print("\nWritten to: ", output_file)
			else:
				print("ERROR: Failed to write to file: ", output_file)
