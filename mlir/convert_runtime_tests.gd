extends SceneTree

# Script to convert all copied runtime test scripts to StableHLO
# Run with: godot --headless --script mlir/convert_runtime_tests.gd

func _init():
	print("Converting runtime test scripts to StableHLO...")
	
	# Find all .gd files in tests_from_runtime directory
	var test_dir = "res://mlir/tests_from_runtime"
	var dir = DirAccess.open(test_dir)
	if dir == null:
		print("Failed to open directory: ", test_dir)
		quit()
		return
	
	var scripts = []
	dir.list_dir_begin()
	var file_name = dir.get_next()
	while file_name != "":
		if file_name.ends_with(".gd"):
			scripts.append(test_dir + "/" + file_name)
		file_name = dir.get_next()
	
	print("Found ", str(scripts.size()), " test script(s)")
	
	var converter = GDScriptToStableHLO.new()
	var total_converted = 0
	
	for script_path in scripts:
		if not ResourceLoader.exists(script_path):
			print("Skipping (not found): ", script_path)
			continue
		
		print("\n=== Processing: ", script_path, " ===")
		
		var script = load(script_path) as GDScript
		if script == null:
			print("Failed to load: ", script_path)
			continue
		
		var err = script.reload()
		if err != OK:
			print("Failed to reload: ", script_path, " (error: ", err, ")")
			continue
		
		# Get all functions
		var methods = script.get_script_method_list()
		var converted_count = 0
		
		for method_info in methods:
			var method_name = method_info["name"]
			
			if converter.can_convert_script_function(script, method_name):
				print("  Converting function: ", method_name)
				
				var stablehlo_text = converter.convert_script_function_to_stablehlo_text(script, method_name)
				if stablehlo_text.is_empty():
					print("    ERROR: Failed to generate StableHLO")
					continue
				
				# Save to file
				var base_name = script_path.get_file().get_basename()
				var method_name_str = String(method_name)
				var output_path = "res://mlir/tests_from_runtime/" + base_name + "_" + method_name_str + ".mlir.stablehlo"
				var result_path = converter.generate_mlir_file_from_script(script, method_name, output_path)
				
				if result_path.is_empty():
					print("    WARNING: Failed to save file")
				else:
					print("    Saved to: ", result_path)
					converted_count += 1
					total_converted += 1
			else:
				print("  Skipping (cannot convert): ", method_name)
		
		print("  Converted ", String(converted_count), " function(s) from this script")
	
	print("\n=== Conversion complete ===")
	print("Total functions converted: ", String(total_converted))
	quit()
