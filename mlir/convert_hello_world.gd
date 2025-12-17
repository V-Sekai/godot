extends SceneTree

# Script to convert hello_world.gd to StableHLO
# Run with: godot --script convert_hello_world.gd --headless

func _init():
	print("Converting hello_world.gd to StableHLO...")
	
	# Load the hello_world script
	var script_path = "res://mlir/hello_world.gd"
	var script = load(script_path) as GDScript
	
	if script == null:
		print("Failed to load script: " + script_path)
		quit()
		return
	
	# Reload to ensure it's compiled
	var err = script.reload()
	if err != OK:
		print("Failed to reload script: " + str(err))
		quit()
		return
	
	print("Script loaded successfully")
	
	# Use GDScriptToStableHLO to convert
	var converter = GDScriptToStableHLO.new()
	
	# Check if function can be converted
	if converter.can_convert_script_function(script, "hello_world"):
		print("Function 'hello_world' can be converted to StableHLO")
		
		# Convert to StableHLO text
		var stablehlo_text = converter.convert_script_function_to_stablehlo_text(script, "hello_world")
		
		if stablehlo_text.is_empty():
			print("ERROR: Failed to generate StableHLO text")
		else:
			print("\n=== Generated StableHLO ===")
			print(stablehlo_text)
			print("============================\n")
			
			# Save to file
			var output_path = "res://mlir/hello_world_converted.mlir"
			var result_path = converter.generate_mlir_file_from_script(script, "hello_world", output_path)
			
			if result_path.is_empty():
				print("WARNING: Failed to save to file, but text was generated")
			else:
				print("Saved StableHLO to: " + result_path)
	else:
		print("ERROR: Function 'hello_world' cannot be converted to StableHLO")
		print("(May contain unsupported opcodes)")
	
	quit()
