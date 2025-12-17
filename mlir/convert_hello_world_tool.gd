extends EditorScript

# Editor script to convert hello_world.gd to StableHLO
# Run from: File > Run Script (or use the command line)

func _run():
	print("Converting hello_world.gd to StableHLO...")
	
	# Load the hello_world script
	var script_path = "res://mlir/hello_world.gd"
	var script = load(script_path) as GDScript
	
	if script == null:
		print("Failed to load script: " + script_path)
		return
	
	# Reload to ensure it's compiled
	var err = script.reload()
	if err != OK:
		print("Failed to reload script: " + str(err))
		return
	
	# Try to access the function through the script's internal API
	# Note: This requires access to GDScriptFunction which may not be available from GDScript
	# We'll need to use the C++ API through an editor plugin or tool
	
	print("Script loaded successfully")
	print("Function names in script:")
	
	# Get available functions
	var functions = script.get_script_method_list()
	for func_info in functions:
		print("  - " + func_info.name)
	
	# Try to use GDScriptToStableHLO if available
	# Note: Static methods may not be exposed to GDScript
	if ClassDB.class_exists("GDScriptToStableHLO"):
		print("\nGDScriptToStableHLO class is available")
		# Try to call the static method (may not work from GDScript)
		# var converter = GDScriptToStableHLO.new()
		# var stablehlo = converter.convert_function_to_stablehlo_text(function)
	else:
		print("\nGDScriptToStableHLO class not found")
		print("Conversion requires C++ API access")
		print("Use the test framework or create an editor plugin")
