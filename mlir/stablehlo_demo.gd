extends SceneTree

# GDScript to StableHLO demonstration script
# Run with: godot --script stablehlo_demo.gd --headless

func _init():
	print("GDScript to StableHLO Generator Demo")
	print("=====================================")

	# Test basic arithmetic function
	test_function("add_numbers", "
func add_numbers(a: int, b: int) -> int:
	return a + b
")

	# Test conditional logic
	test_function("max_value", "
func max_value(x: int, y: int) -> int:
	if x > y:
		return x
	else:
		return y
")

	# Test variable assignment and arithmetic
	test_function("calculate_area", "
func calculate_area(width: float, height: float) -> float:
	var area = width * height
	return area
")

	# Test boolean logic
	test_function("is_even", "
func is_even(n: int) -> bool:
	return n % 2 == 0
")

	# Test simple loop (if supported)
	test_function("sum_range", "
func sum_range(n: int) -> int:
	var sum = 0
	var i = 1
	while i <= n:
		sum = sum + i
		i = i + 1
	return sum
")

	print("Demo completed!")
	quit()

func test_function(func_name: String, gdscript_code: String):
	print("\\n--- Testing function: " + func_name + " ---")

	# Create a temporary script
	var script = GDScript.new()
	script.source_code = gdscript_code

	var err = script.reload()
	if err != OK:
		print("Failed to reload script for " + func_name + ": " + str(err))
		return

	# Get the function
	var functions = script.get_script_method_list()
	var target_func = null

	for func_info in functions:
		if func_info.name == func_name:
			target_func = func_info
			break

	if target_func == null:
		print("Function " + func_name + " not found in script")
		return

	print("Function found: " + str(target_func))

	# Try to access the GDScriptToStableHLO functionality
	# Note: This is a demo - the actual conversion would need to be done
	# through the C++ GDScriptToStableHLO class
	print("Note: Actual StableHLO conversion requires C++ API access")
	print("GDScript code:")
	print(gdscript_code.strip_edges())

	# Show what the StableHLO might look like (conceptual)
	show_conceptual_stablehlo(func_name, gdscript_code)

func show_conceptual_stablehlo(func_name: String, gdscript_code: String):
	print("\\nConceptual StableHLO representation:")
	print("module {")

	if "add_numbers" in func_name:
		print("  func.func @add_numbers(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {")
		print("    %0 = stablehlo.add %arg0, %arg1 : tensor<i32>")
		print("    return %0 : tensor<i32>")
		print("  }")
	elif "max_value" in func_name:
		print("  func.func @max_value(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {")
		print("    %0 = stablehlo.maximum %arg0, %arg1 : tensor<i32>")
		print("    return %0 : tensor<i32>")
		print("  }")
	elif "calculate_area" in func_name:
		print("  func.func @calculate_area(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {")
		print("    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f32>")
		print("    return %0 : tensor<f32>")
		print("  }")
	elif "is_even" in func_name:
		print("  func.func @is_even(%arg0: tensor<i32>) -> tensor<i1> {")
		print("    %c2 = arith.constant dense<2> : tensor<i32>")
		print("    %0 = arith.remsi %arg0, %c2 : tensor<i32>")
		print("    %c0 = arith.constant dense<0> : tensor<i32>")
		print("    %1 = arith.cmpi eq, %0, %c0 : tensor<i1>")
		print("    return %1 : tensor<i1>")
		print("  }")
	elif "sum_range" in func_name:
		print("  func.func @sum_range(%arg0: tensor<i32>) -> tensor<i32> {")
		print("    // Loop-based accumulation (simplified)")
		print("    %c0 = arith.constant dense<0> : tensor<i32>")
		print("    %c1 = arith.constant dense<1> : tensor<i32>")
		print("    // Loop implementation would be more complex")
		print("    return %c0 : tensor<i32>")
		print("  }")
	else:
		print("  // StableHLO representation not implemented for this function type")

	print("}")