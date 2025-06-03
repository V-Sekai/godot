# Test script for the ExecuTorch Godot module
extends Node

class_name ExecuTorchModuleTest

func _ready():
    print("=== ExecuTorch Godot Module Test ===")
    test_linear_regression()

func test_linear_regression():
    print("\n--- Linear Regression Test ---")

    # Create runtime instance (this will be available after building the module)
    var runtime = ExecuTorchRuntime.new()

    # Test with known linear function: y = 2x + 3
    var test_cases = [
        {"input": 0.0, "expected": 3.0, "name": "Zero"},
        {"input": 1.0, "expected": 5.0, "name": "Unit"},
        {"input": 2.0, "expected": 7.0, "name": "Double"},
        {"input": -1.0, "expected": 1.0, "name": "Negative"}
    ]

    var passed_tests = 0
    var total_tests = test_cases.size()

    print("Running ", total_tests, " test cases...")

    for test_case in test_cases:
        print("\nTest: ", test_case.name)
        print("Input: ", test_case.input)

        var inputs = {
            "input_0": [test_case.input]
        }

        # Run inference using the module
        var result = runtime.run_inference(inputs)

        if result.has("output_0"):
            var actual = result["output_0"][0]
            var expected = test_case.expected
            var error = abs(actual - expected)

            print("Expected: ", expected)
            print("Actual: ", actual)
            print("Error: ", error)

            if error < 0.1:
                print("✓ PASSED")
                passed_tests += 1
            else:
                print("✗ FAILED")
        else:
            print("✗ FAILED - No output")

    print("\n--- Test Summary ---")
    print("Passed: ", passed_tests, "/", total_tests)
    print("Success rate: ", float(passed_tests) / float(total_tests) * 100.0, "%")

    # Test MCP features
    test_mcp_features(runtime)

    # Test performance monitoring
    test_performance_monitoring(runtime)

func test_mcp_features(runtime: ExecuTorchRuntime):
    print("\n--- MCP Features Test ---")

    # List available tools
    var tools = runtime.list_mcp_tools()
    print("Available MCP tools: ", tools)

    # Get model information
    var model_info = runtime.get_model_info()
    print("Model info: ", model_info)

    # Health check
    var health = runtime.health_check()
    print("Health status: ", health)

    # Direct tool call
    var tool_result = runtime.call_mcp_tool("run_inference", {
        "input_0": [2.5]
    })
    print("Tool result: ", tool_result)

func test_performance_monitoring(runtime: ExecuTorchRuntime):
    print("\n--- Performance Monitoring Test ---")

    runtime.reset_performance_stats()

    # Run multiple inferences
    var num_tests = 10
    print("Running ", num_tests, " inferences...")

    for i in range(num_tests):
        runtime.run_inference({"input_0": [float(i)]})

    print("Total inferences: ", runtime.get_total_inferences())
    print("Last inference time: ", runtime.get_last_inference_time(), "ms")

    print("\n✅ All module tests completed!")
