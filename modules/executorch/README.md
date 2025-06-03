<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Godot ExecuTorch Module](#godot-executorch-module)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage in Godot](#usage-in-godot)
    - [Basic Setup](#basic-setup)
    - [Running Inference](#running-inference)
    - [Using MCP Tools](#using-mcp-tools)
    - [Performance Monitoring](#performance-monitoring)
  - [Module Architecture](#module-architecture)
    - [Core Components](#core-components)
    - [Class Hierarchy](#class-hierarchy)
  - [Module Integration Benefits](#module-integration-benefits)
  - [Development](#development)
    - [Testing the Module](#testing-the-module)
  - [Deployment](#deployment)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Godot ExecuTorch Module

This project is currently in **early development** and is **NOT ready for
production use**.

A native C++ module for Godot Engine that provides ExecuTorch machine learning
inference capabilities with built-in MCP (Model Context Protocol) support.

**Current Status:**

- 🔴 **API is unstable** and subject to breaking changes
- 🔴 **Limited testing** - may contain bugs and memory leaks
- 🔴 **Incomplete features** - many planned features are not implemented
- 🔴 **No official releases** - use at your own risk
- 🔴 **Documentation may be outdated** or incorrect

**Do not use this in:**

- Production games or applications
- Commercial projects
- Critical systems

**This is for experimentation and development only!**

## Overview

This module integrates directly into Godot Engine as a C++ module (not a
GDExtension), providing seamless access to ExecuTorch models from within Godot
projects.

## Installation

## Usage in Godot

### Basic Setup

```gdscript
extends Node

var runtime: ExecuTorchRuntime

func _ready():
    # The class is automatically available after building the module
    runtime = ExecuTorchRuntime.new()

    # Load a model
    if runtime.load_model_from_file("res://models/linear_regression.pte"):
        print("Model loaded successfully!")
    else:
        print("Failed to load model")
```

### Running Inference

```gdscript
func run_prediction(input_value: float):
    if not runtime.is_model_loaded():
        print("No model loaded")
        return

    # Prepare input data
    var inputs = {
        "input_0": [input_value]
    }

    # Run inference
    var result = runtime.run_inference(inputs)

    if result.has("output_0"):
        var prediction = result["output_0"][0]
        print("Input: ", input_value, " -> Prediction: ", prediction)
        return prediction

    return null
```

### Using MCP Tools

```gdscript
func test_mcp_features():
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
```

### Performance Monitoring

```gdscript
func benchmark_model():
    runtime.reset_performance_stats()

    # Run multiple inferences
    for i in range(100):
        runtime.run_inference({"input_0": [float(i)]})

    print("Total inferences: ", runtime.get_total_inferences())
    print("Last inference time: ", runtime.get_last_inference_time(), "ms")
```

## Module Architecture

### Core Components

- **ExecuTorchRuntime**: Main interface class exposed to GDScript
- **ExecuTorchModel**: Internal model wrapper and inference engine
- **MCPServerInternal**: Built-in Model Context Protocol server
- **register_types**: Godot module registration and lifecycle

### Class Hierarchy

```
ExecuTorchRuntime (exposed to GDScript)
├── ExecuTorchModel (internal model handling)
└── MCPServerInternal (MCP protocol implementation)
```

## Module Integration Benefits

Compared to GDExtension, the C++ module approach provides:

- **Native Integration**: Direct access to Godot's core APIs
- **Better Performance**: No marshaling overhead between GDScript and C++
- **Simplified Distribution**: Built directly into the engine binary
- **Easier Debugging**: Full access to Godot's debugging infrastructure
- **Reduced Dependencies**: No external .so/.dll files needed

## Development

### Testing the Module

The module includes the same linear regression test case:

```gdscript
extends Node

func _ready():
    test_linear_regression()

func test_linear_regression():
    var runtime = ExecuTorchRuntime.new()

    # Test with known linear function: y = 2x + 3
    var test_cases = [
        {"input": 0.0, "expected": 3.0},   # 2*0 + 3 = 3
        {"input": 1.0, "expected": 5.0},   # 2*1 + 3 = 5
        {"input": 2.0, "expected": 7.0},   # 2*2 + 3 = 7
        {"input": -1.0, "expected": 1.0}   # 2*(-1) + 3 = 1
    ]

    for test_case in test_cases:
        var result = runtime.run_inference({
            "input_0": [test_case.input]
        })

        if result.has("output_0"):
            var actual = result["output_0"][0]
            var expected = test_case.expected
            var error = abs(actual - expected)

            print("Input: ", test_case.input)
            print("Expected: ", expected, ", Actual: ", actual)
            print("Error: ", error)
            print("✓ Test passed" if error < 0.1 else "✗ Test failed")
            print("---")
```

## Deployment

When distributing your Godot project:

1. **Export Templates**: Use export templates built with the module
2. **No Additional Files**: The module is built into the engine binary
3. **Cross-Platform**: Works on all platforms where Godot runs
4. **Model Files**: Include your `.pte` model files in the project

## Troubleshooting

### Common Issues

**Module not found:**

- Ensure the module is in `modules/executorch/` in Godot source
- Rebuild Godot completely after adding the module

**Compilation errors:**

- Check that all source files are present
- Verify C++17 support is available
- Update SCons if needed

**Runtime errors:**

- Check model file paths and formats
- Verify ExecuTorch model compatibility
- Enable verbose logging for debugging
