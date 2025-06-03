# ExecuTorchResource for Godot Engine

A comprehensive implementation of ExecuTorch (.pte) files as Godot Engine
resources, providing both high-level and low-level APIs for machine learning
inference.

## Overview

The ExecuTorchResource class makes PyTorch ExecuTorch models first-class
citizens in Godot Engine by implementing them as native Godot Resources. This
allows for seamless integration with Godot's resource system, including:

- Native .pte file loading and saving
- Resource Inspector integration
- Memory-efficient model management
- Both simplified and advanced APIs

## Architecture

### High-Level API (ExecuTorch Module Class)

The high-level interface uses ExecuTorch's Module class for simplified model
loading and execution:

```cpp
// High-level usage
auto resource = std::make_unique<ExecuTorchResource>();
resource->load_from_file("model.pte");
Dictionary result = resource->forward(inputs);
```

### Low-Level API (Custom Memory Management)

The low-level interface provides direct control over memory allocation, operator
placement, and execution:

```cpp
// Low-level usage
resource->configure_memory(ExecuTorchResource::MEMORY_POLICY_STATIC, 2 * 1024 * 1024);
resource->set_optimization_level(ExecuTorchResource::OPTIMIZATION_AGGRESSIVE);
resource->enable_profiling(true);
```

## Key Features

### 1. **Flexible Memory Management**

- **Auto Policy**: Automatic memory management (default)
- **Static Policy**: Pre-allocated memory pool for embedded systems
- **Custom Policy**: User-defined memory allocators

### 2. **Performance Optimization**

- **Optimization Levels**: None, Basic, Aggressive
- **Profiling Support**: Detailed performance metrics
- **Memory Monitoring**: Real-time allocation tracking

### 3. **Resource Integration**

- **Native Godot Resource**: Full Inspector support
- **PCK Integration**: Models can be embedded in game packages
- **Streaming**: Large models can be loaded from disk

### 4. **Developer-Friendly APIs**

- **Simple Forward Pass**: Dictionary-based input/output
- **Array Interface**: Direct array processing
- **Metadata Access**: Model introspection capabilities

## Usage Examples

### Basic Linear Regression

```cpp
#include "executorch_resource.h"

// Create and configure resource
auto model = std::make_unique<ExecuTorchResource>();
model->load_from_file("linear_regression.pte");

// Run inference: y = 2x + 3
Dictionary inputs;
inputs["input_0"] = std::vector<float>{1.0f};

Dictionary outputs = model->forward(inputs);
float result = outputs["output_0"][0]; // Expected: 5.0
```

### Advanced Configuration

```cpp
// Configure memory for embedded system
model->configure_memory(
    ExecuTorchResource::MEMORY_POLICY_STATIC,
    1024 * 1024  // 1MB static pool
);

// Enable aggressive optimizations
model->set_optimization_level(ExecuTorchResource::OPTIMIZATION_AGGRESSIVE);

// Enable detailed profiling
model->enable_profiling(true);

// Monitor performance
Dictionary memory_info = model->get_memory_info();
double inference_time = model->get_last_inference_time();
```

### Godot Integration (GDScript)

```gdscript
extends Node

var model: ExecuTorchResource

func _ready():
    model = load("res://models/my_model.pte")

    if model.is_loaded():
        var result = model.forward({"input_0": [2.5]})
        print("Prediction: ", result["output_0"][0])
```

## Implementation Details

### ExecuTorch C++ API Integration

The implementation leverages ExecuTorch's C++ APIs:

```cpp
// High-level Module class usage
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/method.h>

// Create module from .pte file
auto module = std::make_unique<torch::executor::Module>(file_path);

// Execute inference
auto outputs = module->forward({inputs});
```

### Memory Management

```cpp
// Low-level memory control
#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/core/memory_allocator.h>

// Configure custom allocator
auto allocator = std::make_unique<torch::executor::MemoryAllocator>(
    memory_pool, pool_size
);
```

### Performance Profiling

```cpp
// Enable detailed profiling
#include <executorch/runtime/core/event_tracer.h>

auto tracer = std::make_unique<torch::executor::EventTracer>();
tracer->start_profiling();
```

## Testing

The implementation includes comprehensive unit tests using doctest:

```bash
# Build and run tests
chmod +x build_and_test.sh
./build_and_test.sh
```

### Test Coverage

- **Resource Lifecycle**: Creation, loading, unloading
- **Memory Management**: All memory policies and configurations
- **Performance Tracking**: Inference timing and statistics
- **Linear Regression**: End-to-end mathematical validation
- **Error Handling**: Invalid inputs and edge cases

### Linear Regression Test Cases

The tests validate a linear regression model `y = 2x + 3`:

| Input | Expected Output | Test Case      |
| ----- | --------------- | -------------- |
| 0.0   | 3.0             | Zero input     |
| 1.0   | 5.0             | Unit input     |
| 2.0   | 7.0             | Double input   |
| -1.0  | 1.0             | Negative input |

## Building

### As Godot Module

1. Copy to Godot's modules directory:

   ```bash
   cp -r godot-executorch /path/to/godot/modules/executorch
   ```

2. Build Godot with the module:
   ```bash
   scons platform=linuxbsd target=template_debug
   ```

### Standalone Testing

```bash
# Compile and test
cd godot-executorch
chmod +x build_and_test.sh
./build_and_test.sh
```

## Integration with ExecuTorch

### Supported ExecuTorch Features

- **Model Loading**: .pte file format support
- **Inference Execution**: Forward pass execution
- **Memory Management**: Custom allocators and memory pools
- **Operator Support**: All ExecuTorch operators
- **Platform Support**: Mobile, embedded, desktop

### ExecuTorch Version Compatibility

This implementation is designed for ExecuTorch v0.1+ and supports:

- PyTorch 2.0+ model export
- Quantized models (INT8, FP16)
- Mobile-optimized operators
- Custom operators (if registered)

## Performance Characteristics

### Memory Usage

- **High-Level API**: Automatic memory management
- **Low-Level API**: User-controlled allocation
- **Static Pools**: Predictable memory usage for embedded systems

### Inference Speed

- **Optimization Levels**: Configurable performance vs. accuracy trade-offs
- **Profiling**: Detailed per-operator timing
- **Memory Locality**: Optimized data layout

### Platform Support

- **Desktop**: Linux, Windows, macOS
- **Mobile**: Android, iOS
- **Embedded**: ARM, RISC-V (with static memory)

## Future Enhancements

- **GPU Support**: Integration with ExecuTorch GPU backends
- **Quantization**: Runtime quantization support
- **Streaming**: Large model streaming and paging
- **Batch Processing**: Multi-sample inference batching
- **Custom Operators**: Easy registration of custom ops
