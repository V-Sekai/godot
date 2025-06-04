# ExecuTorch Scripts - Usage Guide

This directory contains Python scripts and automation tools for converting PyTorch models to ExecuTorch format and testing equivalency between the two implementations.

## Prerequisites

- Python 3.8+
- [just](https://github.com/casey/just) command runner
- PyTorch and ExecuTorch (will be installed automatically)

## Quick Start

1. **Install dependencies and setup:**
   ```bash
   just install
   just setup
   ```

2. **Convert PyTorch model to ExecuTorch:**
   ```bash
   just convert
   ```

3. **Test equivalency:**
   ```bash
   just test-equivalency
   ```

4. **Run full pipeline:**
   ```bash
   just all
   ```

## Available Commands

### Core Workflow
- `just install` - Install Python dependencies (torch, torchvision, executorch)
- `just setup` - Create models directory
- `just convert` - Convert PyTorch model to ExecuTorch format
- `just test-equivalency` - Run basic equivalency test
- `just all` - Run complete pipeline (setup → convert → test → validate)

### Testing & Validation
- `just validate` - Comprehensive validation of model files and outputs
- `just benchmark` - Performance comparison between PyTorch and ExecuTorch
- `just stress-test` - Test with 10 random inputs to verify consistency
- `just test-custom INPUT="1.0,2.0,3.0,4.0"` - Test with custom input values
- `just quick-test` - Fast equivalency check (assumes model exists)

### Utilities
- `just info` - Show model files and available scripts
- `just clean` - Remove generated model files
- `just help` - Show available commands

## Scripts Overview

### Main Scripts
- **`convert_model.py`** - Converts a simple linear regression model from PyTorch to ExecuTorch format
- **`run.py`** - Loads both PyTorch and ExecuTorch models and compares their outputs

### Testing Scripts
- **`benchmark.py`** - Performance comparison (1000 inference runs)
- **`stress_test.py`** - Equivalency testing with random inputs
- **`test_custom.py`** - Test with user-provided input values
- **`validate.py`** - Validate model files and run equivalency checks

## Expected Output

After running `just convert`, you should see:
```
models/
├── simple_linear.pte          # ExecuTorch model
└── simple_linear_weights.pth  # PyTorch weights
```

## Example Usage

### Basic Workflow
```bash
# Setup everything
just all

# Test with custom input
just test-custom INPUT="5.0,10.0,15.0,20.0"

# Run performance benchmark
just benchmark

# Stress test with random inputs
just stress-test
```

### Manual Steps
```bash
# Individual steps
just install
just setup
just convert
just validate
just benchmark
```

## What Each Test Validates

1. **Equivalency Test**: Ensures PyTorch and ExecuTorch produce identical outputs (within tolerance)
2. **Stress Test**: Validates consistency across multiple random inputs
3. **Benchmark**: Compares inference speed between implementations
4. **Custom Test**: Allows testing with specific input values

## Troubleshooting

### Common Issues

**"Model file not found"**
- Run `just convert` first to generate the ExecuTorch model

**"ExecuTorch not installed"**
- Run `just install` to install dependencies

**"Permission denied"**
- Ensure Python and pip are properly installed
- On macOS/Linux, you might need to use `pip3` instead of `pip`

**Equivalency test failures**
- Small numerical differences are expected due to floating-point precision
- Differences > 1e-5 may indicate conversion issues

### Expected Tolerances
- **Relative tolerance**: 1e-3 (0.1%)
- **Absolute tolerance**: 1e-5
- Most differences should be < 1e-6 for simple models

## Model Details

The test model is a simple linear regression with:
- **Input size**: 4 features
- **Output size**: 1 prediction
- **Architecture**: Single linear layer
- **Training**: 100 epochs on synthetic data

This provides a minimal but complete example of the PyTorch → ExecuTorch conversion pipeline.