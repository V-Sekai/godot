#!/usr/bin/env python3
"""
Unified script for PyTorch to ExecuTorch conversion and validation
"""

import argparse
import time

import numpy as np
import torch
from executorch.runtime import Runtime


#!/usr/bin/env python3
"""
Unified script for PyTorch to ExecuTorch conversion and validation
"""

import argparse
import os
import time

import numpy as np
import torch
from common_utils import (
    SimpleLinearModel,
    ModelTrainer,
    ModelValidator,
    ExecuTorchConverter,
    setup_directories,
    get_model_path,
    load_executorch_runtime
)


def setup_model():
    """Setup and export the model"""
    print("=== Setting up model ===")

    setup_directories()
    model_path = get_model_path("model.pte")

    # Create model
    model = SimpleLinearModel()
    trainer = ModelTrainer(model)
    
    # Load existing weights or train new model
    weights_path = get_model_path("stress_test_weights.pth")
    if not trainer.load_weights(weights_path):
        print("Training new model...")
        trainer.train(num_epochs=100, verbose=False)
        trainer.save_weights(weights_path)

    # Convert to ExecuTorch if needed
    if not os.path.exists(model_path):
        converter = ExecuTorchConverter()
        success = converter.convert_model(model, model.get_example_input(), model_path, verbose=False)
        if not success:
            raise RuntimeError("Failed to convert model")

    # Load ExecuTorch runtime
    runtime, program, method = load_executorch_runtime(model_path)
    if method is None:
        raise RuntimeError("Failed to load ExecuTorch runtime")

    print("✅ Model setup complete")
    return model, method


def run_single_test(model, method):
    """Run a single test with fixed input"""
    print("=== Running Single Test ===")

    validator = ModelValidator()
    test_input = model.get_test_input()

    # Run both models
    pytorch_out = model(test_input)
    executorch_out = method.execute([test_input])[0]

    match = validator.compare_outputs(pytorch_out, executorch_out)

    if match:
        print("✅ Single test passed")
    else:
        print("❌ Single test failed")

    return match


def stress_test_equivalency(model, method):
    """Test equivalency with multiple random inputs"""
    print("=== Stress Test with Random Inputs ===")

    validator = ModelValidator()
    torch.manual_seed(123)
    num_tests = 10
    passed = 0

    for i in range(num_tests):
        # Generate random input
        test_input = torch.randn(1, 4)

        # Run both models
        pytorch_out = model(test_input)
        executorch_out = method.execute([test_input])[0]

        # Check equivalency
        match = validator.compare_outputs(pytorch_out, executorch_out, verbose=False)
        diff = torch.abs(pytorch_out - executorch_out).max().item()

        status = "✅ PASS" if match else "❌ FAIL"
        print(f"Test {i + 1:2d}: diff={diff:.6f}, match={match} {status}")

        if match:
            passed += 1

    print("\n=== Summary ===")
    print(f"Total tests: {num_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {num_tests - passed}")
    print(f"Success rate: {100 * passed / num_tests:.1f}%")

    if passed == num_tests:
        print("🎉 All tests passed! Models are equivalent.")
    else:
        print("⚠️ Some tests failed. Check model consistency.")


def benchmark_performance(model, method, num_runs=100):
    """Benchmark performance between PyTorch and ExecuTorch"""
    print("=== Performance Benchmark ===")

    # Warm up
    test_input = torch.randn(1, 4)
    for _ in range(10):
        _ = model(test_input)
        _ = method.execute([test_input])

    # Benchmark PyTorch
    torch_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(test_input)
        end = time.perf_counter()
        torch_times.append(end - start)

    # Benchmark ExecuTorch
    executorch_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = method.execute([test_input])
        end = time.perf_counter()
        executorch_times.append(end - start)

    # Calculate statistics
    torch_mean = np.mean(torch_times) * 1000  # Convert to ms
    torch_std = np.std(torch_times) * 1000
    executorch_mean = np.mean(executorch_times) * 1000
    executorch_std = np.std(executorch_times) * 1000

    speedup = torch_mean / executorch_mean if executorch_mean > 0 else 0

    print(f"PyTorch:    {torch_mean:.3f} ± {torch_std:.3f} ms")
    print(f"ExecuTorch: {executorch_mean:.3f} ± {executorch_std:.3f} ms")
    print(f"Speedup:    {speedup:.2f}x")

    if speedup > 1:
        print(f"✅ ExecuTorch is {speedup:.2f}x faster")
    else:
        print(f"⚠️ PyTorch is {1 / speedup:.2f}x faster")


def validate_outputs(model, method, tolerance=1e-5):
    """Validate that outputs match within tolerance"""
    print("=== Output Validation ===")

    validator = ModelValidator(tolerance=tolerance)
    return validator.validate_equivalency(model, method, verbose=True)


def test_custom_input(model, method, input_values):
    """Test with custom input values"""
    print("=== Custom Input Test ===")

    try:
        # Parse input values
        values = [float(x.strip()) for x in input_values.split(",")]
        if len(values) != 4:
            print("❌ Error: Expected 4 input values")
            return False

        test_input = torch.tensor([values])
        print(f"Input: {values}")

        # Run both models
        pytorch_out = model(test_input)
        executorch_out = method.execute([test_input])[0]

        # Use validator for comparison
        validator = ModelValidator()
        match = validator.compare_outputs(pytorch_out, executorch_out)

        if match:
            print("✅ Custom test passed")
        else:
            print("❌ Custom test failed")

        return match

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main function with command line options"""
    parser = argparse.ArgumentParser(description="PyTorch to ExecuTorch conversion and testing")
    parser.add_argument(
        "--mode",
        choices=["single", "stress", "benchmark", "validate", "custom", "both"],
        default="both",
        help="Test mode: single test, stress test, benchmark, validate, custom input, or both",
    )
    parser.add_argument("--input", type=str, help="Custom input values (comma-separated, for custom mode)")
    args = parser.parse_args()

    # Setup model
    model, method = setup_model()

    # Run tests based on mode
    if args.mode in ["single", "both"]:
        run_single_test(model, method)

    if args.mode in ["stress", "both"]:
        if args.mode == "both":
            print()  # Add spacing between tests
        stress_test_equivalency(model, method)

    if args.mode == "benchmark":
        benchmark_performance(model, method)

    if args.mode == "validate":
        validate_outputs(model, method)

    if args.mode == "custom":
        if args.input:
            test_custom_input(model, method, args.input)
        else:
            print("❌ Error: --input required for custom mode")
            print("Example: python stress_test.py --mode custom --input '1.0,2.0,3.0,4.0'")


if __name__ == "__main__":
    main()
