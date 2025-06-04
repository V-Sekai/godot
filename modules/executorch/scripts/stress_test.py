#!/usr/bin/env python3
"""
Unified script for PyTorch to ExecuTorch conversion and validation
"""

import argparse
import time

import numpy as np
import torch
from executorch.runtime import Runtime


# Model definition
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)


def setup_model():
    """Setup and export the model"""
    print("=== Setting up model ===")

    model = SimpleModel()
    example_input = torch.randn(1, 4)

    from executorch.exir import to_edge

    edge_program = to_edge(torch.export.export(model, (example_input,)))

    executorch_program = edge_program.to_executorch()

    with open("models/model.pte", "wb") as f:
        f.write(executorch_program.buffer)

    runtime = Runtime.get()
    program = runtime.load_program("models/model.pte")
    method = program.load_method("forward")

    print("✅ Model setup complete")
    return model, method


def run_single_test(model, method):
    """Run a single test with fixed input"""
    print("=== Running Single Test ===")

    # Test input
    test_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    # Run both models
    pytorch_out = model(test_input)
    executorch_out = method.execute([test_input])[0]

    print(f"PyTorch output: {pytorch_out}")
    print(f"ExecuTorch output: {executorch_out}")

    # Check equivalency
    diff = torch.abs(pytorch_out - executorch_out).max().item()
    match = torch.allclose(pytorch_out, executorch_out, rtol=1e-3, atol=1e-5)

    print(f"Difference: {diff:.6f}")
    print(f"Match: {match}")

    if match:
        print("✅ Single test passed")
    else:
        print("❌ Single test failed")

    return match


def stress_test_equivalency(model, method):
    """Test equivalency with multiple random inputs"""
    print("=== Stress Test with Random Inputs ===")

    torch.manual_seed(123)
    num_tests = 10
    passed = 0
    failed = 0

    for i in range(num_tests):
        # Generate random input
        test_input = torch.randn(1, 4)

        # Run both models
        pytorch_out = model(test_input)
        executorch_out = method.execute([test_input])[0]

        # Check equivalency
        diff = torch.abs(pytorch_out - executorch_out).max().item()
        match = torch.allclose(pytorch_out, executorch_out, rtol=1e-3, atol=1e-5)

        status = "✅ PASS" if match else "❌ FAIL"
        print(f"Test {i + 1:2d}: diff={diff:.6f}, match={match} {status}")

        if match:
            passed += 1
        else:
            failed += 1

    print("\n=== Summary ===")
    print(f"Total tests: {num_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {100 * passed / num_tests:.1f}%")

    if failed == 0:
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

    test_cases = [
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[-1.0, -2.0, -3.0, -4.0]]),
        torch.randn(1, 4),
        torch.randn(1, 4) * 10,
    ]

    all_passed = True
    for i, test_input in enumerate(test_cases):
        pytorch_out = model(test_input)
        executorch_out = method.execute([test_input])[0]

        diff = torch.abs(pytorch_out - executorch_out).max().item()
        match = diff < tolerance

        status = "✅ PASS" if match else "❌ FAIL"
        print(f"Test case {i + 1}: diff={diff:.8f} {status}")

        if not match:
            all_passed = False

    if all_passed:
        print("🎉 All validation tests passed!")
    else:
        print("❌ Some validation tests failed!")

    return all_passed


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

        print(f"PyTorch output: {pytorch_out.item():.6f}")
        print(f"ExecuTorch output: {executorch_out.item():.6f}")

        # Check equivalency
        diff = torch.abs(pytorch_out - executorch_out).max().item()
        match = torch.allclose(pytorch_out, executorch_out, rtol=1e-3, atol=1e-5)

        print(f"Difference: {diff:.8f}")
        print(f"Match: {match}")

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
