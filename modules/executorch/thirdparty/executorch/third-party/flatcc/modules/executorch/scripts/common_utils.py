#!/usr/bin/env python3
"""
Common utilities and shared code for ExecuTorch model conversion and testing
"""

import os
import sys
from typing import Tuple

import torch
import torch.nn as nn


class SimpleLinearModel(nn.Module):
    """Simple linear regression model for demonstration and testing"""

    def __init__(self, input_size: int = 4, output_size: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_example_input(self, batch_size: int = 1) -> torch.Tensor:
        """Generate example input tensor for this model"""
        return torch.randn(batch_size, self.input_size)

    def get_test_input(self) -> torch.Tensor:
        """Get standard test input for consistent testing"""
        return torch.tensor([[1.0, 2.0, 3.0, 4.0]])


class ModelTrainer:
    """Utility class for training simple models"""

    def __init__(self, model: nn.Module, learning_rate: float = 0.01, seed: int = 42):
        self.model = model
        self.learning_rate = learning_rate
        self.seed = seed
        torch.manual_seed(seed)

    def create_synthetic_data(self, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create synthetic training data for linear regression"""
        X = torch.randn(num_samples, 4)
        y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(num_samples, 1)
        return X, y

    def train(self, num_epochs: int = 100, verbose: bool = True) -> nn.Module:
        """Train the model with synthetic data"""
        X, y = self.create_synthetic_data()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        if verbose:
            print("Training model...")

        self.model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            predictions = self.model(X)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

        self.model.eval()
        return self.model

    def save_weights(self, path: str) -> None:
        """Save model weights to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load_weights(self, path: str) -> bool:
        """Load model weights from file"""
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, weights_only=True))
            print(f"Loaded trained weights from {path}")
            return True
        else:
            print(f"Warning: Weights file not found at {path}")
            return False


class ExecuTorchConverter:
    """Utility class for converting PyTorch models to ExecuTorch format"""

    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if ExecuTorch dependencies are available"""
        try:
            from executorch.exir import to_edge
            from torch.export import export
        except ImportError:
            print("ExecuTorch not installed. Please install it first:")
            print("pip install executorch torchvision")
            sys.exit(1)

    def convert_model(
        self, model: nn.Module, example_input: torch.Tensor, output_path: str, verbose: bool = True
    ) -> bool:
        """Convert PyTorch model to ExecuTorch format"""
        try:
            from executorch.exir import to_edge
            from torch.export import export

            if verbose:
                print("Converting model to ExecuTorch format...")

            # Export to PyTorch IR
            if verbose:
                print("Exporting model...")
            exported_program = export(model, (example_input,))

            # Convert to Edge IR
            if verbose:
                print("Converting to Edge IR...")
            edge_program = to_edge(exported_program)

            # Convert to ExecuTorch format
            if verbose:
                print("Converting to ExecuTorch format...")
            executorch_program = edge_program.to_executorch()

            # Save to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if verbose:
                print(f"Saving to {output_path}...")
            with open(output_path, "wb") as f:
                f.write(executorch_program.buffer)

            if verbose:
                print(f"Model successfully converted and saved to {output_path}")
                self._print_model_info(model, example_input, output_path)

            return True

        except Exception as e:
            print(f"Conversion failed: {e}")
            return False

    def _print_model_info(self, model: nn.Module, example_input: torch.Tensor, output_path: str) -> None:
        """Print information about the converted model"""
        print("\n=== Model Information ===")
        print(f"File size: {os.path.getsize(output_path)} bytes")
        print(f"Input shape: {list(example_input.shape)}")

        # Test inference
        model.eval()
        with torch.no_grad():
            if hasattr(model, "get_test_input"):
                test_input = model.get_test_input()
            else:
                test_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

            output = model(test_input)
            print(f"Test inference (PyTorch): {output.item():.4f}")


class ModelValidator:
    """Utility class for validating model outputs and performance"""

    def __init__(self, tolerance: float = 1e-5):
        self.tolerance = tolerance

    def compare_outputs(self, pytorch_out: torch.Tensor, executorch_out: torch.Tensor, verbose: bool = True) -> bool:
        """Compare outputs from PyTorch and ExecuTorch models"""
        diff = torch.abs(pytorch_out - executorch_out).max().item()
        match = torch.allclose(pytorch_out, executorch_out, rtol=1e-3, atol=1e-5)

        if verbose:
            print(f"PyTorch output: {pytorch_out}")
            print(f"ExecuTorch output: {executorch_out}")
            print(f"Difference: {diff:.6f}")
            print(f"Match: {match}")

        return match

    def validate_equivalency(
        self, pytorch_model: nn.Module, executorch_method, test_inputs: list = None, verbose: bool = True
    ) -> bool:
        """Validate equivalency between PyTorch and ExecuTorch models"""
        if test_inputs is None:
            test_inputs = [
                torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[-1.0, -2.0, -3.0, -4.0]]),
                torch.randn(1, 4),
                torch.randn(1, 4) * 10,
            ]

        all_passed = True
        pytorch_model.eval()

        for i, test_input in enumerate(test_inputs):
            with torch.no_grad():
                pytorch_out = pytorch_model(test_input)
            executorch_out = executorch_method.execute([test_input])[0]

            diff = torch.abs(pytorch_out - executorch_out).max().item()
            match = diff < self.tolerance

            if verbose:
                status = "✅ PASS" if match else "❌ FAIL"
                print(f"Test case {i + 1}: diff={diff:.8f} {status}")

            if not match:
                all_passed = False

        if verbose:
            if all_passed:
                print("🎉 All validation tests passed!")
            else:
                print("❌ Some validation tests failed!")

        return all_passed


def setup_directories() -> None:
    """Create necessary directories for model files"""
    os.makedirs("models", exist_ok=True)


def get_model_path(filename: str) -> str:
    """Get full path for model file"""
    return os.path.join("models", filename)


def load_executorch_runtime(model_path: str):
    """Load ExecuTorch runtime and method"""
    try:
        from executorch.runtime import Runtime

        runtime = Runtime.get()
        program = runtime.load_program(model_path)
        method = program.load_method("forward")

        return runtime, program, method
    except ImportError:
        print("ExecuTorch runtime not available")
        return None, None, None
    except Exception as e:
        print(f"Failed to load ExecuTorch runtime: {e}")
        return None, None, None
