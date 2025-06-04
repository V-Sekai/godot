#!/usr/bin/env python3
"""
Convert a simple PyTorch model to ExecuTorch format
This script demonstrates the conversion pipeline for the simple linear regression model
"""

import os
import sys

import torch
import torch.nn as nn

try:
    from executorch.exir import to_edge
    from executorch.exir.backend.backend_api import to_backend
    from torch.export import export
except ImportError:
    print("ExecuTorch not installed. Please install it first:")
    print("pip install executorch torchvision")
    sys.exit(1)


class SimpleLinearModel(nn.Module):
    """Simple linear regression model for demonstration"""

    def __init__(self, input_size=4, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def create_and_train_model():
    """Create and minimally train a simple model"""
    print("Creating simple linear regression model...")

    model = SimpleLinearModel()

    torch.manual_seed(42)
    X = torch.randn(100, 4)
    y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(100, 1)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print("Training model...")
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    model.eval()

    # Save the trained model weights
    torch.save(model.state_dict(), "models/simple_linear_weights.pth")
    print("Model weights saved to models/simple_linear_weights.pth")

    return model


def convert_to_executorch(model, output_path="simple_linear.pte"):
    """Convert PyTorch model to ExecuTorch format"""
    print("Converting model to ExecuTorch format...")

    example_input = torch.randn(1, 4)

    try:
        print("Exporting model...")
        exported_program = export(model, (example_input,))

        print("Converting to Edge IR...")
        edge_program = to_edge(exported_program)

        print("Converting to ExecuTorch format...")
        executorch_program = edge_program.to_executorch()

        print(f"Saving to {output_path}...")
        with open(output_path, "wb") as f:
            f.write(executorch_program.buffer)

        print(f"Model successfully converted and saved to {output_path}")

        print("\n=== Model Information ===")
        print(f"File size: {os.path.getsize(output_path)} bytes")
        print(f"Input shape: {list(example_input.shape)}")

        with torch.no_grad():
            test_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            output = model(test_input)
            print(f"Test inference (PyTorch): {output.item():.4f}")

        return True

    except Exception as e:
        print(f"Conversion failed: {e}")
        return False


def main():
    """Main conversion pipeline"""
    print("PyTorch to ExecuTorch Conversion Script")
    print("=" * 40)

    os.makedirs("models", exist_ok=True)
    output_path = "models/simple_linear.pte"

    model = create_and_train_model()

    success = convert_to_executorch(model, output_path)

    if success:
        print("\n✅ Conversion completed successfully!")
        print(f"ExecuTorch model saved to: {output_path}")
        print("\nYou can now use this model in the Godot ExecuTorch Runtime:")
        print(f"  runtime.load_model_from_file('{output_path}')")
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
