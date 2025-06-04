#!/usr/bin/env python3
"""
Convert a simple PyTorch model to ExecuTorch format
This script demonstrates the conversion pipeline for the simple linear regression model
"""

import os
import sys
from common_utils import (
    SimpleLinearModel, 
    ModelTrainer, 
    ExecuTorchConverter, 
    setup_directories,
    get_model_path
)


def create_and_train_model():
    """Create and minimally train a simple model"""
    print("Creating simple linear regression model...")

    model = SimpleLinearModel()
    trainer = ModelTrainer(model)
    
    # Train the model
    trained_model = trainer.train(num_epochs=100, verbose=True)
    
    # Save the trained weights
    weights_path = get_model_path("simple_linear_weights.pth")
    trainer.save_weights(weights_path)

    return trained_model


def convert_to_executorch(model, output_path="simple_linear.pte"):
    """Convert PyTorch model to ExecuTorch format"""
    converter = ExecuTorchConverter()
    example_input = model.get_example_input()
    
    return converter.convert_model(model, example_input, output_path, verbose=True)


def main():
    """Main conversion pipeline"""
    print("PyTorch to ExecuTorch Conversion Script")
    print("=" * 40)

    setup_directories()
    output_path = get_model_path("simple_linear.pte")

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
