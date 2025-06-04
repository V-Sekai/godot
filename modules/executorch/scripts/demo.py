#!/usr/bin/env python3
"""
Simple training script using common utilities
"""

from common_utils import (
    SimpleLinearModel,
    ModelTrainer,
    ExecuTorchConverter,
    ModelValidator,
    setup_directories,
    get_model_path,
    load_executorch_runtime
)


def main():
    """Demonstration of the simplified workflow"""
    print("ExecuTorch Common Utilities Demo")
    print("=" * 32)

    # Setup
    setup_directories()

    # 1. Create and train model
    print("\n1. Creating and training model...")
    model = SimpleLinearModel(input_size=4, output_size=1)
    trainer = ModelTrainer(model, learning_rate=0.01)
    
    trained_model = trainer.train(num_epochs=50, verbose=True)
    weights_path = get_model_path("demo_weights.pth")
    trainer.save_weights(weights_path)

    # 2. Convert to ExecuTorch
    print("\n2. Converting to ExecuTorch...")
    converter = ExecuTorchConverter()
    model_path = get_model_path("demo_model.pte")
    
    success = converter.convert_model(
        trained_model, 
        trained_model.get_example_input(), 
        model_path, 
        verbose=True
    )
    
    if not success:
        print("❌ Conversion failed!")
        return

    # 3. Load and validate
    print("\n3. Loading ExecuTorch runtime and validating...")
    runtime, program, method = load_executorch_runtime(model_path)
    
    if method is None:
        print("❌ Failed to load ExecuTorch runtime!")
        return

    # 4. Validate equivalency
    validator = ModelValidator(tolerance=1e-5)
    all_passed = validator.validate_equivalency(trained_model, method, verbose=True)
    
    if all_passed:
        print("\n🎉 Demo completed successfully!")
        print("✅ Model trained, converted, and validated")
    else:
        print("\n⚠️ Demo completed with validation errors")

    print(f"\nFiles created:")
    print(f"  - Model weights: {weights_path}")
    print(f"  - ExecuTorch model: {model_path}")


if __name__ == "__main__":
    main()