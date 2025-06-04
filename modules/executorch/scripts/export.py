#!/usr/bin/env python3
"""
Export a simple PyTorch model to ExecuTorch format with XNNPACK backend
"""

import os

from common_utils import ModelTrainer, SimpleLinearModel, get_model_path

try:
    import torch
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower
except ImportError:
    print("ExecuTorch or PyTorch not installed. Please install them first:")
    print("pip install torch executorch")
    exit(1)


def main():
    """Export model with XNNPACK backend optimization"""
    print("Exporting model with XNNPACK backend...")

    # Create and train model
    model = SimpleLinearModel(input_size=4, output_size=1).eval()
    trainer = ModelTrainer(model)

    # Load trained weights if available
    weights_path = get_model_path("simple_linear_weights.pth")
    if not trainer.load_weights(weights_path):
        print("Training new model...")
        trainer.train(num_epochs=100, verbose=True)
        trainer.save_weights(weights_path)

    # Generate sample input
    sample_inputs = (model.get_example_input(),)

    # Export with XNNPACK optimization
    try:
        et_program = to_edge_transform_and_lower(
            torch.export.export(model, sample_inputs), partitioner=[XnnpackPartitioner()]
        ).to_executorch()

        output_path = get_model_path("model_mv2_xnnpack.pte")
        with open(output_path, "wb") as f:
            f.write(et_program.buffer)

        print(f"✅ Model exported successfully to {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")

    except Exception as e:
        print(f"❌ Export failed: {e}")


if __name__ == "__main__":
    main()
