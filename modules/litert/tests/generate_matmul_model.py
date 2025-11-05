#!/usr/bin/env python3
"""
Generate a simple matrix multiplication TFLite model for testing.
This creates a 2x3 * 3x2 = 2x2 matrix multiplication model.
"""

import tensorflow as tf

# Create a simple model: matmul(input, weights) where:
# input: [2, 3] (batch_size=2, features=3)
# weights: [3, 2] (fixed weights)
# output: [2, 2]

# Create the model
input_layer = tf.keras.layers.Input(shape=(3,), name="input")
weights = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
dense_layer = tf.keras.layers.Dense(2, use_bias=False, name="matmul")
dense_layer.build((None, 3))
dense_layer.set_weights([weights.numpy()])
output = dense_layer(input_layer)

model = tf.keras.Model(inputs=input_layer, outputs=output)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
]
converter.optimizations = []  # No optimizations for simplicity

tflite_model = converter.convert()

# Save the model to tests/data/ directory
import os
output_dir = os.path.join(os.path.dirname(__file__), "../../../tests/data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "matmul_model.tflite")
with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ Generated {output_path}")
print(f"   Model size: {len(tflite_model)} bytes")
print("")
print("Model structure:")
print("  Input: [2, 3] (can be batched)")
print("  Output: [2, 2]")
print("")
print("Expected weights:")
print("  [[1.0, 2.0],")
print("   [3.0, 4.0],")
print("   [5.0, 6.0]]")
print("")
print("Test input: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]")
print("Expected output:")
print("  [[1*1+2*3+3*5, 1*2+2*4+3*6],")
print("   [4*1+5*3+6*5, 4*2+5*4+6*6]]")
print("  = [[22.0, 28.0], [49.0, 64.0]]")
