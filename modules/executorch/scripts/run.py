import torch
import torch.nn as nn
from executorch.runtime import Runtime
from typing import List
import os

# Simple Linear Regression Model (same as in export.py)
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size=4, output_size=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

runtime = Runtime.get()

# Use the same test input for consistent comparison
torch.manual_seed(42)
input_tensor: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

program = runtime.load_program("model_xnnpack.pte")
method = program.load_method("forward")
output: List[torch.Tensor] = method.execute([input_tensor])
print("Run successfully via executorch")

# Create eager reference model with same weights for comparison
eager_reference_model = LinearRegressionModel(input_size=4, output_size=1).eval()

# Load the same trained weights
weights_path = "../../models/simple_linear_weights.pth"
if os.path.exists(weights_path):
    eager_reference_model.load_state_dict(torch.load(weights_path, weights_only=True))
    print(f"Loaded trained weights from {weights_path}")
else:
    print("Warning: Using random weights. Results may not match.")

eager_reference_output = eager_reference_model(input_tensor)

print("Comparing against original PyTorch module")
print(f"ExecutorTorch output: {output[0]}")
print(f"PyTorch eager output: {eager_reference_output}")
print(f"Shapes match: {output[0].shape == eager_reference_output.shape}")
print(f"Values match (close): {torch.allclose(output[0], eager_reference_output, rtol=1e-3, atol=1e-5)}")
print(f"Absolute difference: {torch.abs(output[0] - eager_reference_output).max().item()}")
