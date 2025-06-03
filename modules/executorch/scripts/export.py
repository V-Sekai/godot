import os

import torch
import torch.nn as nn
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower


# Simple Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size=4, output_size=1):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel(input_size=4, output_size=1).eval()

# Load trained weights if available
weights_path = "../../models/simple_linear_weights.pth"
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    print(f"Loaded trained weights from {weights_path}")
else:
    print("Warning: Using random weights. Run convert_model.py first to get trained weights.")

sample_inputs = (torch.randn(1, 4),)

et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs), partitioner=[XnnpackPartitioner()]
).to_executorch()

with open("model_mv2_xnnpack.pte", "wb") as f:
    f.write(et_program.buffer)
