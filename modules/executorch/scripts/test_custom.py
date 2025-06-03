#!/usr/bin/env python3
"""
Test custom input values with both PyTorch and ExecuTorch models
"""

import sys
import torch
import os
from typing import List

def test_custom_input(input_string):
    """Test with custom input values"""
    print(f"Testing with custom input: {input_string}")
    
    # Parse input values
    try:
        input_vals = [float(x) for x in input_string.split(',')]
        if len(input_vals) != 4:
            print("Error: Expected 4 input values")
            return
    except ValueError:
        print("Error: Invalid input format. Use comma-separated numbers like '1.0,2.0,3.0,4.0'")
        return
    
    test_input = torch.tensor([input_vals])
    print(f"Input tensor: {test_input}")
    
    # Import models from run.py if available
    try:
        exec(open('run.py').read(), globals())
        
        # Test with both models
        pytorch_out = eager_reference_model(test_input)
        executorch_out = method.execute([test_input])[0]
        
        print(f"PyTorch output: {pytorch_out}")
        print(f"ExecuTorch output: {executorch_out}")
        
        # Check equivalency
        diff = torch.abs(pytorch_out - executorch_out).max().item()
        match = torch.allclose(pytorch_out, executorch_out, rtol=1e-3, atol=1e-5)
        
        print(f"Difference: {diff:.6f}")
        print(f"Match: {match}")
        
    except Exception as e:
        print(f"Error running models: {e}")
        print("Make sure to run 'just convert' first")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_custom.py <input_values>")
        print("Example: python test_custom.py 1.0,2.0,3.0,4.0")
        sys.exit(1)
    
    test_custom_input(sys.argv[1])