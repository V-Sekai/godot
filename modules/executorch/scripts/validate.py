#!/usr/bin/env python3
"""
Validation script to check model files and run equivalency tests
"""

import os
import torch
from typing import List

def validate_models():
    """Validate model outputs match within tolerance"""
    print('=== Validation Report ===')
    
    # Check if model files exist
    model_file = "models/simple_linear.pte"
    weights_file = "models/simple_linear_weights.pth"
    
    print(f'Model file exists: {os.path.exists(model_file)}')
    print(f'Weights file exists: {os.path.exists(weights_file)}')
    
    if not os.path.exists(model_file):
        print("❌ Model file not found. Run 'just convert' first.")
        return False
    
    if not os.path.exists(weights_file):
        print("❌ Weights file not found. Run 'just convert' first.")
        return False
    
    # Run the equivalency test
    try:
        print("\n=== Running Equivalency Check ===")
        exec(open('run.py').read(), globals())
        print("✅ Validation completed successfully")
        return True
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    validate_models()