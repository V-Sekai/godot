#!/usr/bin/env python3
"""
Stress test script to validate PyTorch vs ExecuTorch equivalency with random inputs
"""

import torch
import numpy as np
import os
from typing import List

# Import the model and setup from run.py
exec(open('run.py').read())

def stress_test_equivalency():
    """Test equivalency with multiple random inputs"""
    print('=== Stress Test with Random Inputs ===')
    
    torch.manual_seed(123)
    num_tests = 10
    passed = 0
    failed = 0
    
    for i in range(num_tests):
        # Generate random input
        test_input = torch.randn(1, 4)
        
        # Run both models
        pytorch_out = eager_reference_model(test_input)
        executorch_out = method.execute([test_input])[0]
        
        # Check equivalency
        diff = torch.abs(pytorch_out - executorch_out).max().item()
        match = torch.allclose(pytorch_out, executorch_out, rtol=1e-3, atol=1e-5)
        
        status = "✅ PASS" if match else "❌ FAIL"
        print(f'Test {i+1:2d}: diff={diff:.6f}, match={match} {status}')
        
        if match:
            passed += 1
        else:
            failed += 1
    
    print(f'\n=== Summary ===')
    print(f'Total tests: {num_tests}')
    print(f'Passed: {passed}')
    print(f'Failed: {failed}')
    print(f'Success rate: {100*passed/num_tests:.1f}%')
    
    if failed == 0:
        print("🎉 All tests passed! Models are equivalent.")
    else:
        print("⚠️ Some tests failed. Check model consistency.")

if __name__ == "__main__":
    stress_test_equivalency()