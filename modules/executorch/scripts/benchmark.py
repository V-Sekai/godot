#!/usr/bin/env python3
"""
Benchmark script to compare PyTorch eager vs ExecuTorch performance
"""

import time
import torch
import os
from typing import List

# Import the model and setup from run.py
exec(open('run.py').read())

def benchmark_models():
    """Compare performance between PyTorch eager and ExecuTorch"""
    print('=== Performance Benchmark ===')
    
    test_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    num_runs = 1000
    
    # Benchmark PyTorch eager
    print("Benchmarking PyTorch eager mode...")
    start = time.time()
    for _ in range(num_runs):
        _ = eager_reference_model(test_input)
    pytorch_time = time.time() - start
    print(f'PyTorch eager ({num_runs} runs): {pytorch_time:.4f}s')
    
    # Benchmark ExecuTorch
    print("Benchmarking ExecuTorch...")
    start = time.time()
    for _ in range(num_runs):
        _ = method.execute([test_input])
    executorch_time = time.time() - start
    print(f'ExecuTorch ({num_runs} runs): {executorch_time:.4f}s')
    
    # Calculate speedup
    if executorch_time > 0:
        speedup = pytorch_time / executorch_time
        print(f'Speedup: {speedup:.2f}x')
        if speedup > 1:
            print("✅ ExecuTorch is faster")
        else:
            print("⚠️ PyTorch eager is faster")
    else:
        print("❌ ExecuTorch timing error")

if __name__ == "__main__":
    benchmark_models()