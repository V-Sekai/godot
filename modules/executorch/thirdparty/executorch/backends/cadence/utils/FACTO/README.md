# FACTO

Framework for Algorithmic Correctness Testing of Operators

## InputGen

InputGen is a Python library to generate inputs for torch operators, given certain specifications. These specifications can be provided by the user, or can be retrieved from a library of specifications such as SpecDB. They provide a complete description of the space of valid inputs to the operator.

InputGen uses the `ArgumentTupleGenerator` class to systematically generate valid input combinations based on operator specifications. It handles:
- Type constraints (e.g., alpha in add must be a scalar, input must be a tensor)
- Inter-argument dependencies (e.g., output tensor shape depends on input tensor shapes)
- Shape and dtype compatibility (e.g., tensors in add operation must be broadcastable)

Here is an [overview](facto/inputgen/overview.md) of InputGen

## SpecDB

SpecDB is a [database](facto/specdb/db.py#L30) of specifications covering most of the Core ATen Ops. They have been developed using the ATen CPU kernels as a reference.

## Instalation
```bash
git clone https://github.com/meta-pytorch/FACTO.git
cd FACTO
pip install .
```

## Example Usage

The code below is a minimal example to test add.Tensor using FACTO.
```python
import torch
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.specdb.db import SpecDictDB

# Retrieve the specification from SpecDB
spec = SpecDictDB["add.Tensor"]

# Initialize generator
generator = ArgumentTupleGenerator(spec)

op = torch.ops.aten.add.Tensor

# Generate input tuples
for posargs, inkwargs, outargs in generator.gen():
    # Evaluate op with given inputs
    op(*posargs, **inkwargs, **outargs)
```

## Calibrator

Calibrator is under development. It is intended to be a tool for calibrating the specifications against the behavior of the reference op implementation.

## Reporting problems

If you encounter a bug or some other problem with FACTO, please file an issue on
https://github.com/meta-pytorch/facto/issues.

## License

FACTO has a BSD 3-Clause License, as found in the LICENSE file.
