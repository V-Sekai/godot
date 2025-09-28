# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.inputgen.utils.config import TensorConfig
from facto.specdb.db import SpecDictDB


def qualify_tensor(tensor):
    order = tensor.dim_order()
    dims = sum(i != order[i] for i in range(len(order)))
    empty = tensor.numel() == 0
    transposed = dims == 2
    permuted = dims > 2
    strided = len(tensor.storage()) - tensor.storage_offset() != tensor.numel()
    return empty, transposed, permuted, strided


def qualify_tensor_string(tensor):
    empty, transposed, permuted, strided = qualify_tensor(tensor)
    s = "E" if empty else ""
    s += "P" if permuted else "T" if transposed else ""
    s += "S" if strided else ""
    return s


def pretty_print_add_args(posargs, inkwargs, outargs):
    return "".join(
        [
            "Tensor{",
            qualify_tensor_string(posargs[0]),
            "}(",
            str(list(posargs[0].shape)),
            ", ",
            str(posargs[0].dtype)[6:],
            ") + Tensor{",
            qualify_tensor_string(posargs[1]),
            "}(",
            str(list(posargs[1].shape)),
            ", dtype=",
            str(posargs[0].dtype)[6:],
            ") alpha = ",
            str(inkwargs["alpha"]),
        ]
    )


def generate_inputs():
    spec = SpecDictDB["add.Tensor"]

    config = TensorConfig(
        device="mps",
        empty=False,
        transposed=False,
        permuted=True,
        strided=True,
    ).set_probability(0.7)

    print("Generating inputs with the following config:")
    print("\tdevice: ", config.device)
    print("\tconditions: ", config.conditions)

    generator = ArgumentTupleGenerator(spec, config=config)
    for ix, tup in enumerate(generator.gen()):
        posargs, inkwargs, outargs = tup
        # Pretty printing the inputs and outputs
        print(f"Tuple #{ix}: {pretty_print_add_args(posargs, inkwargs, outargs)}")
        yield posargs, inkwargs, outargs


def test_add_op():
    op = torch.ops.aten.add.Tensor
    for posargs, inkwargs, outargs in generate_inputs():
        try:
            op(*posargs, **inkwargs, **outargs)
        except Exception as e:
            print(f"Failed with error: {e}")
            return False
    return True


def main():
    print("Testing add.Tensor with the following input tuples:")
    success = test_add_op()
    if success:
        print("Success!")
    else:
        print("Failure!")


if __name__ == "__main__":
    main()
