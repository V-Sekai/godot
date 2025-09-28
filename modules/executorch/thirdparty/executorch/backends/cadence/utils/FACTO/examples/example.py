# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.specdb.db import SpecDictDB


def pretty_print_add_args(posargs, inkwargs, outargs):
    return "".join(
        [
            "tensor(shape=",
            str(list(posargs[0].shape)),
            ", dtype=",
            str(posargs[0].dtype),
            ") + tensor(shape=",
            str(list(posargs[1].shape)),
            ", dtype=",
            str(posargs[0].dtype),
            ") alpha = ",
            str(inkwargs["alpha"]),
        ]
    )


def generate_inputs():
    spec = SpecDictDB["add.Tensor"]
    generator = ArgumentTupleGenerator(spec)
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
        except Exception:
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
