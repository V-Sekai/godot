# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.inputgen.utils.random_manager import random_manager
from facto.specdb.db import SpecDictDB


def main():
    # example to seed all random number generators
    random_manager.seed(1729)

    spec = SpecDictDB["add.Tensor"]
    op = torch.ops.aten.add.Tensor
    for ix, (posargs, inkwargs, outargs) in enumerate(
        ArgumentTupleGenerator(spec).gen()
    ):
        op(*posargs, **inkwargs, **outargs)
        print(
            posargs[0].shape,
            posargs[0].dtype,
            posargs[1].shape,
            posargs[1].dtype,
            inkwargs["alpha"],
        )
        if ix == 1:
            print(posargs[0])


if __name__ == "__main__":
    main()
