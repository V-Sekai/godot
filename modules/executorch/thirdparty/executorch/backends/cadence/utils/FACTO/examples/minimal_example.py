# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.specdb.db import SpecDictDB


def main():
    # minimal example to test add.Tensor using FACTO
    spec = SpecDictDB["add.Tensor"]
    op = torch.ops.aten.add.Tensor
    for posargs, inkwargs, outargs in ArgumentTupleGenerator(spec).gen():
        op(*posargs, **inkwargs, **outargs)


if __name__ == "__main__":
    main()
