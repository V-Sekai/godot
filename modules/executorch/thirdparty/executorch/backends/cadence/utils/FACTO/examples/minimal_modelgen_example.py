# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from facto.inputgen.utils.config import TensorConfig
from facto.modelgen.gen import OpModelGenerator
from facto.specdb.db import SpecDictDB
from facto.utils.ops import get_op_overload


def main():
    op_name = "add.Tensor"
    spec = SpecDictDB[op_name]
    op = get_op_overload(op_name)
    config = TensorConfig(device="cpu", half_precision=False)
    for model, args, kwargs in OpModelGenerator(op, spec, config).gen(verbose=True):
        model(*args, **kwargs)


if __name__ == "__main__":
    main()
