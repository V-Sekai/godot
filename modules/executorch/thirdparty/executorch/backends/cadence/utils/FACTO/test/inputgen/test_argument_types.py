# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from facto.inputgen.argument.type import ArgType


class TestArgType(unittest.TestCase):
    def test_methods(self):
        argtype = ArgType.Tensor
        self.assertTrue(argtype.is_tensor())

        argtype = ArgType.TensorList
        self.assertTrue(argtype.is_tensor_list())

        argtype = ArgType.Scalar
        self.assertTrue(argtype.is_scalar())

        argtype = ArgType.DimList
        self.assertTrue(argtype.is_dim_list())


if __name__ == "__main__":
    unittest.main()
