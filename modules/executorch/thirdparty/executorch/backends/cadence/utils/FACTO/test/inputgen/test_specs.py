# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from facto.inputgen.argument.type import ArgType
from facto.inputgen.specs.model import InKwArg, InPosArg, OutArg, Return


class TestArgSpecs(unittest.TestCase):
    def test_inpos(self):
        arg = InPosArg(ArgType.Tensor, name="self")
        self.assertEqual(arg.name, "self")
        self.assertEqual(arg.type, ArgType.Tensor)
        self.assertFalse(arg.kw)
        self.assertFalse(arg.out)
        self.assertFalse(arg.ret)

    def test_inkw(self):
        arg = InKwArg(ArgType.Scalar, name="alpha")
        self.assertEqual(arg.name, "alpha")
        self.assertEqual(arg.type, ArgType.Scalar)
        self.assertTrue(arg.kw)
        self.assertFalse(arg.out)
        self.assertFalse(arg.ret)

    def test_out(self):
        arg = OutArg(ArgType.TensorList)
        self.assertEqual(arg.name, "out")
        self.assertEqual(arg.type, ArgType.TensorList)
        self.assertTrue(arg.kw)
        self.assertTrue(arg.out)
        self.assertFalse(arg.ret)

    def test_ret(self):
        arg = Return(ArgType.Tensor)
        self.assertEqual(arg.name, "__ret")
        self.assertEqual(arg.type, ArgType.Tensor)
        self.assertFalse(arg.kw)
        self.assertFalse(arg.out)
        self.assertTrue(arg.ret)


if __name__ == "__main__":
    unittest.main()
