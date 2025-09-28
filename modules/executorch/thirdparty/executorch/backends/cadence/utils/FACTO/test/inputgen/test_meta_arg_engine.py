# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from facto.inputgen.argument.engine import MetaArgEngine
from facto.inputgen.argument.type import ArgType
from facto.inputgen.attribute.model import Attribute
from facto.inputgen.specs.model import ConstraintProducer as cp
from facto.inputgen.variable.type import SUPPORTED_TENSOR_DTYPES


class TestMetaArgEngine(unittest.TestCase):
    def test_tensor(self):
        constraints = [
            cp.Rank.Le(lambda deps: deps[0] + 2),
            cp.Size.NotIn(lambda deps, length, ix: [1, 3]),
            cp.Size.Le(lambda deps, length, ix: 5),
            cp.Value.Ne(lambda deps, dtype, struct: 0),
        ]
        deps = [2]
        outarg = False

        engine = MetaArgEngine(outarg, ArgType.Tensor, constraints, deps, True)
        ms = list(engine.gen(Attribute.DTYPE))
        self.assertEqual(len(ms), len(SUPPORTED_TENSOR_DTYPES))
        self.assertEqual({m.dtype for m in ms}, set(SUPPORTED_TENSOR_DTYPES))
        self.assertTrue(all(0 <= m.rank() <= 4 for m in ms))
        for m in ms:
            self.assertTrue(
                all(0 <= size <= 5 and size not in [1, 3] for size in m.structure)
            )
        for m in ms:
            self.assertEqual(str(m.value), "[-inf, 0.0) (0.0, inf]")

        ms = list(engine.gen(Attribute.RANK))
        self.assertEqual(len(ms), 4)
        ranks = {len(m.structure) for m in ms}
        self.assertTrue(0 in ranks)
        self.assertTrue(4 in ranks)
        self.assertTrue(all(0 <= r <= 4 for r in ranks))

    def test_dim_list(self):
        constraints = [
            cp.Length.Le(lambda deps: deps[0] + deps[1]),
            cp.Value.Gen(
                lambda deps, length: ({(deps[0],) * length}, {(deps[1],) * length})
            ),
        ]
        deps = [2, 3]
        outarg = False

        engine = MetaArgEngine(outarg, ArgType.DimList, constraints, deps, True)
        ms = list(engine.gen(Attribute.VALUE))
        self.assertEqual(len(ms), 1)
        self.assertTrue(1 <= len(ms[0].value) <= 5)
        self.assertTrue(all(v == 2 for v in ms[0].value))

        engine = MetaArgEngine(outarg, ArgType.DimList, constraints, deps, False)
        ms = list(engine.gen(Attribute.VALUE))
        self.assertEqual(len(ms), 1)
        self.assertTrue(1 <= len(ms[0].value) <= 5)
        self.assertTrue(all(v == 3 for v in ms[0].value))

    def test_optional_int(self):
        constraints = [cp.Optional.Eq(lambda deps: True)]
        deps = []
        outarg = False

        engine = MetaArgEngine(outarg, ArgType.IntOpt, constraints, deps, True)
        ms = list(engine.gen(Attribute.OPTIONAL))  # focus is OPTIONAL
        self.assertEqual(len(ms), 1)
        self.assertEqual(str(ms[0]), "ArgType.IntOpt None")

        engine = MetaArgEngine(outarg, ArgType.IntOpt, constraints, deps, True)
        ms = list(engine.gen(Attribute.VALUE))  # focus is VALUE
        self.assertEqual(len(ms), 1)
        self.assertEqual(str(ms[0]), "ArgType.IntOpt None")


if __name__ == "__main__":
    unittest.main()
