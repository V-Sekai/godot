# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from facto.inputgen.argument.engine import StructuralEngine
from facto.inputgen.argument.type import ArgType
from facto.inputgen.attribute.model import Attribute
from facto.inputgen.specs.model import ConstraintProducer as cp


class TestStructuralEngine(unittest.TestCase):
    def test_engine(self):
        constraints = [
            cp.Rank.Le(lambda deps: deps[0] + 2),
            cp.Size.NotIn(lambda deps, length, ix: [1, 3]),
            cp.Size.Le(lambda deps, length, ix: 5),
            cp.Value.Ne(lambda deps: 0),
        ]
        deps = [2]

        engine = StructuralEngine(ArgType.Tensor, constraints, deps, True)
        for s in engine.gen(Attribute.VALUE):
            self.assertTrue(1 <= len(s) <= 4)
            self.assertTrue(all(v in [2, 4, 5] for v in s))


if __name__ == "__main__":
    unittest.main()
