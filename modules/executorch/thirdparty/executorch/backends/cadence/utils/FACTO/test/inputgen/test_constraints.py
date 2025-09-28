# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from facto.inputgen.attribute.model import Attribute
from facto.inputgen.specs.model import (
    Constraint,
    ConstraintProducer as cp,
    ConstraintSuffix,
)


class TestConstraint(unittest.TestCase):
    def test_constraint(self):
        constraint = cp.Optional.Ne(lambda deps: False)
        self.assertTrue(isinstance(constraint, Constraint))
        self.assertEqual(constraint.attribute, Attribute.OPTIONAL)
        self.assertEqual(constraint.suffix, ConstraintSuffix.NE)
        self.assertEqual(constraint.fn(None), False)


if __name__ == "__main__":
    unittest.main()
