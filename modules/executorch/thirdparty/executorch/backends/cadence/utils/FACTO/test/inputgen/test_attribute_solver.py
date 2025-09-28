# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from facto.inputgen.argument.type import ArgType
from facto.inputgen.attribute.model import Attribute
from facto.inputgen.attribute.solve import AttributeSolver
from facto.inputgen.specs.model import ConstraintProducer as cp
from facto.inputgen.variable.type import ScalarDtype


class TestAttributeSolver(unittest.TestCase):
    def test_solver(self):
        solver = AttributeSolver(Attribute.VALUE, ArgType.Scalar, ScalarDtype.float)
        constraints = [
            cp.Value.Ge(lambda x, y: 1),
            cp.Value.Le(lambda x, y: x + 3),
            cp.Value.Ne(lambda x, y: y + 1),
        ]
        x = 2
        y = 1
        variables = list(solver.solve(constraints, Attribute.VALUE, True, x, y))
        self.assertEqual(len(variables), 1)
        self.assertEqual(str(variables[0].space), "[1.0, 2.0) (2.0, 5.0]")

        variables = list(solver.solve(constraints, Attribute.VALUE, False, x, y))
        self.assertEqual(len(variables), 3)
        self.assertEqual(str(variables[0].space), "[-inf, 1.0)")
        self.assertEqual(str(variables[1].space), "(5.0, inf]")
        self.assertEqual(str(variables[2].space), "{2.0}")

    def test_hidden_constraints(self):
        solver = AttributeSolver(Attribute.RANK, ArgType.Tensor)
        constraints = [
            cp.Rank.Le(lambda x: x + 3),
        ]
        x = 2

        # valid case: test focus constraint
        variables = list(solver.solve(constraints, Attribute.SIZE, True, x))
        self.assertEqual(len(variables), 1)
        self.assertEqual(str(variables[0].space), "[1, 5]")

        # valid case: test hard constraint
        variables = list(solver.solve(constraints, Attribute.RANK, True, x))
        self.assertEqual(len(variables), 1)
        self.assertEqual(str(variables[0].space), "[0, 5]")

        # invalid case: test focus constraint
        variables = list(solver.solve(constraints, Attribute.SIZE, False, x))
        self.assertEqual(len(variables), 1)
        self.assertEqual(str(variables[0].space), "[1, 5]")

        constraints = [
            cp.Rank.Ge(lambda x: x),
            cp.Rank.Le(lambda x: x + 4),
        ]
        x = 3

        # invalid case: test hard constraint
        variables = list(solver.solve(constraints, Attribute.RANK, False, x))
        self.assertEqual(len(variables), 2)
        self.assertEqual(str(variables[0].space), "[0, 3)")
        self.assertEqual(str(variables[1].space), "(7, inf)")

    def test_scalar_type(self):
        solver = AttributeSolver(Attribute.VALUE, ArgType.ScalarType)
        constraints = []

        variables = list(solver.solve(constraints, Attribute.VALUE, True))
        self.assertEqual(len(variables), 1)
        self.assertFalse(variables[0].space.empty())

        variables = list(solver.solve(constraints, Attribute.VALUE, False))
        self.assertEqual(len(variables), 0)

    def test_optional(self):
        solver = AttributeSolver(Attribute.OPTIONAL, ArgType.IntOpt)
        constraints = [cp.Optional.Eq(lambda: True)]

        variables = list(solver.solve(constraints, Attribute.OPTIONAL, True))
        self.assertEqual(len(variables), 1)
        self.assertEqual(str(variables[0].space), "{True}")

        solver = AttributeSolver(Attribute.OPTIONAL, ArgType.IntOpt)
        constraints = [cp.Optional.Eq(lambda: False)]

        variables = list(solver.solve(constraints, Attribute.OPTIONAL, True))
        self.assertEqual(len(variables), 1)
        self.assertEqual(str(variables[0].space), "{False}")


if __name__ == "__main__":
    unittest.main()
