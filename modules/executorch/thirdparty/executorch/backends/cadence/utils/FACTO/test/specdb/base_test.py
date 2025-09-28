# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional

from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.inputgen.utils.config import TensorConfig
from facto.specdb.db import SpecDictDB
from facto.utils.ops import get_op_overload


class BaseSpecDBTest(unittest.TestCase):
    """Base test class for validating all specs in SpecDB using gen_errors."""

    def _run_op(
        self,
        op_name: str,
        *,
        config: Optional[TensorConfig] = None,
        check_correctness: bool = False,
    ):
        """
        Run a single op in SpecDB with a given TensorConfig

        This test calls ArgumentTupleGenerator.gen_errors with valid=True, out=False
        for a single operation. The operation is tested as a subtest.
        """
        print("Testing op: ", op_name)
        with self.subTest(op=op_name):
            try:
                # Get the spec and operation
                spec = SpecDictDB[op_name]
                op = get_op_overload(op_name)
                generator = ArgumentTupleGenerator(spec, config)
            except Exception as e:
                # If we can't resolve the operation or there's another issue,
                # fail this subtest with a descriptive message
                self.fail(f"Failed to test operation {op_name}: {e}")

            try:
                errors = list(
                    generator.gen_errors(
                        op,
                        valid=True,
                        out=False,
                        verbose=True,
                        check_correctness=check_correctness,
                    )
                )
            except Exception as e:
                self.fail(f"Failed while testing operation {op_name}: {e}")

            if len(errors) > 0:
                self.fail(
                    f"Found {len(errors)} errors for {op_name} with valid=True, out=False"
                )

    def _run_all_ops(
        self,
        *,
        config: Optional[TensorConfig] = None,
        skip_ops=[],
        check_correctness: bool = False,
    ):
        """
        Run all ops in SpecDB with a given TensorConfig

        This test iterates through all operations in SpecDB and calls
        ArgumentTupleGenerator.gen_errors with valid=True, out=False
        for each operation. Each operation is tested as a subtest.
        """
        # Get all operation names from SpecDB
        op_names = list(SpecDictDB.keys())

        for op_name in op_names:
            if op_name in skip_ops:
                continue
            self._run_op(op_name, config=config, check_correctness=check_correctness)
