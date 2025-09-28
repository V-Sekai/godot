# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from base_test import BaseSpecDBTest
from facto.inputgen.utils.config import TensorConfig


class TestSpecDBOperationsCPU(BaseSpecDBTest):
    """Test class for validating all specs in SpecDB using gen_errors on CPU."""

    SKIP_OPS = []

    def test_all_ops_cpu(self):
        config = TensorConfig(device="cpu", half_precision=False)
        self._run_all_ops(config=config, skip_ops=self.SKIP_OPS)

    def test_all_ops_cpu_half(self):
        skip_ops = self.SKIP_OPS.copy()
        # not implemented for 'Half'
        skip_ops += [
            "_cdist_forward.default",
            "_pdist_forward.default",
            "_upsample_bilinear2d_aa.default",
        ]

        config = TensorConfig(device="cpu", half_precision=True)
        self._run_all_ops(config=config, skip_ops=skip_ops)

    def test_all_ops_cpu_transposed(self):
        skip_ops = self.SKIP_OPS.copy()
        # Expected X.is_contiguous(memory_format) to be true, but got false.
        skip_ops += ["native_group_norm.default"]
        # _pdist_forward requires contiguous input
        skip_ops += ["_pdist_forward.default"]
        config = TensorConfig(device="cpu", transposed=True)
        self._run_all_ops(config=config, skip_ops=skip_ops)

    def test_all_ops_cpu_permuted(self):
        skip_ops = self.SKIP_OPS.copy()
        # Expected X.is_contiguous(memory_format) to be true, but got false.
        skip_ops += ["native_group_norm.default"]
        # _pdist_forward requires contiguous input
        skip_ops += ["_pdist_forward.default"]
        # Unsupported memory format. Supports only ChannelsLast3d, Contiguous
        skip_ops += ["max_pool3d_with_indices.default"]
        # Unsupported memory format. Supports only ChannelsLast, Contiguous
        skip_ops += ["pixel_shuffle.default"]
        config = TensorConfig(device="cpu", permuted=True)
        self._run_all_ops(config=config, skip_ops=skip_ops)

    def test_all_ops_cpu_strided(self):
        skip_ops = self.SKIP_OPS.copy()
        # Expected X.is_contiguous(memory_format) to be true, but got false.
        skip_ops += ["native_group_norm.default"]
        # _pdist_forward requires contiguous input
        skip_ops += ["_pdist_forward.default"]
        config = TensorConfig(device="cpu", strided=True)
        self._run_all_ops(config=config, skip_ops=skip_ops)


if __name__ == "__main__":
    unittest.main()
