# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from base_test import BaseSpecDBTest
from facto.inputgen.utils.config import TensorConfig


class TestSpecDBOperationsMPS(BaseSpecDBTest):
    """Test class for validating all specs in SpecDB using gen_errors on MPS."""

    SKIP_OPS = [
        # https://github.com/pytorch/pytorch/issues/160208
        "add.Tensor",
        "add.Scalar",
        "rsub.Scalar",
        "sub.Tensor",
        "sub.Scalar",
        # crash: https://github.com/pytorch/pytorch/issues/154887
        "_native_batch_norm_legit_no_training.default",
        "_native_batch_norm_legit.default",
        "_native_batch_norm_legit.no_stats",
        # not implemented
        "_pdist_forward.default",
        # impl: clamp tensor number of dims must not be greater than that of input tensor
        # https://github.com/pytorch/pytorch/issues/160734
        "clamp.Tensor",
        # invalid padding argument of size 0
        # https://github.com/pytorch/pytorch/issues/161066
        "constant_pad_nd.default",
        # crash: https://github.com/pytorch/pytorch/issues/154881
        "cumsum.default",
        # sparse_grad not supported in MPS yet
        "gather.default",
        # Dimension specified as -1 but tensor has no dimensions
        # https://github.com/pytorch/pytorch/issues/160737
        "index_select.default",
        # crash: https://github.com/pytorch/pytorch/issues/154882
        "max_pool2d_with_indices.default",
        # not implemented
        "native_dropout.default",
        # On-going issue on MPSGraph topk when ndims() - axis > 4, see issue #154890
        # https://github.com/pytorch/pytorch/issues/154890
        "topk.default",
        # var_mps: reduction dim must be in the range of input shape
        # https://github.com/pytorch/pytorch/issues/160738
        "var.correction",
        "var.dim",
    ]

    def test_all_ops_mps(self):
        config = TensorConfig(
            device="mps", disallow_dtypes=[torch.float64], half_precision=False
        )
        self._run_all_ops(config=config, skip_ops=self.SKIP_OPS)

    def test_all_ops_mps_half(self):
        skip_ops = self.SKIP_OPS.copy()
        # ConvTranspose 3D with BF16 or FP16 types is not supported on MPS
        # https://github.com/pytorch/pytorch/issues/160739
        skip_ops += ["convolution.default"]

        config = TensorConfig(
            device="mps", disallow_dtypes=[torch.float64], half_precision=True
        )
        self._run_all_ops(config=config, skip_ops=skip_ops)

    def test_all_ops_mps_transposed(self):
        skip_ops = self.SKIP_OPS.copy()
        # argmax.default ['ArgType.Tensor torch.float32 (8, 8)', 'ArgType.DimOpt None', 'ArgType.Bool True']
        # Exception occurred: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        # https://github.com/pytorch/pytorch/issues/160740
        skip_ops += ["argmax.default", "argmin.default"]

        config = TensorConfig(
            device="mps", disallow_dtypes=[torch.float64], transposed=True
        )
        self._run_all_ops(config=config, skip_ops=skip_ops)

    def test_all_ops_mps_permuted(self):
        skip_ops = self.SKIP_OPS.copy()
        skip_ops += ["argmax.default", "argmin.default"]

        config = TensorConfig(
            device="mps", disallow_dtypes=[torch.float64], permuted=True
        )
        self._run_all_ops(config=config, skip_ops=skip_ops)

    def test_all_ops_mps_strided(self):
        skip_ops = self.SKIP_OPS.copy()
        skip_ops += ["argmax.default", "argmin.default"]

        config = TensorConfig(
            device="mps", disallow_dtypes=[torch.float64], strided=True
        )
        self._run_all_ops(config=config, skip_ops=skip_ops)

    def test_correctness_all_ops_mps(self):
        config = TensorConfig(
            device="mps", disallow_dtypes=[torch.float64], half_precision=False
        )
        self._run_all_ops(config=config, skip_ops=self.SKIP_OPS, check_correctness=True)


if __name__ == "__main__":
    unittest.main()
