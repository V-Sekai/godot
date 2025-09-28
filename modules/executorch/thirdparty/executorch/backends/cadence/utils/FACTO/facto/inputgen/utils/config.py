# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

import torch


class Condition(str, Enum):
    ALLOW_ZERODIM = "zerodim"
    ALLOW_EMPTY = "empty"
    ALLOW_TRANSPOSED = "transposed"
    ALLOW_PERMUTED = "permuted"
    ALLOW_STRIDED = "strided"
    DISALLOW_DTYPES = "disallow_dtypes"
    HALF_PRECISION = "half_precision"


class TensorConfig:
    def __init__(self, device="cpu", disallow_dtypes=None, **conditions):
        self.device = device
        self.disallow_dtypes = disallow_dtypes or []
        self.conditions = {condition: False for condition in Condition}
        self.conditions[Condition.ALLOW_ZERODIM] = True  # allow zerodim by default
        for condition, value in conditions.items():
            if condition in self.conditions:
                self.conditions[condition] = value
        if self.conditions[Condition.HALF_PRECISION] is False:
            self.disallow_dtypes += [torch.float16, torch.bfloat16]
        self.probability = 0.5

    def is_allowed(self, condition: Condition) -> bool:
        return self.conditions.get(condition, False)

    def is_dtype_disallowed(self, dtype) -> bool:
        """Check if a given dtype is in the disallow list."""
        return dtype in self.disallow_dtypes

    def set_probability(self, probability: float) -> "TensorConfig":
        self.probability = probability
        return self
