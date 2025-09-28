# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random

import torch


class RandomManager:
    def __init__(self):
        self._rng = random.Random()
        self._torch_rng = torch.Generator()

    def seed(self, seed):
        """
        Seeds the random number generators for random and torch.
        """
        self._rng.seed(seed)
        self._torch_rng.manual_seed(seed)

    def get_random(self):
        # self._rng.seed(42)
        return self._rng

    def get_torch(self):
        # self._torch_rng.manual_seed(42)
        return self._torch_rng


random_manager = RandomManager()
seeded_random_manager = RandomManager()
# 1792 is a random seed that was used to generate the random inputs for the Facto benchmark results
seeded_random_manager.seed(1792)
