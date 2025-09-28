# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import facto.utils.dtypes as dt
import torch
from facto.inputgen.argument.engine import MetaArg
from facto.inputgen.utils.config import Condition, TensorConfig
from facto.inputgen.utils.random_manager import seeded_random_manager
from facto.inputgen.variable.gen import VariableGenerator
from facto.inputgen.variable.space import VariableSpace


FLOAT_RESOLUTION = 8


@dataclass
class TensorTransformation:
    """Represents a transformation to be applied to a tensor."""

    transpose: Optional[Tuple[int, int]] = None
    permute: Optional[Tuple[int, ...]] = None
    permute_inverse: Optional[Tuple[int, ...]] = None
    slice_steps: Optional[Tuple[int, ...]] = None


class TensorTransformationGenerator:
    def __init__(
        self,
        dtype: Optional[torch.dtype],
        structure: Tuple,
        config: Optional[TensorConfig] = None,
    ):
        self.dtype: Optional[torch.dtype] = dtype
        self.structure: Tuple = structure
        self.config: Optional[TensorConfig] = config

    def gen_slice_steps(self):
        slice_steps = []

        # Select a non-empty subset of dimensions to modify
        num_dims = len(self.structure)
        num_to_modify = seeded_random_manager.get_random().randint(1, min(num_dims, 3))
        dims_to_modify = seeded_random_manager.get_random().sample(
            range(num_dims), num_to_modify
        )

        # Modify the selected dimensions
        for i in range(num_dims):
            if i in dims_to_modify:
                factor = seeded_random_manager.get_random().choice([2, 3, 4])
                slice_steps.append(factor)
            else:
                slice_steps.append(1)

        return tuple(slice_steps)

    def gen(self):
        if self.dtype is None or self.config is None:
            return None

        tt = TensorTransformation()

        ndim = len(self.structure)

        if (
            self.config.is_allowed(Condition.ALLOW_PERMUTED)
            and ndim >= 2
            and seeded_random_manager.get_random().random() < self.config.probability
        ):
            dims = list(range(ndim))
            seeded_random_manager.get_random().shuffle(dims)

            # Compute inverse permutation
            inverse_dims = [0] * len(dims)
            for new_pos, original_pos in enumerate(dims):
                inverse_dims[original_pos] = new_pos

            tt.permute = tuple(dims)
            tt.permute_inverse = tuple(inverse_dims)

        elif (
            self.config.is_allowed(Condition.ALLOW_TRANSPOSED)
            and ndim >= 2
            and seeded_random_manager.get_random().random() < self.config.probability
        ):
            dims = list(range(ndim))
            dim1, dim2 = seeded_random_manager.get_random().sample(dims, 2)

            tt.transpose = (dim1, dim2)

        if (
            self.config
            and self.config.is_allowed(Condition.ALLOW_STRIDED)
            and len(self.structure) >= 1
            and seeded_random_manager.get_random().random() < self.config.probability
        ):
            tt.slice_steps = self.gen_slice_steps()

        return tt


class TensorGenerator:
    def __init__(
        self,
        dtype: Optional[torch.dtype],
        structure: Tuple,
        space: VariableSpace,
        device: str = "cpu",
        transformation: Optional[TensorTransformation] = None,
    ):
        self.dtype = dtype
        self.structure = structure
        self.space = space
        self.device = device
        self.transformation = transformation

    def gen(self):
        if self.dtype is None:
            return None
        vg = VariableGenerator(self.space)
        min_val = vg.gen_min()
        max_val = vg.gen_max()
        if min_val == float("-inf"):
            min_val = None
        if max_val == float("inf"):
            max_val = None

        underlying_shape = self.structure
        if self.transformation and self.transformation.slice_steps:
            underlying_shape = [
                self.structure[i] * self.transformation.slice_steps[i]
                for i in range(len(self.structure))
            ]

        # Generate tensor with the calculated size
        tensor = self.get_random_tensor(
            size=underlying_shape, dtype=self.dtype, high=max_val, low=min_val
        )
        tensor = tensor.to(self.device)

        # Apply transformations as instructed
        tensor = self._apply_transformation(tensor)

        return tensor

    def _apply_transformation(self, tensor):
        """Apply transformations as instructed by the TensorTransformation."""
        if self.transformation is None:
            return tensor

        original_size = tuple(self.structure)

        # Apply permutation
        if self.transformation.permute and self.transformation.permute_inverse:
            tensor = (
                tensor.permute(self.transformation.permute_inverse)
                .contiguous()
                .permute(self.transformation.permute)
            )

        # Apply transposition
        elif self.transformation.transpose:
            tensor = (
                tensor.transpose(*self.transformation.transpose)
                .contiguous()
                .transpose(*self.transformation.transpose)
            )

        # Apply non-contiguity
        if self.transformation.slice_steps:
            tensor = self._apply_noncontiguity(tensor)

        # Ensure the final tensor has the expected size
        assert (
            tuple(tensor.size()) == original_size
        ), f"Expected size {original_size}, got {tuple(tensor.size())}"

        return tensor

    def _apply_noncontiguity(self, tensor):
        if self.transformation is None or self.transformation.slice_steps is None:
            return tensor

        indices = []
        for i in range(len(self.structure)):
            indices.append(slice(None, None, self.transformation.slice_steps[i]))
        indices = tuple(indices)

        return tensor[indices]

    def get_random_tensor(self, size, dtype, high=None, low=None) -> torch.Tensor:
        torch_rng = seeded_random_manager.get_torch()

        if low is None and high is None:
            low = -100
            high = 100
        elif low is None:
            assert high is not None
            low = high - 100
        elif high is None:
            assert low is not None
            high = low + 100
        size = tuple(size)
        if dtype == torch.bool:
            if not self.space.contains(0):
                return torch.full(size, True, dtype=dtype)
            elif not self.space.contains(1):
                return torch.full(size, False, dtype=dtype)
            else:
                return torch.randint(
                    low=0, high=2, size=size, dtype=dtype, generator=torch_rng
                )

        if dtype in dt._int:
            low = math.ceil(low)
            high = math.floor(high) + 1
        elif dtype in dt._floating:
            low = math.ceil(FLOAT_RESOLUTION * low)
            high = math.floor(FLOAT_RESOLUTION * high) + 1
        else:
            raise ValueError(f"Unsupported Dtype: {dtype}")

        if dtype == torch.uint8:
            if not self.space.contains(0):
                return torch.randint(
                    low=max(1, low),
                    high=high,
                    size=size,
                    dtype=dtype,
                    generator=torch_rng,
                )
            else:
                return torch.randint(
                    low=max(0, low),
                    high=high,
                    size=size,
                    dtype=dtype,
                    generator=torch_rng,
                )

        t = torch.randint(
            low=low, high=high, size=size, dtype=torch.float, generator=torch_rng
        )
        if not self.space.contains(0):
            if high > 0:
                pos = torch.randint(
                    low=max(1, low),
                    high=high,
                    size=size,
                    dtype=torch.float,
                    generator=torch_rng,
                )
            else:
                pos = torch.randint(
                    low=low, high=0, size=size, dtype=torch.float, generator=torch_rng
                )
            t = torch.where(t == 0, pos, t)

        if dtype in dt._int:
            return t.to(dtype)
        if dtype in dt._floating:
            return (t / FLOAT_RESOLUTION).to(dtype)
        raise ValueError(f"Unsupported Dtype: {dtype}")


class ArgumentGenerator:
    def __init__(self, meta: MetaArg, config=None):
        self.meta = meta
        self.config = config

    def gen(self):
        device = "cpu" if self.config is None else self.config.device

        if self.meta.optional:
            return None
        elif self.meta.argtype.is_tensor():
            # Generate transformation first
            transformation_generator = TensorTransformationGenerator(
                dtype=self.meta.dtype, structure=self.meta.structure, config=self.config
            )
            transformation = transformation_generator.gen()

            # Generate tensor with transformation
            return TensorGenerator(
                dtype=self.meta.dtype,
                structure=self.meta.structure,
                space=self.meta.value,
                device=device,
                transformation=transformation,
            ).gen()
        elif self.meta.argtype.is_tensor_list():
            tensors = []
            for i in range(len(self.meta.dtype)):
                # Generate transformation for each tensor
                transformation_generator = TensorTransformationGenerator(
                    dtype=self.meta.dtype[i],
                    structure=self.meta.structure[i],
                    config=self.config,
                )
                transformation = transformation_generator.gen()

                # Generate tensor with transformation
                tensor = TensorGenerator(
                    dtype=self.meta.dtype[i],
                    structure=self.meta.structure[i],
                    space=self.meta.value,
                    device=device,
                    transformation=transformation,
                ).gen()
                tensors.append(tensor)
            return tensors
        else:
            return self.meta.value
