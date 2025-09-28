# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def get_op_overload(op_name: str):
    """
    Get the torch operation overload from an operation name.

    Args:
        op_name: Operation name in the format "op_base.overload" (e.g., "add.Tensor")

    Returns:
        The torch operation overload (e.g., torch.ops.aten.add.Tensor)

    Raises:
        AttributeError: If the operation is not found
        ValueError: If the operation name format is invalid
    """
    if "." not in op_name:
        raise ValueError(
            f"Operation name '{op_name}' must contain a '.' to separate base and overload"
        )

    parts = op_name.split(".")
    if len(parts) != 2:
        raise ValueError(
            f"Operation name '{op_name}' must be in format 'op_base.overload'"
        )

    op_base, overload = parts

    # Get the operation from torch.ops.aten
    if not hasattr(torch.ops.aten, op_base):
        raise AttributeError(f"Operation base '{op_base}' not found in torch.ops.aten")

    op_obj = getattr(torch.ops.aten, op_base)

    if not hasattr(op_obj, overload):
        raise AttributeError(
            f"Overload '{overload}' not found for operation '{op_base}'"
        )

    return getattr(op_obj, overload)
