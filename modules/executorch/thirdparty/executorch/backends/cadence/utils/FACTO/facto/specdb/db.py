# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import facto.specdb.function as fn
import facto.utils.dtypes as dt
import torch
from facto.inputgen.argument.type import ArgType
from facto.inputgen.specs.model import (
    ConstraintProducer as cp,
    InKwArg,
    InPosArg,
    OutArg,
    Spec,
)
from facto.inputgen.variable.type import ScalarDtype
from facto.specdb.default import (
    DimDefault,
    DimListDefault,
    IndexDefault,
    MemoryFormatDefault,
    ShapeDefault,
)


SpecDB = [
    Spec(
        op="_cdist_forward.default",  # (Tensor x1, Tensor x2, float p, int? compute_mode) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="x1",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                    cp.Rank.Ge(lambda deps: 2),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="x2",
                deps=[0],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                    cp.Rank.Ge(lambda deps: 2),
                    cp.Size.Eq(
                        lambda deps, r, d: (
                            fn.safe_size(deps[0], deps[0].dim() - 1)
                            if d == r - 1
                            else None
                        )
                    ),
                    cp.Size.In(
                        lambda deps, r, d: (
                            fn.broadcast_with(deps[0].shape[:-2], r - 2, d)
                            if d < r - 2
                            else None
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.Float,
                name="p",
                constraints=[
                    cp.Value.Ge(lambda deps: 0),
                    cp.Value.Be(
                        lambda deps: [
                            0,
                            0.1,
                            0.5,
                            1,
                            1.5,
                            2,
                            2.5,
                            3,
                            5,
                            10,
                            float("inf"),
                        ]
                    ),
                ],
            ),
            InPosArg(
                ArgType.IntOpt,
                name="compute_mode",
                constraints=[
                    cp.Value.In(lambda deps: [0, 1, 2]),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            )
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="_log_softmax.default",  # (Tensor self, int dim, bool half_to_float) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                ],
            ),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=DimDefault,
            ),
            InPosArg(
                ArgType.Bool,
                name="half_to_float",
                # TODO(mcandales): CPU specific constraint
                constraints=[
                    cp.Value.Eq(lambda deps: False),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate ATen's implementation.
        op="_native_batch_norm_legit_no_training.default",  # (Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="input",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                    cp.Rank.Ge(lambda deps: 2),
                ],
            ),
            InPosArg(
                ArgType.TensorOpt,
                name="weight",
                deps=[0],
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float64 if deps[0].dtype == torch.float64 else None
                        )
                    ),
                    cp.Dtype.In(
                        lambda deps: (
                            [deps[0].dtype, torch.float]
                            if deps[0].dtype != torch.float64
                            else None
                        )
                    ),
                    cp.Rank.Eq(lambda deps: 1),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_size(deps[0], 1)),
                ],
            ),
            InPosArg(
                ArgType.TensorOpt,
                name="bias",
                deps=[0, 1],
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float64 if deps[0].dtype == torch.float64 else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[1] is not None and deps[1].dtype == deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float
                            if deps[1] is not None and deps[1].dtype != deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.In(lambda deps: [deps[0].dtype, torch.float]),
                    cp.Rank.Eq(lambda deps: 1),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_size(deps[0], 1)),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="running_mean",
                deps=[0, 1, 2],
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float64 if deps[0].dtype == torch.float64 else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[1] is not None and deps[1].dtype == deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float
                            if deps[1] is not None and deps[1].dtype != deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[2] is not None and deps[2].dtype == deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float
                            if deps[2] is not None and deps[2].dtype != deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.In(lambda deps: [deps[0].dtype, torch.float]),
                    cp.Rank.Eq(lambda deps: 1),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_size(deps[0], 1)),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="running_var",
                deps=[0, 1, 2, 3],
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float64 if deps[0].dtype == torch.float64 else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[1] is not None and deps[1].dtype == deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float
                            if deps[1] is not None and deps[1].dtype != deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[2] is not None and deps[2].dtype == deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float
                            if deps[2] is not None and deps[2].dtype != deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[3] is not None and deps[3].dtype == deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float
                            if deps[3] is not None and deps[3].dtype != deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.In(lambda deps: [deps[0].dtype, torch.float]),
                    cp.Rank.Eq(lambda deps: 1),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_size(deps[0], 1)),
                ],
            ),
            InPosArg(ArgType.Float, name="momentum"),
            InPosArg(ArgType.Float, name="eps"),
        ],
        outspec=[
            OutArg(ArgType.Tensor, name="out0"),
            OutArg(ArgType.Tensor, name="out1"),
            OutArg(ArgType.Tensor, name="out2"),
        ],
    ),
    Spec(
        op="_native_batch_norm_legit.default",  # (Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="input",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                    cp.Rank.Ge(lambda deps: 2),
                ],
            ),
            InPosArg(
                ArgType.TensorOpt,
                name="weight",
                deps=[0],
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float64 if deps[0].dtype == torch.float64 else None
                        )
                    ),
                    cp.Dtype.In(
                        lambda deps: (
                            [deps[0].dtype, torch.float]
                            if deps[0].dtype != torch.float64
                            else None
                        )
                    ),
                    cp.Rank.Eq(lambda deps: 1),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_size(deps[0], 1)),
                ],
            ),
            InPosArg(
                ArgType.TensorOpt,
                name="bias",
                deps=[0, 1],
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float64 if deps[0].dtype == torch.float64 else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[1] is not None and deps[1].dtype == deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float
                            if deps[1] is not None and deps[1].dtype != deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.In(lambda deps: [deps[0].dtype, torch.float]),
                    cp.Rank.Eq(lambda deps: 1),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_size(deps[0], 1)),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="running_mean",
                deps=[0, 1, 2],
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float64 if deps[0].dtype == torch.float64 else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[1] is not None and deps[1].dtype == deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float
                            if deps[1] is not None and deps[1].dtype != deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[2] is not None and deps[2].dtype == deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float
                            if deps[2] is not None and deps[2].dtype != deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.In(lambda deps: [deps[0].dtype, torch.float]),
                    cp.Rank.Eq(lambda deps: 1),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_size(deps[0], 1)),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="running_var",
                deps=[0, 1, 2, 3],
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float64 if deps[0].dtype == torch.float64 else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[1] is not None and deps[1].dtype == deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float
                            if deps[1] is not None and deps[1].dtype != deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[2] is not None and deps[2].dtype == deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float
                            if deps[2] is not None and deps[2].dtype != deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[3] is not None and deps[3].dtype == deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float
                            if deps[3] is not None and deps[3].dtype != deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.In(lambda deps: [deps[0].dtype, torch.float]),
                    cp.Rank.Eq(lambda deps: 1),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_size(deps[0], 1)),
                ],
            ),
            InPosArg(
                ArgType.Bool,
                name="training",
            ),
            InPosArg(ArgType.Float, name="momentum"),
            InPosArg(ArgType.Float, name="eps"),
        ],
        outspec=[
            OutArg(ArgType.Tensor, name="out0"),
            OutArg(ArgType.Tensor, name="out1"),
            OutArg(ArgType.Tensor, name="out2"),
        ],
    ),
    Spec(
        op="_native_batch_norm_legit.no_stats",  # (Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="input",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                    cp.Rank.Ge(lambda deps: 2),
                ],
            ),
            InPosArg(
                ArgType.TensorOpt,
                name="weight",
                deps=[0],
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float64 if deps[0].dtype == torch.float64 else None
                        )
                    ),
                    cp.Dtype.In(
                        lambda deps: (
                            [deps[0].dtype, torch.float]
                            if deps[0].dtype != torch.float64
                            else None
                        )
                    ),
                    cp.Rank.Eq(lambda deps: 1),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_size(deps[0], 1)),
                ],
            ),
            InPosArg(
                ArgType.TensorOpt,
                name="bias",
                deps=[0, 1],
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float64 if deps[0].dtype == torch.float64 else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[1] is not None and deps[1].dtype == deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.float
                            if deps[1] is not None and deps[1].dtype != deps[0].dtype
                            else None
                        )
                    ),
                    cp.Dtype.In(lambda deps: [deps[0].dtype, torch.float]),
                    cp.Rank.Eq(lambda deps: 1),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_size(deps[0], 1)),
                ],
            ),
            InPosArg(
                ArgType.Bool,
                name="training",
                constraints=[
                    cp.Value.Eq(lambda deps: True),
                ],
            ),
            InPosArg(ArgType.Float, name="momentum"),
            InPosArg(ArgType.Float, name="eps"),
        ],
        outspec=[
            OutArg(ArgType.Tensor, name="out0"),
            OutArg(ArgType.Tensor, name="out1"),
            OutArg(ArgType.Tensor, name="out2"),
        ],
    ),
    Spec(
        op="_pdist_forward.default",  # (Tensor self, float p=2) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                    cp.Rank.Eq(lambda deps: 2),
                ],
            ),
            InPosArg(
                ArgType.Float,
                name="p",
                constraints=[
                    cp.Value.Ge(lambda deps: 0),
                    cp.Value.Be(
                        lambda deps: [
                            0,
                            0.1,
                            0.5,
                            1,
                            1.5,
                            2,
                            2.5,
                            3,
                            5,
                            10,
                            float("inf"),
                        ]
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            )
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="_softmax.default",  # (Tensor self, int dim, bool half_to_float) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                ],
            ),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=DimDefault,
            ),
            InPosArg(
                ArgType.Bool,
                name="half_to_float",
                # TODO(mcandales): CPU specific constraint
                constraints=[
                    cp.Value.Eq(lambda deps: False),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="_to_copy.default",  # (Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InKwArg(
                ArgType.Bool,
                name="non_blocking",
                # TODO(mcandales): Executorch specific constraint
                # constraints=[cp.Value.In(lambda deps: [False])],
            ),
            InKwArg(ArgType.ScalarTypeOpt, name="dtype"),
            InKwArg(
                ArgType.MemoryFormat,
                name="memory_format",
                constraints=MemoryFormatDefault,
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="_upsample_bilinear2d_aa.default",  # (Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                    cp.Rank.Eq(lambda deps: 4),
                ],
            ),
            InPosArg(
                ArgType.Shape,
                name="output_size",
                constraints=[
                    cp.Length.Eq(lambda deps: 2),
                    cp.Value.Ge(lambda deps, length, ix: 1),
                ],
            ),
            InPosArg(ArgType.Bool, name="align_corners"),
            InPosArg(
                ArgType.FloatOpt,
                name="scale_h",
                constraints=[
                    cp.Value.Gt(lambda deps: 1e-2),
                    cp.Value.Le(
                        lambda deps: 10.0
                    ),  # restrict to avoid storage overflow
                ],
            ),
            InPosArg(
                ArgType.FloatOpt,
                name="scale_w",
                constraints=[
                    cp.Value.Gt(lambda deps: 1e-2),
                    cp.Value.Le(
                        lambda deps: 10.0
                    ),  # restrict to avoid storage overflow
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="abs.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="acos.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="acosh.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="add.Tensor",  # (Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
            InKwArg(
                ArgType.Scalar,
                name="alpha",
                deps=[0, 1],
                constraints=[
                    cp.Dtype.In(
                        lambda deps: fn.add_alpha_st(
                            fn.dt_to_st(
                                torch.promote_types(deps[0].dtype, deps[1].dtype)
                            )
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(
                        lambda deps: dt.can_cast_from(
                            torch.promote_types(deps[0].dtype, deps[1].dtype)
                        )
                    ),
                ],
            ),
        ],
    ),
    Spec(
        op="add.Scalar",  # (Tensor self, Scalar other, Scalar alpha=1) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(ArgType.Scalar, name="other"),
            InPosArg(
                ArgType.Scalar,
                name="alpha",
                deps=[0, 1],
                constraints=[
                    cp.Dtype.Ne(
                        lambda deps: (
                            ScalarDtype.bool
                            if deps[0].dtype != torch.bool
                            or deps[1] != ScalarDtype.bool
                            else None
                        )
                    ),
                    cp.Dtype.In(
                        lambda deps: fn.st_le(
                            fn.dt_to_st(
                                fn.promote_type_with_scalar(deps[0].dtype, deps[1])
                            )
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: (
                            fn.promote_type_with_scalar(deps[0].dtype, deps[1])
                        )
                    ),
                ],
            )
        ],
    ),
    Spec(
        op="addmm.default",  # (Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                deps=[1, 2],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                    cp.Rank.Le(lambda deps: 2),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_to(
                            (fn.safe_size(deps[0], 0), fn.safe_size(deps[1], 1)), r, d
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="mat1",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                    cp.Rank.Eq(lambda deps: 2),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="mat2",
                deps=[1],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                    cp.Rank.Eq(lambda deps: 2),
                    cp.Size.Eq(
                        lambda deps, r, d: fn.safe_size(deps[0], 1) if d == 0 else None
                    ),
                ],
            ),
            InKwArg(
                ArgType.Scalar,
                name="beta",
                deps=[1],
                constraints=[
                    cp.Value.Ge(
                        lambda deps, dtype: fn.dtype_lower_bound(deps[0].dtype)
                    ),
                    cp.Value.Le(
                        lambda deps, dtype: fn.dtype_upper_bound(deps[0].dtype)
                    ),
                ],
            ),
            InKwArg(
                ArgType.Scalar,
                name="alpha",
                deps=[1],
                constraints=[
                    cp.Value.Ge(
                        lambda deps, dtype: fn.dtype_lower_bound(deps[0].dtype)
                    ),
                    cp.Value.Le(
                        lambda deps, dtype: fn.dtype_upper_bound(deps[0].dtype)
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="alias_copy.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="amax.default",  # (Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.DimList,
                name="dim",
                deps=[0],
                constraints=[
                    cp.Length.Ge(lambda deps: 1 if deps[0].numel() == 0 else 0),
                    cp.Value.Gen(
                        lambda deps, length: (
                            fn.valid_dim_list_non_zero_size(deps[0], length),
                            fn.invalid_dim_list_non_zero_size(deps[0], length),
                        )
                    ),
                ],
            ),
            InPosArg(ArgType.Bool, name="keepdim"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="amin.default",  # (Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.DimList,
                name="dim",
                deps=[0],
                constraints=[
                    cp.Length.Ge(lambda deps: 1 if deps[0].numel() == 0 else 0),
                    cp.Value.Gen(
                        lambda deps, length: (
                            fn.valid_dim_list_non_zero_size(deps[0], length),
                            fn.invalid_dim_list_non_zero_size(deps[0], length),
                        )
                    ),
                ],
            ),
            InPosArg(ArgType.Bool, name="keepdim"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="any.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(lambda deps: [torch.bool, torch.uint8]),
                ],
            ),
        ],
    ),
    Spec(
        op="any.dim",  # (Tensor self, int dim, bool keepdim=False) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=DimDefault,
            ),
            InPosArg(ArgType.Bool, name="keepdim"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(lambda deps: [torch.bool, torch.uint8]),
                ],
            ),
        ],
    ),
    Spec(
        op="any.dims",  # (Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.DimListOpt,
                name="dim",
                deps=[0],
                constraints=DimListDefault,
            ),
            InPosArg(ArgType.Bool, name="keepdim"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(lambda deps: [torch.bool, torch.uint8]),
                ],
            ),
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="arange.default",  # (Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Scalar,
                name="end",
                constraints=[
                    cp.Value.Ge(lambda deps, dtype: 0),
                    cp.Value.NotIn(lambda deps, dtype: [float("-inf"), float("inf")]),
                    # This is a safeguard to avoid storage overflow
                    cp.Value.Le(lambda deps, dtype: 1e3),
                ],
            ),
            InKwArg(
                ArgType.ScalarTypeOpt,
                name="dtype",
                constraints=[
                    cp.Value.Ne(lambda deps: torch.bool),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="arange.start_step",  # (Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Scalar,
                name="start",
                deps=[3],
                constraints=[
                    cp.Value.NotIn(lambda deps, dtype: [float("-inf"), float("inf")]),
                    cp.Value.Ge(lambda deps, dtype: fn.arange_lower_bound(deps[0])),
                    cp.Value.Le(lambda deps, dtype: fn.arange_upper_bound(deps[0])),
                ],
            ),
            InPosArg(
                ArgType.Scalar,
                name="end",
                deps=[3],
                constraints=[
                    cp.Value.NotIn(lambda deps, dtype: [float("-inf"), float("inf")]),
                    cp.Value.Ge(lambda deps, dtype: fn.arange_lower_bound(deps[0])),
                    cp.Value.Le(lambda deps, dtype: fn.arange_upper_bound(deps[0])),
                ],
            ),
            InPosArg(
                ArgType.Scalar,
                name="step",
                deps=[0, 1, 3],
                constraints=[
                    cp.Value.Ge(lambda deps, dtype: fn.arange_lower_bound(deps[2])),
                    cp.Value.Le(lambda deps, dtype: fn.arange_upper_bound(deps[2])),
                    cp.Value.Gt(lambda deps, dtype: 0 if deps[0] < deps[1] else None),
                    cp.Value.Lt(lambda deps, dtype: 0 if deps[0] > deps[1] else None),
                    cp.Value.Ne(lambda deps, dtype: 0),
                    # The following is a safeguard to avoid storage overflow
                    cp.Value.Ge(
                        lambda deps, dtype: (
                            (deps[1] - deps[0]) / 1000.0 if deps[0] < deps[1] else None
                        )
                    ),
                    cp.Value.Le(
                        lambda deps, dtype: (
                            (deps[1] - deps[0]) / 1000.0 if deps[0] > deps[1] else None
                        )
                    ),
                ],
            ),
            InKwArg(
                ArgType.ScalarTypeOpt,
                name="dtype",
                constraints=[
                    cp.Value.Ne(lambda deps: torch.bool),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="argmax.default",  # (Tensor self, int? dim=None, bool keepdim=False) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                ],
            ),
            InPosArg(
                ArgType.DimOpt,
                name="dim",
                deps=[0],
                constraints=[
                    cp.Optional.Eq(
                        lambda deps: False if deps[0].numel() == 0 else None
                    ),
                    cp.Value.In(lambda deps: fn.dim_non_zero_size(deps[0])),
                ],
            ),
            InPosArg(ArgType.Bool, name="keepdim"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="argmin.default",  # (Tensor self, int? dim=None, bool keepdim=False) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    # TODO(mcandales): Enable self-context
                    # cp.Dtype.Ne(lambda deps, ctx: torch.bool if not ctx.Empty else None),
                    cp.Dtype.Ne(lambda deps: torch.bool),
                ],
            ),
            InPosArg(
                ArgType.DimOpt,
                name="dim",
                deps=[0],
                constraints=[
                    cp.Optional.Eq(
                        lambda deps: False if deps[0].numel() == 0 else None
                    ),
                    cp.Value.In(lambda deps: fn.dim_non_zero_size(deps[0])),
                ],
            ),
            InPosArg(ArgType.Bool, name="keepdim"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(  # TODO(mcandales) Implement numel constraint prefix
        op="as_strided_copy.default",  # (Tensor self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                deps=[1, 2, 3],
                constraints=[
                    cp.Rank.Ge(
                        lambda deps: (
                            1
                            if fn.as_strided_min_numel(deps[0], deps[1], deps[2]) > 1
                            else 0
                        )
                    ),
                    cp.Size.Gen(
                        lambda deps, rank: (
                            fn.valid_as_strided_sizes(deps[0], deps[1], deps[2], rank),
                            fn.invalid_as_strided_sizes(
                                deps[0], deps[1], deps[2], rank
                            ),
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.Shape,
                name="size",
                constraints=ShapeDefault,
            ),
            InPosArg(
                ArgType.LengthList,
                name="stride",
                deps=[1],
                constraints=[
                    cp.Length.Eq(lambda deps: len(deps[0])),
                    cp.Value.Ge(lambda deps, length, ix: 0),
                ],
            ),
            InPosArg(
                ArgType.LengthOpt,
                name="storage_offset",
                constraints=[
                    cp.Value.Ge(lambda deps: 0),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="asin.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="asinh.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="atan.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="atan2.default",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(lambda deps: dt.can_cast_from(torch.float)),
                    cp.Dtype.In(
                        lambda deps: dt.can_cast_from(
                            torch.promote_types(deps[0].dtype, deps[1].dtype)
                        )
                    ),
                ],
            )
        ],
    ),
    Spec(
        op="atanh.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(  # TODO(mcandales) Add input constraints to ensure valid output shape
        op="avg_pool2d.default",  # (Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
        inspec=[
            InPosArg(  # self
                ArgType.Tensor,
                name="self",
                deps=[1, 2, 3, 4],
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating + [torch.long]),
                    cp.Rank.In(lambda deps: [3, 4]),
                    cp.Size.Ge(
                        lambda deps, r, d: fn.pool_input_size_min(
                            2, deps[0], deps[1], deps[2], [], deps[3], r, d
                        )
                    ),
                ],
            ),
            InPosArg(  # kernel_size
                ArgType.LengthList,
                name="kernel_size",
                constraints=[
                    cp.Length.In(lambda deps: [1, 2]),
                    cp.Value.Ge(lambda deps, length, ix: 1),
                ],
            ),
            InPosArg(  # stride
                ArgType.LengthList,
                name="stride",
                constraints=[
                    cp.Length.In(lambda deps: [0, 1, 2]),
                    cp.Value.Ge(lambda deps, length, ix: 1),
                ],
            ),
            InPosArg(  # padding
                ArgType.LengthList,
                name="padding",
                deps=[1],
                constraints=[
                    cp.Length.In(lambda deps: [1, 2]),
                    cp.Value.Ge(lambda deps, length, ix: 0),
                    cp.Value.Le(
                        lambda deps, length, ix: fn.pool_padding_max(
                            deps[0], length, ix
                        )
                    ),
                ],
            ),
            InPosArg(ArgType.Bool, name="ceil_mode"),  # ceil_mode
            InPosArg(ArgType.Bool, name="count_include_pad"),  # count_include_pad
            InPosArg(  # divisor_override
                ArgType.IntOpt,
                name="divisor_override",
                constraints=[
                    cp.Value.Ne(lambda deps: 0),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="bitwise_and.Scalar",  # (Tensor self, Scalar other) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._int_and_bool),
                ],
            ),
            InPosArg(
                ArgType.Scalar,
                name="other",
                constraints=[
                    cp.Dtype.In(lambda deps: [ScalarDtype.bool, ScalarDtype.int]),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="bitwise_and.Tensor",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._int_and_bool),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Dtype.In(lambda deps: dt._int_and_bool),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="bitwise_not.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._int_and_bool),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="bitwise_or.Scalar",  # (Tensor self, Scalar other) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._int_and_bool),
                ],
            ),
            InPosArg(
                ArgType.Scalar,
                name="other",
                constraints=[
                    cp.Dtype.In(lambda deps: [ScalarDtype.bool, ScalarDtype.int]),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="bitwise_or.Tensor",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._int_and_bool),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Dtype.In(lambda deps: dt._int_and_bool),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="bitwise_xor.Scalar",  # (Tensor self, Scalar other) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._int_and_bool),
                ],
            ),
            InPosArg(
                ArgType.Scalar,
                name="other",
                constraints=[
                    cp.Dtype.In(lambda deps: [ScalarDtype.bool, ScalarDtype.int]),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="bitwise_xor.Tensor",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._int_and_bool),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Dtype.In(lambda deps: dt._int_and_bool),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="bmm.default",  # (Tensor self, Tensor mat2) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                    cp.Rank.Eq(lambda deps: 3),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="mat2",
                deps=[0],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                    cp.Rank.Eq(lambda deps: 3),
                    cp.Size.Eq(lambda deps, r, d: fn.bmm_mat2_size_eq(deps[0], d)),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="ceil.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="clamp.default",  # (Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(ArgType.ScalarOpt, name="min"),
            InPosArg(
                ArgType.ScalarOpt,
                name="max",
                deps=[0, 1],
                constraints=[
                    cp.Optional.Eq(
                        lambda deps: fn.clamp_max_is_optional(deps[0], deps[1])
                    ),
                    cp.Dtype.Ne(
                        lambda deps: fn.dt_to_st(
                            fn.clamp_max_ne_dtype(deps[0], deps[1])
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: fn.promote_type_with_opt_scalar(
                            fn.promote_type_with_opt_scalar(deps[0].dtype, deps[1]),
                            deps[2],
                        )
                    ),
                ],
            )
        ],
    ),
    Spec(
        op="clamp.Tensor",  # (Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.TensorOpt,
                name="min",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: (fn.broadcast_with(deps[0].shape, r, d))
                    ),
                ],
            ),
            InPosArg(
                ArgType.TensorOpt,
                name="max",
                deps=[0, 1],
                constraints=[
                    cp.Optional.Eq(lambda deps: False if deps[1] is None else None),
                    cp.Dtype.Ne(
                        lambda deps: (
                            torch.bool
                            if deps[0].dtype == torch.bool
                            and (deps[1] is None or deps[1].dtype == torch.bool)
                            else None
                        )
                    ),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(
                            (
                                fn.broadcasted_shape(deps[0].shape, deps[1].shape)
                                if deps[1] is not None
                                else deps[0].shape
                            ),
                            r,
                            d,
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(
                        lambda deps: dt.can_cast_from(
                            torch.promote_types(
                                torch.promote_types(
                                    deps[0].dtype,
                                    (
                                        deps[1].dtype
                                        if deps[1] is not None
                                        else deps[0].dtype
                                    ),
                                ),
                                deps[2].dtype if deps[2] is not None else deps[0].dtype,
                            )
                        )
                    ),
                ],
            )
        ],
    ),
    Spec(
        op="clone.default",  # (Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InKwArg(
                ArgType.MemoryFormat,
                name="memory_format",
                constraints=MemoryFormatDefault,
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="constant_pad_nd.default",  # (Tensor self, SymInt[] pad, Scalar value=0) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                deps=[1],
                constraints=[
                    cp.Rank.Ge(lambda deps: len(deps[0]) / 2),
                    # size + left padding >= 0
                    cp.Size.Ge(
                        lambda deps, r, d: (
                            -deps[0][2 * (r - d) - 2]
                            if (r - d) <= len(deps[0]) / 2
                            else None
                        )
                    ),
                    # size + right padding >= 0
                    cp.Size.Ge(
                        lambda deps, r, d: (
                            -deps[0][2 * (r - d) - 1]
                            if (r - d) <= len(deps[0]) / 2
                            else None
                        )
                    ),
                    # size + left padding + right padding >= 0
                    # Here, we use Gt instead of Ge to avoid a bug in ATen
                    # https://github.com/pytorch/pytorch/issues/161014
                    cp.Size.Gt(
                        lambda deps, r, d: (
                            -(deps[0][2 * (r - d) - 2] + deps[0][2 * (r - d) - 1])
                            if (r - d) <= len(deps[0]) / 2
                            else None
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.LengthList,
                name="pad",
                constraints=[
                    # This ensures that the number of elements in pad is even
                    cp.Length.NotIn(lambda deps: [2 * i + 1 for i in range(500)]),
                ],
            ),
            InPosArg(
                ArgType.Scalar,
                name="value",
                deps=[0],
                constraints=[
                    cp.Value.Ge(
                        lambda deps, dtype: fn.dtype_lower_bound(deps[0].dtype)
                    ),
                    cp.Value.Le(
                        lambda deps, dtype: fn.dtype_upper_bound(deps[0].dtype)
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="convolution.default",  # (Tensor input, Tensor weight, Tensor? bias, int[] stride, SymInt[] padding, int[] dilation, bool transposed, SymInt[] output_padding, int groups) -> Tensor
        inspec=[
            InPosArg(  # input
                ArgType.Tensor,
                name="input",
                deps=[1, 3, 4, 5, 6, 7, 8],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                    cp.Rank.Eq(lambda deps: deps[0].dim()),
                    cp.Size.Eq(
                        lambda deps, r, d: fn.conv_input_size_eq(
                            deps[0], deps[4], deps[6], d
                        )
                    ),
                    cp.Size.Ge(
                        lambda deps, r, d: fn.conv_input_size_min(
                            deps[0], deps[1], deps[2], deps[3], deps[4], deps[5], d
                        )
                    ),
                ],
            ),
            InPosArg(  # weight
                ArgType.Tensor,
                name="weight",
                deps=[6],
                constraints=[
                    cp.Dtype.In(
                        lambda deps: dt._floating
                    ),  # TODO(mcandales): Calibrate.
                    cp.Rank.Ge(lambda deps: 3),
                    cp.Rank.Le(lambda deps: 5),
                    cp.Size.Ge(lambda deps, r, d: 0 if (d == 1 and not deps[0]) else 1),
                    # TODO(mcandales): Executorch specific constraint
                    # cp.Rank.In(lambda deps: [3, 4]),
                ],
            ),
            InPosArg(  # bias
                ArgType.TensorOpt,
                name="bias",
                deps=[1, 6, 8],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                    # cp.Dtype.In(
                    #    lambda deps: [deps[0].dtype] if deps[1] else dt._floating
                    # ),  # TODO(mcandales): Calibrate.
                    cp.Rank.Eq(lambda deps: 1),
                    cp.Size.Eq(
                        lambda deps, r, d: fn.conv_bias_size_eq(
                            deps[0], deps[1], deps[2]
                        )
                    ),
                ],
            ),
            InPosArg(  # stride
                ArgType.LengthList,
                name="stride",
                deps=[1],
                constraints=[
                    cp.Length.In(lambda deps: list({1, deps[0].dim() - 2})),
                    cp.Value.Ge(lambda deps, length, ix: 1),
                ],
            ),
            InPosArg(  # padding
                ArgType.LengthList,
                name="padding",
                deps=[1],
                constraints=[
                    cp.Length.In(lambda deps: list({1, deps[0].dim() - 2})),
                    cp.Value.Ge(lambda deps, length, ix: 0),
                ],
            ),
            InPosArg(  # dilation
                ArgType.LengthList,
                name="dilation",
                deps=[1],
                constraints=[
                    cp.Length.In(lambda deps: list({1, deps[0].dim() - 2})),
                    cp.Value.Ge(lambda deps, length, ix: 1),
                ],
            ),
            InPosArg(ArgType.Bool, name="transposed"),
            InPosArg(  # output_padding
                ArgType.LengthList,
                name="output_padding",
                deps=[1, 3, 5, 6],
                constraints=[
                    cp.Length.In(lambda deps: list({1, deps[0].dim() - 2})),
                    cp.Value.Ge(lambda deps, length, ix: 0),
                    cp.Value.Le(
                        lambda deps, length, ix: fn.conv_output_padding_max(
                            deps[1], deps[2], deps[3], length, ix
                        )
                    ),
                ],
            ),
            InPosArg(  # groups
                ArgType.Int,
                name="groups",
                deps=[1],
                constraints=[
                    cp.Value.In(
                        lambda deps: [
                            d
                            for d in range(1, fn.safe_size(deps[0], 0) + 1)
                            if fn.safe_size(deps[0], 0) % d == 0
                        ]
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: deps[0].dtype,
                    ),
                ],
            )
        ],
    ),
    Spec(
        op="copy.default",  # (Tensor self, Tensor src, bool non_blocking=False) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="src",
                deps=[0],
                constraints=[
                    cp.Rank.Le(lambda deps: deps[0].dim()),
                    cp.Size.In(lambda deps, r, d: fn.broadcast_to(deps[0].shape, r, d)),
                ],
            ),
            InPosArg(
                ArgType.Bool,
                name="non_blocking",
                # TODO(mcandales): Executorch specific constraint
                # constraints=[cp.Value.In(lambda deps: [False])],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="cos.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="cosh.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="cumsum.default",  # (Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=DimDefault,
            ),
            InKwArg(
                ArgType.ScalarTypeOpt,
                name="dtype",
                deps=[0],
                constraints=[
                    cp.Value.Ne(
                        lambda deps: (
                            torch.bool
                            if deps[0].dim() > 0 and deps[0].numel() > 0
                            else None
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="detach_copy.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="diagonal_copy.default",  # (Tensor self, int offset=0, int dim1=0, int dim2=1) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Rank.Ge(lambda deps: 2),
                ],
            ),
            InPosArg(ArgType.Int, name="offset"),
            InPosArg(
                ArgType.Dim,
                name="dim1",
                deps=[0],
                constraints=DimDefault,
            ),
            InPosArg(
                ArgType.Dim,
                name="dim2",
                deps=[0, 2],
                constraints=DimDefault
                + [
                    cp.Value.Ne(lambda deps: fn.normalize(deps[1], deps[0].dim())),
                    cp.Value.Ne(
                        lambda deps: fn.normalize(deps[1], deps[0].dim())
                        - deps[0].dim()
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            )
        ],
    ),
    Spec(
        op="div.Scalar",  # (Tensor self, Scalar other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(ArgType.Scalar, name="other"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[0].dtype in dt._floating
                            else torch.float32
                        )
                    ),
                ],
            )
        ],
    ),
    Spec(
        op="div.Scalar_mode",  # (Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Scalar,
                name="other",
                deps=[0, 2],
                constraints=[
                    cp.Dtype.Ne(
                        lambda deps: (
                            ScalarDtype.bool if deps[0].dtype == torch.bool else None
                        )
                    ),
                    cp.Value.Ne(
                        lambda deps, dtype: (
                            0
                            if deps[0].numel() > 0
                            and deps[1] is not None
                            and fn.promote_type_with_scalar_dtype(deps[0].dtype, dtype)
                            not in dt._floating
                            else None
                        )
                    ),
                ],
            ),
            InKwArg(
                ArgType.StringOpt,
                name="rounding_mode",
                constraints=[
                    cp.Value.In(lambda deps: ["trunc", "floor"]),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(
                        lambda deps: (
                            fn.promote_type_with_scalar(deps[0].dtype, deps[1])
                        )
                    ),
                ],
            )
        ],
    ),
    Spec(
        op="div.Tensor",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                ],
            ),
        ],
    ),
    Spec(
        op="div.Tensor_mode",  # (Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0, 2],
                constraints=[
                    cp.Dtype.Ne(
                        lambda deps: torch.bool if deps[0].dtype == torch.bool else None
                    ),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                    cp.Value.Ne(
                        lambda deps, dtype, struct: (
                            0
                            if (deps[0].numel() > 0 and math.prod(struct) > 0)
                            and deps[1] is not None
                            and torch.promote_types(deps[0].dtype, dtype)
                            not in dt._floating
                            else None
                        )
                    ),
                ],
            ),
            InKwArg(
                ArgType.StringOpt,
                name="rounding_mode",
                constraints=[
                    cp.Value.In(lambda deps: ["trunc", "floor"]),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(
                        lambda deps: dt.can_cast_from(
                            torch.promote_types(deps[0].dtype, deps[1].dtype)
                        )
                    ),
                ],
            ),
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="embedding.default",  # (Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="weight",
                constraints=[
                    cp.Rank.Eq(lambda deps: 2),
                    cp.Size.Ge(lambda deps, r, d: 1 if d == 0 else None),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="indices",
                deps=[0],
                constraints=[
                    cp.Dtype.In(lambda deps: [torch.long, torch.int]),
                    cp.Value.Ge(
                        lambda deps, dtype, struct: 0 if math.prod(struct) > 0 else None
                    ),
                    cp.Value.Le(
                        lambda deps, dtype, struct: (
                            max(fn.safe_size(deps[0], 0) - 1, 0)
                            if math.prod(struct) > 0
                            else None
                        )
                    ),
                ],
            ),
            InPosArg(ArgType.Int, name="padding_idx"),
            InPosArg(ArgType.Bool, name="scale_grad_by_freq"),
            InPosArg(ArgType.Bool, name="sparse"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="empty.memory_format",  # (SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Shape,
                name="size",
                constraints=ShapeDefault,
            ),
            InKwArg(
                ArgType.MemoryFormat,
                name="memory_format",
                constraints=MemoryFormatDefault,
            ),
            InKwArg(ArgType.ScalarTypeOpt, name="dtype"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="eq.Scalar",  # (Tensor self, Scalar other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(ArgType.Scalar, name="other"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="eq.Tensor",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="elu.default",  # (Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                ],
            ),
            InPosArg(ArgType.Scalar, name="alpha"),
            InPosArg(ArgType.Scalar, name="scale"),
            InPosArg(ArgType.Scalar, name="input_scale"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="erf.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="exp.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="expand_copy.default",  # (Tensor self, SymInt[] size, *, bool implicit=False) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Shape,
                name="size",
                deps=[0],
                constraints=[
                    cp.Length.Ge(lambda deps: deps[0].dim()),
                    cp.Value.Ge(lambda deps, length, ix: -1),
                    cp.Value.Ge(
                        lambda deps, length, ix: (
                            0 if ix < length - deps[0].dim() else None
                        )
                    ),
                    cp.Value.In(
                        lambda deps, length, ix: fn.expand_copy_size_in(
                            deps[0].shape, length, ix
                        )
                    ),
                ],
            ),
            InKwArg(
                ArgType.Bool,
                name="implicit",
                # TODO(mcandales): Executorch specific constraint
                # constraints=[cp.Value.In(lambda deps: [False])],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="expm1.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(lambda deps: dt.can_cast_from(torch.float)),
                    cp.Dtype.In(lambda deps: dt.can_cast_from(deps[0].dtype)),
                ],
            )
        ],
    ),
    Spec(
        op="fill.Scalar",  # (Tensor self, Scalar value) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Scalar,
                name="value",
                deps=[0],
                constraints=[
                    cp.Value.Ge(
                        lambda deps, dtype: fn.dtype_lower_bound(deps[0].dtype)
                    ),
                    cp.Value.Le(
                        lambda deps, dtype: fn.dtype_upper_bound(deps[0].dtype)
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="fill.Tensor",  # (Tensor self, Tensor value) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="value",
                constraints=[
                    cp.Rank.Eq(lambda deps: 0),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="flip.default",  # (Tensor self, int[] dims) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.DimList,
                name="dims",
                deps=[0],
                constraints=DimListDefault,
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            )
        ],
    ),
    Spec(
        op="floor.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="floor_divide.default",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Dtype.Ne(
                        lambda deps: torch.bool if deps[0].dtype == torch.bool else None
                    ),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                    cp.Value.Ne(
                        lambda deps, dtype, struct: (
                            0
                            if (deps[0].numel() > 0 and math.prod(struct) > 0)
                            and torch.promote_types(deps[0].dtype, dtype)
                            not in dt._floating
                            else None
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="fmod.Tensor",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Dtype.Ne(
                        lambda deps: torch.bool if deps[0].dtype == torch.bool else None
                    ),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                    cp.Value.Ne(
                        lambda deps, dtype, struct: (
                            0
                            if (deps[0].numel() > 0 and math.prod(struct) > 0)
                            and torch.promote_types(deps[0].dtype, dtype)
                            not in dt._floating
                            else None
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="fmod.Scalar",  # (Tensor self, Scalar other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Scalar,
                name="other",
                deps=[0],
                constraints=[
                    cp.Dtype.Ne(
                        lambda deps: (
                            ScalarDtype.bool if deps[0].dtype == torch.bool else None
                        )
                    ),
                    cp.Value.Ne(
                        lambda deps, dtype: (
                            0
                            if (
                                deps[0].numel() > 0
                                and fn.promote_type_with_scalar(deps[0].dtype, dtype)
                                not in dt._floating
                            )
                            else None
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="full.default",  # (SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Shape,
                name="size",
                constraints=ShapeDefault,
            ),
            InPosArg(
                ArgType.Scalar,
                name="fill_value",
                deps=[2],
                constraints=[
                    cp.Value.Ge(lambda deps, dtype: fn.dtype_lower_bound(deps[0])),
                    cp.Value.Le(lambda deps, dtype: fn.dtype_upper_bound(deps[0])),
                ],
            ),
            InKwArg(ArgType.ScalarTypeOpt, name="dtype"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="full_like.default",  # (Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Scalar,
                name="fill_value",
                deps=[0, 2],
                constraints=[
                    cp.Value.Ge(
                        lambda deps, dtype: fn.dtype_lower_bound(
                            deps[0].dtype if deps[1] is None else deps[1]
                        )
                    ),
                    cp.Value.Le(
                        lambda deps, dtype: fn.dtype_upper_bound(
                            deps[0].dtype if deps[1] is None else deps[1]
                        )
                    ),
                ],
            ),
            InKwArg(ArgType.ScalarTypeOpt, name="dtype"),
            InKwArg(
                ArgType.MemoryFormat,
                name="memory_format",
                constraints=MemoryFormatDefault,
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="gather.default",  # (Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=[
                    cp.Value.In(lambda deps: fn.dim_non_zero_size(deps[0])),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="index",
                deps=[0, 1],
                # TODO(mcandales) Handle index.numel() == 0 case
                constraints=[
                    cp.Dtype.Eq(lambda deps: torch.long),
                    cp.Rank.Eq(
                        lambda deps: deps[0].dim() if deps[0].dim() >= 2 else None
                    ),
                    cp.Rank.In(
                        lambda deps: [0, 1] if deps[0].dim() in [0, 1] else None
                    ),
                    cp.Size.Le(
                        lambda deps, r, d: (
                            fn.safe_size(deps[0], d)
                            if d != fn.normalize(deps[1], deps[0].dim())
                            else None
                        )
                    ),
                    cp.Value.Ge(lambda deps, dtype, struct: 0),
                    cp.Value.Le(
                        lambda deps, dtype, struct: (
                            0
                            if deps[0].dim() == 0
                            else max(0, fn.safe_size(deps[0], deps[1]) - 1)
                        )
                    ),
                ],
            ),
            InKwArg(ArgType.Bool, name="sparse_grad"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            ),
        ],
    ),
    Spec(
        op="ge.Scalar",  # (Tensor self, Scalar other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(ArgType.Scalar, name="other"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="ge.Tensor",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="gelu.default",  # (Tensor self, *, str approximate="none") -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[cp.Dtype.In(lambda deps: dt._floating)],
            ),
            InKwArg(
                ArgType.String,
                name="approximate",
                constraints=[
                    cp.Value.In(lambda deps: ["none", "tanh"]),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="glu.default",  # (Tensor self, int dim=-1) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                    cp.Rank.Ge(lambda deps: 1),
                ],
            ),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=[
                    cp.Value.In(
                        lambda deps: [
                            d
                            for d in range(-deps[0].dim(), deps[0].dim())
                            if fn.safe_size(deps[0], d) % 2 == 0
                        ]
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="gt.Scalar",  # (Tensor self, Scalar other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(ArgType.Scalar, name="other"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="gt.Tensor",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="hardtanh.default",  # (Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                ],
            ),
            InPosArg(
                ArgType.Scalar,
                name="min_val",
                deps=[0],
                constraints=[
                    cp.Value.Gt(
                        lambda deps, dtype: fn.dtype_strict_lower_bound(deps[0].dtype)
                    ),
                    cp.Value.Lt(
                        lambda deps, dtype: fn.dtype_strict_upper_bound(deps[0].dtype)
                    ),
                ],
            ),
            InPosArg(
                ArgType.Scalar,
                name="max_val",
                deps=[0],
                constraints=[
                    cp.Value.Gt(
                        lambda deps, dtype: fn.dtype_strict_lower_bound(deps[0].dtype)
                    ),
                    cp.Value.Lt(
                        lambda deps, dtype: fn.dtype_strict_upper_bound(deps[0].dtype)
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="index_select.default",  # (Tensor self, int dim, Tensor index) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=[
                    cp.Value.In(lambda deps: fn.dim_non_zero_size(deps[0])),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="index",
                deps=[0, 1],
                constraints=[
                    cp.Dtype.In(lambda deps: [torch.int, torch.long]),
                    cp.Rank.Le(lambda deps: 1),
                    cp.Size.Eq(lambda deps, r, d: 1 if deps[0].dim() == 0 else None),
                    cp.Value.Ge(lambda deps, dtype, struct: 0),
                    cp.Value.Le(
                        lambda deps, dtype, struct: (
                            0
                            if deps[0].dim() == 0
                            else fn.safe_size(deps[0], deps[1]) - 1
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="isinf.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="isnan.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="le.Scalar",  # (Tensor self, Scalar other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(ArgType.Scalar, name="other"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="le.Tensor",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="leaky_relu.default",  # (Tensor self, Scalar negative_slope=0.01) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                ],
            ),
            InPosArg(ArgType.Scalar, name="negative_slope"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="lift_fresh_copy.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="log.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(lambda deps: dt.can_cast_from(torch.float)),
                    cp.Dtype.In(lambda deps: dt.can_cast_from(deps[0].dtype)),
                ],
            )
        ],
    ),
    Spec(
        op="log10.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(lambda deps: dt.can_cast_from(torch.float)),
                    cp.Dtype.In(lambda deps: dt.can_cast_from(deps[0].dtype)),
                ],
            )
        ],
    ),
    Spec(
        op="log1p.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(lambda deps: dt.can_cast_from(torch.float)),
                    cp.Dtype.In(lambda deps: dt.can_cast_from(deps[0].dtype)),
                ],
            )
        ],
    ),
    Spec(
        op="log2.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(lambda deps: dt.can_cast_from(torch.float)),
                    cp.Dtype.In(lambda deps: dt.can_cast_from(deps[0].dtype)),
                ],
            )
        ],
    ),
    Spec(
        op="logical_and.default",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="logical_not.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="logical_or.default",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="logical_xor.default",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="logit.default",  # (Tensor self, float? eps=None) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(ArgType.FloatOpt, name="eps"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="lt.Scalar",  # (Tensor self, Scalar other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(ArgType.Scalar, name="other"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="lt.Tensor",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="masked_fill.Scalar",  # (Tensor self, Tensor mask, Scalar value) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="mask",
                deps=[0],
                constraints=[
                    cp.Dtype.Eq(lambda deps: torch.bool),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
            InPosArg(
                ArgType.Scalar,
                name="value",
                deps=[0],
                constraints=[
                    cp.Value.Ge(
                        lambda deps, dtype: fn.dtype_lower_bound(deps[0].dtype)
                    ),
                    cp.Value.Le(
                        lambda deps, dtype: fn.dtype_upper_bound(deps[0].dtype)
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    # TODO(mcandales): InputGen always generates tensor values after generating the entire tuple of
    # meta arguments. However, for this op, we need to generate the mask values before the source shape
    Spec(
        op="masked_scatter.default",  # (Tensor self, Tensor mask, Tensor source) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
            ),
            InPosArg(
                ArgType.Tensor,
                name="mask",
                deps=[0],
                constraints=[
                    cp.Dtype.Eq(lambda deps: torch.bool),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="source",
                deps=[0, 1],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                    # The constraints below are a hack to ensure that the source tensor has enough elements
                    cp.Rank.Ge(lambda deps: 1),
                    cp.Size.Ge(
                        lambda deps, r, d: (
                            math.prod(
                                fn.broadcasted_shape(deps[0].shape, deps[1].shape)
                            )
                            if d == 0
                            else None
                        )
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="masked_select.default",  # (Tensor self, Tensor mask) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
            ),
            InPosArg(
                ArgType.Tensor,
                name="mask",
                deps=[0],
                constraints=[
                    cp.Dtype.Eq(lambda deps: torch.bool),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="max.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="max.dim",  # (Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=[
                    cp.Value.In(lambda deps: fn.dim_non_zero_size(deps[0])),
                ],
            ),
            InPosArg(ArgType.Bool, name="keepdim"),
        ],
        outspec=[
            OutArg(ArgType.Tensor, name="values"),
            OutArg(ArgType.Tensor, name="indices"),
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="max_pool2d_with_indices.default",  # (Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
        inspec=[
            InPosArg(  # self
                ArgType.Tensor,
                name="self",
                deps=[1, 2, 3, 4, 5],
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                    cp.Rank.In(lambda deps: [3, 4]),
                    # cp.Size.Ge(lambda deps, r, d: 0 if d == 0 and r == 4 else 1),
                    cp.Size.Ge(
                        lambda deps, r, d: fn.pool_input_size_min(
                            2, deps[0], deps[1], deps[2], deps[3], deps[4], r, d
                        )
                    ),
                ],
            ),
            InPosArg(  # kernel_size
                ArgType.LengthList,
                name="kernel_size",
                constraints=[
                    cp.Length.In(lambda deps: [1, 2]),
                    cp.Value.Ge(lambda deps, length, ix: 1),
                ],
            ),
            InPosArg(  # stride
                ArgType.LengthList,
                name="stride",
                constraints=[
                    cp.Length.In(lambda deps: [0, 1, 2]),
                    cp.Value.Ge(lambda deps, length, ix: 1),
                ],
            ),
            InPosArg(  # padding
                ArgType.LengthList,
                name="padding",
                deps=[1],
                constraints=[
                    cp.Length.In(lambda deps: [1, 2]),
                    cp.Value.Ge(lambda deps, length, ix: 0),
                    cp.Value.Le(
                        lambda deps, length, ix: fn.pool_padding_max(
                            deps[0], length, ix
                        )
                    ),
                ],
            ),
            InPosArg(  # dilation
                ArgType.LengthList,
                name="dilation",
                constraints=[
                    cp.Length.In(lambda deps: [1, 2]),
                    cp.Value.Ge(lambda deps, length, ix: 1),
                ],
            ),
            InPosArg(ArgType.Bool, name="ceil_mode"),  # ceil_mode
        ],
        outspec=[
            OutArg(ArgType.Tensor, name="out"),
            OutArg(ArgType.Tensor, name="indices"),
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="max_pool3d_with_indices.default",  # (Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
        inspec=[
            InPosArg(  # self
                ArgType.Tensor,
                name="self",
                deps=[1, 2, 3, 4, 5],
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                    cp.Rank.In(lambda deps: [4, 5]),
                    # cp.Size.Ge(lambda deps, r, d: 0 if d == 0 and r == 5 else 1),
                    cp.Size.Ge(
                        lambda deps, r, d: fn.pool_input_size_min(
                            3, deps[0], deps[1], deps[2], deps[3], deps[4], r, d
                        )
                    ),
                ],
            ),
            InPosArg(  # kernel_size
                ArgType.LengthList,
                name="kernel_size",
                constraints=[
                    cp.Length.In(lambda deps: [1, 3]),
                    cp.Value.Ge(lambda deps, length, ix: 1),
                ],
            ),
            InPosArg(  # stride
                ArgType.LengthList,
                name="stride",
                constraints=[
                    cp.Length.In(lambda deps: [0, 1, 3]),
                    cp.Value.Ge(lambda deps, length, ix: 1),
                ],
            ),
            InPosArg(  # padding
                ArgType.LengthList,
                name="padding",
                deps=[1],
                constraints=[
                    cp.Length.In(lambda deps: [1, 3]),
                    cp.Value.Ge(lambda deps, length, ix: 0),
                    cp.Value.Le(
                        lambda deps, length, ix: fn.pool_padding_max(
                            deps[0], length, ix
                        )
                    ),
                ],
            ),
            InPosArg(  # dilation
                ArgType.LengthList,
                name="dilation",
                constraints=[
                    cp.Length.In(lambda deps: [1, 3]),
                    cp.Value.Ge(lambda deps, length, ix: 1),
                ],
            ),
            InPosArg(ArgType.Bool, name="ceil_mode"),  # ceil_mode
        ],
        outspec=[
            OutArg(ArgType.Tensor, name="out"),
            OutArg(ArgType.Tensor, name="indices"),
        ],
    ),
    Spec(
        op="maximum.default",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(
                        lambda deps: dt.can_cast_from(
                            torch.promote_types(deps[0].dtype, deps[1].dtype)
                        )
                    ),
                ],
            )
        ],
    ),
    Spec(
        op="mean.dim",  # (Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.DimListOpt,
                name="dim",
                deps=[0],
                constraints=DimListDefault,
            ),
            InPosArg(ArgType.Bool, name="keepdim"),
            InKwArg(
                ArgType.ScalarTypeOpt,
                name="dtype",
                deps=[0],
                constraints=[
                    cp.Optional.Eq(
                        lambda deps: (
                            False if deps[0].dtype not in dt._floating else None
                        )
                    ),
                    cp.Value.In(lambda deps: dt._floating),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="mean.default",  # (Tensor self, *, ScalarType? dtype=None) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
            ),
            InKwArg(
                ArgType.ScalarTypeOpt,
                name="dtype",
                deps=[0],
                constraints=[
                    cp.Optional.Eq(
                        lambda deps: (
                            False if deps[0].dtype not in dt._floating else None
                        )
                    ),
                    cp.Value.In(lambda deps: dt._floating),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="min.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="min.dim",  # (Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=[
                    cp.Value.In(lambda deps: fn.dim_non_zero_size(deps[0])),
                ],
            ),
            InPosArg(ArgType.Bool, name="keepdim"),
        ],
        outspec=[
            OutArg(ArgType.Tensor, name="values"),
            OutArg(ArgType.Tensor, name="indices"),
        ],
    ),
    Spec(
        op="minimum.default",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(
                        lambda deps: dt.can_cast_from(
                            torch.promote_types(deps[0].dtype, deps[1].dtype)
                        )
                    ),
                ],
            )
        ],
    ),
    Spec(
        op="mm.default",  # (Tensor self, Tensor mat2) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                    cp.Rank.Eq(lambda deps: 2),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="mat2",
                deps=[0],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                    cp.Rank.Eq(lambda deps: 2),
                    cp.Size.Eq(
                        lambda deps, r, d: fn.safe_size(deps[0], 1) if d == 0 else None
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="mul.Tensor",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="mul.Scalar",  # (Tensor self, Scalar other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(ArgType.Scalar, name="other"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="native_group_norm.default",  # (Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="input",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                    cp.Rank.Ge(lambda deps: 2),
                ],
            ),
            InPosArg(
                ArgType.TensorOpt,
                name="weight",
                deps=[0],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                    cp.Rank.Eq(lambda deps: 1),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_size(deps[0], 1)),
                ],
            ),
            InPosArg(
                ArgType.TensorOpt,
                name="bias",
                deps=[0],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                    cp.Rank.Eq(lambda deps: 1),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_size(deps[0], 1)),
                ],
            ),
            InPosArg(
                ArgType.Int,
                name="N",
                deps=[0],
                constraints=[
                    cp.Value.Eq(lambda deps: fn.safe_size(deps[0], 0)),
                ],
            ),
            InPosArg(
                ArgType.Int,
                name="C",
                deps=[0],
                constraints=[
                    cp.Value.Eq(lambda deps: fn.safe_size(deps[0], 1)),
                ],
            ),
            InPosArg(
                ArgType.Int,
                name="HxW",
                deps=[0],
                constraints=[
                    cp.Value.Eq(
                        lambda deps: math.prod(
                            fn.safe_size(deps[0], d) for d in range(2, deps[0].dim())
                        ),
                    ),
                ],
            ),
            InPosArg(
                ArgType.Int,
                name="groups",
                deps=[0],
                constraints=[
                    cp.Value.In(
                        lambda deps: [
                            d
                            for d in range(1, fn.safe_size(deps[0], 1) + 1)
                            if fn.safe_size(deps[0], 1) % d == 0
                        ]
                    ),
                ],
            ),
            InPosArg(ArgType.Float, name="eps"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                name="out0",
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            ),
            OutArg(
                ArgType.Tensor,
                name="out1",
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            ),
            OutArg(
                ArgType.Tensor,
                name="out2",
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            ),
        ],
    ),
    Spec(
        op="native_layer_norm.default",  # (Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="input",
                deps=[1],
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                    cp.Rank.Ge(lambda deps: len(deps[0])),
                    cp.Size.Eq(lambda deps, r, d: fn.nlm_input_size(deps[0], r, d)),
                ],
            ),
            InPosArg(
                ArgType.Shape,
                name="normalized_shape",
                constraints=[
                    cp.Length.Ge(lambda deps: 1),
                    cp.Value.Ge(lambda deps, length, ix: 0),
                ],
            ),
            InPosArg(
                ArgType.TensorOpt,
                name="weight",
                deps=[0, 1],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                    cp.Rank.Eq(lambda deps: len(deps[1])),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_ix(deps[1], d)),
                ],
            ),
            InPosArg(
                ArgType.TensorOpt,
                name="bias",
                deps=[0, 1],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                    cp.Rank.Eq(lambda deps: len(deps[1])),
                    cp.Size.Eq(lambda deps, r, d: fn.safe_ix(deps[1], d)),
                ],
            ),
            InPosArg(ArgType.Float, name="eps"),
        ],
        outspec=[
            OutArg(ArgType.Tensor, name="out0"),
            OutArg(ArgType.Tensor, name="out1"),
            OutArg(ArgType.Tensor, name="out2"),
        ],
    ),
    Spec(
        op="native_dropout.default",  # (Tensor input, float p, bool? train) -> (Tensor, Tensor)
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="input",
                deps=[2],
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating if deps[0] else None),
                ],
            ),
            InPosArg(
                ArgType.Float,
                name="p",
                constraints=[
                    cp.Value.Ge(lambda deps: 0.0),
                    cp.Value.Le(lambda deps: 1.0),
                ],
            ),
            InPosArg(
                ArgType.Bool,
                name="train",
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor, name="output"),
            OutArg(ArgType.Tensor, name="mask"),
        ],
    ),
    Spec(
        op="narrow_copy.default",  # (Tensor self, int dim, SymInt start, SymInt length) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Rank.Ge(lambda deps: 1),
                ],
            ),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=DimDefault,
            ),
            InPosArg(
                ArgType.Index,
                name="start",
                deps=[0, 1],
                constraints=[
                    cp.Value.Ge(lambda deps: -fn.safe_size(deps[0], deps[1])),
                    cp.Value.Le(lambda deps: fn.safe_size(deps[0], deps[1])),
                ],
            ),
            InPosArg(
                ArgType.Length,
                name="length",
                deps=[0, 1, 2],
                constraints=[
                    cp.Value.Ge(lambda deps: 0),
                    cp.Value.Le(
                        lambda deps: (
                            0
                            if deps[2] == fn.safe_size(deps[0], deps[1])
                            else fn.safe_size(deps[0], deps[1])
                            - fn.normalize(deps[2], fn.safe_size(deps[0], deps[1]))
                        )
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="ne.Scalar",  # (Tensor self, Scalar other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(ArgType.Scalar, name="other"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="ne.Tensor",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="neg.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="nonzero.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="ones.default",  # (SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Shape,
                name="size",
                constraints=ShapeDefault,
            ),
            InKwArg(ArgType.ScalarTypeOpt, name="dtype"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="permute_copy.default",  # (Tensor self, int[] dims) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.DimList,
                name="dims",
                deps=[0],
                constraints=DimListDefault
                + [
                    cp.Length.Eq(lambda deps: deps[0].dim()),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="pixel_shuffle.default",  # (Tensor self, int upscale_factor) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Rank.Ge(lambda deps: 3),
                    cp.Size.Be(
                        lambda deps, r, d: (
                            [
                                0,
                                1,
                                2,
                                3,
                                4,
                                8,
                                9,
                                12,
                                16,
                                18,
                                25,
                            ]
                            if d == r - 3
                            else None
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.Int,
                name="upscale_factor",
                deps=[0],
                constraints=[
                    cp.Value.In(
                        lambda deps: [
                            d
                            for d in range(1, fn.safe_size(deps[0], -3) + 1)
                            if fn.safe_size(deps[0], -3) % (d * d) == 0
                        ]
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="pixel_unshuffle.default",  # (Tensor self, int downscale_factor) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Rank.Ge(lambda deps: 3),  # At least 3D tensor
                    cp.Size.Be(
                        lambda deps, r, d: (
                            [
                                0,
                                1,
                                2,
                                3,
                                4,
                                8,
                                9,
                                12,
                                15,
                                16,
                                18,
                                25,
                            ]
                            if d == r - 1 or d == r - 2
                            else None
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.Int,
                name="downscale_factor",
                deps=[0],
                constraints=[
                    cp.Value.Ge(lambda deps: 1),
                    cp.Value.In(
                        lambda deps: [
                            d
                            for d in range(
                                1,
                                max(
                                    fn.safe_size(deps[0], -1), fn.safe_size(deps[0], -2)
                                )
                                + 1,
                            )
                            if fn.safe_size(deps[0], -1) % d == 0
                            and fn.safe_size(deps[0], -2) % d == 0
                        ]
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="pow.Scalar",  # (Scalar self, Tensor exponent) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Scalar,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: ScalarDtype.bool),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="exponent",
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="pow.Tensor_Scalar",  # (Tensor self, Scalar exponent) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Scalar,
                name="exponent",
                deps=[0],
                constraints=[
                    cp.Dtype.Ne(
                        lambda deps: (
                            ScalarDtype.bool if deps[0].dtype == torch.bool else None
                        )
                    ),
                    cp.Value.Ge(
                        lambda deps, dtype: 0 if dtype == ScalarDtype.int else None
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="pow.Tensor_Tensor",  # (Tensor self, Tensor exponent) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="exponent",
                deps=[0],
                constraints=[
                    cp.Dtype.Ne(
                        lambda deps: torch.bool if deps[0].dtype == torch.bool else None
                    ),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="prod.default",  # (Tensor self, *, ScalarType? dtype=None) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InKwArg(ArgType.ScalarTypeOpt, name="dtype"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                deps=[0, 1],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[1] if deps[1] is not None else None),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.long
                            if deps[1] is None and deps[0].dtype in dt._int_and_bool
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[1] is None and deps[0].dtype not in dt._int_and_bool
                            else None
                        )
                    ),
                ],
            ),
        ],
    ),
    Spec(
        op="prod.dim_int",  # (Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=DimDefault,
            ),
            InPosArg(ArgType.Bool, name="keepdim"),
            InKwArg(ArgType.ScalarTypeOpt, name="dtype"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[3] if deps[3] is not None else None),
                    cp.Dtype.Eq(
                        lambda deps: (
                            torch.long
                            if deps[3] is None and deps[0].dtype in dt._int_and_bool
                            else None
                        )
                    ),
                    cp.Dtype.Eq(
                        lambda deps: (
                            deps[0].dtype
                            if deps[3] is None and deps[0].dtype not in dt._int_and_bool
                            else None
                        )
                    ),
                ],
            ),
        ],
    ),
    Spec(
        op="rand.default",  # (SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Shape,
                name="size",
                constraints=ShapeDefault,
            ),
            InKwArg(
                ArgType.ScalarTypeOpt,
                name="dtype",
                constraints=[
                    cp.Value.In(lambda deps: dt._floating),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="randn.default",  # (SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Shape,
                name="size",
                constraints=ShapeDefault,
            ),
            InKwArg(
                ArgType.ScalarTypeOpt,
                name="dtype",
                constraints=[
                    cp.Value.In(lambda deps: dt._floating),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="reciprocal.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="reflection_pad1d.default",  # (Tensor self, SymInt[2] padding) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                deps=[1],
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                    cp.Rank.In(lambda deps: [2, 3]),
                    cp.Size.Gt(lambda deps, r, d: None if d == 0 and r == 3 else 0),
                    # padding[0] < self.size(-1)
                    cp.Size.Gt(lambda deps, r, d: deps[0][0] if d == r - 1 else None),
                    # padding[1] < self.size(-1)
                    cp.Size.Gt(lambda deps, r, d: deps[0][1] if d == r - 1 else None),
                    # padding[0] + padding[1] + self.size(-1) > 0
                    cp.Size.Gt(
                        lambda deps, r, d: (
                            -(deps[0][0] + deps[0][1]) if d == r - 1 else None
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.LengthList,
                name="padding",
                constraints=[
                    cp.Length.Eq(lambda deps: 2),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            )
        ],
    ),
    Spec(
        op="reflection_pad2d.default",  # (Tensor self, SymInt[2] padding) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                deps=[1],
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                    cp.Rank.In(lambda deps: [3, 4]),
                    cp.Size.Gt(lambda deps, r, d: None if d == 0 and r == 4 else 0),
                    # padding[0] < self.size(-1)
                    cp.Size.Gt(lambda deps, r, d: deps[0][0] if d == r - 1 else None),
                    # padding[1] < self.size(-1)
                    cp.Size.Gt(lambda deps, r, d: deps[0][1] if d == r - 1 else None),
                    # padding[0] + padding[1] + self.size(-1) > 0
                    cp.Size.Gt(
                        lambda deps, r, d: (
                            -(deps[0][0] + deps[0][1]) if d == r - 1 else None
                        )
                    ),
                    # padding[2] < self.size(-2)
                    cp.Size.Gt(lambda deps, r, d: deps[0][2] if d == r - 2 else None),
                    # padding[3] < self.size(-2)
                    cp.Size.Gt(lambda deps, r, d: deps[0][3] if d == r - 2 else None),
                    # padding[2] + padding[3] + self.size(-2) > 0
                    cp.Size.Gt(
                        lambda deps, r, d: (
                            -(deps[0][2] + deps[0][3]) if d == r - 2 else None
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.LengthList,
                name="padding",
                constraints=[
                    cp.Length.Eq(lambda deps: 4),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            )
        ],
    ),
    Spec(
        op="reflection_pad3d.default",  # (Tensor self, SymInt[2] padding) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                deps=[1],
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                    cp.Rank.In(lambda deps: [4, 5]),
                    cp.Size.Gt(lambda deps, r, d: None if d == 0 and r == 5 else 0),
                    # padding[0] < self.size(-1)
                    cp.Size.Gt(lambda deps, r, d: deps[0][0] if d == r - 1 else None),
                    # padding[1] < self.size(-1)
                    cp.Size.Gt(lambda deps, r, d: deps[0][1] if d == r - 1 else None),
                    # padding[0] + padding[1] + self.size(-1) > 0
                    cp.Size.Gt(
                        lambda deps, r, d: (
                            -(deps[0][0] + deps[0][1]) if d == r - 1 else None
                        )
                    ),
                    # padding[2] < self.size(-2)
                    cp.Size.Gt(lambda deps, r, d: deps[0][2] if d == r - 2 else None),
                    # padding[3] < self.size(-2)
                    cp.Size.Gt(lambda deps, r, d: deps[0][3] if d == r - 2 else None),
                    # padding[2] + padding[3] + self.size(-2) > 0
                    cp.Size.Gt(
                        lambda deps, r, d: (
                            -(deps[0][2] + deps[0][3]) if d == r - 2 else None
                        )
                    ),
                    # padding[4] < self.size(-3)
                    cp.Size.Gt(lambda deps, r, d: deps[0][4] if d == r - 3 else None),
                    # padding[5] < self.size(-3)
                    cp.Size.Gt(lambda deps, r, d: deps[0][5] if d == r - 3 else None),
                    # padding[4] + padding[5] + self.size(-3) > 0
                    cp.Size.Gt(
                        lambda deps, r, d: (
                            -(deps[0][4] + deps[0][5]) if d == r - 3 else None
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.LengthList,
                name="padding",
                constraints=[
                    cp.Length.Eq(lambda deps: 6),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            )
        ],
    ),
    Spec(
        op="relu.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="remainder.Tensor",  # (Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Dtype.Ne(
                        lambda deps: torch.bool if deps[0].dtype == torch.bool else None
                    ),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                    cp.Value.Ne(
                        lambda deps, dtype, struct: (
                            0
                            if (deps[0].numel() > 0 and math.prod(struct) > 0)
                            and torch.promote_types(deps[0].dtype, dtype)
                            not in dt._floating
                            else None
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="remainder.Scalar",  # (Tensor self, Scalar other) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Scalar,
                name="other",
                deps=[0],
                constraints=[
                    cp.Dtype.Ne(
                        lambda deps: (
                            ScalarDtype.bool if deps[0].dtype == torch.bool else None
                        )
                    ),
                    cp.Value.Ne(
                        lambda deps, dtype: (
                            0
                            if (
                                deps[0].numel() > 0
                                and fn.promote_type_with_scalar(deps[0].dtype, dtype)
                                not in dt._floating
                            )
                            else None
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="repeat.default",  # (Tensor self, SymInt[] repeats) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.LengthList,
                name="repeats",
                deps=[0],
                constraints=[
                    cp.Length.Ge(lambda deps: deps[0].dim()),
                    cp.Value.Be(lambda deps, length, ix: [0, 1, 2, 3]),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    # TODO(mcandales): InputGen always generates tensor values after generating the entire tuple of
    # meta arguments. However, for this op, we need to generate the tensor values before the output_size
    Spec(
        op="repeat_interleave.Tensor",  # (Tensor repeats, *, SymInt? output_size=None) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="repeats",
                constraints=[
                    cp.Dtype.In(lambda deps: [torch.int, torch.long]),
                    cp.Rank.Eq(lambda deps: 1),
                    cp.Value.Ge(lambda deps, dtype, struct: 0),
                    cp.Value.Eq(lambda deps, dtype, struct: 3),
                    # This is a hack to make sure output_size == repeats.sum()
                ],
            ),
            InKwArg(
                ArgType.IntOpt,
                name="output_size",
                deps=[0],
                constraints=[
                    # cp.Value.Eq(lambda deps: deps[0].sum().item()),
                    cp.Value.Eq(lambda deps: 3 * fn.safe_size(deps[0], 0)),
                    # This is a hack to make sure output_size == repeats.sum()
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="replication_pad1d.default",  # (Tensor self, SymInt[2] padding) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                deps=[1],
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                    cp.Rank.In(lambda deps: [2, 3]),
                    cp.Size.Gt(lambda deps, r, d: None if d == 0 and r == 3 else 0),
                    # padding[0] < self.size(-1)
                    cp.Size.Gt(lambda deps, r, d: deps[0][0] if d == r - 1 else None),
                    # padding[1] < self.size(-1)
                    cp.Size.Gt(lambda deps, r, d: deps[0][1] if d == r - 1 else None),
                    # padding[0] + padding[1] + self.size(-1) > 0
                    cp.Size.Gt(
                        lambda deps, r, d: (
                            -(deps[0][0] + deps[0][1]) if d == r - 1 else None
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.LengthList,
                name="padding",
                constraints=[
                    cp.Length.Eq(lambda deps: 2),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            )
        ],
    ),
    Spec(
        op="replication_pad2d.default",  # (Tensor self, SymInt[2] padding) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                deps=[1],
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                    cp.Rank.In(lambda deps: [3, 4]),
                    cp.Size.Gt(lambda deps, r, d: None if d == 0 and r == 4 else 0),
                    # padding[0] < self.size(-1)
                    cp.Size.Gt(lambda deps, r, d: deps[0][0] if d == r - 1 else None),
                    # padding[1] < self.size(-1)
                    cp.Size.Gt(lambda deps, r, d: deps[0][1] if d == r - 1 else None),
                    # padding[0] + padding[1] + self.size(-1) > 0
                    cp.Size.Gt(
                        lambda deps, r, d: (
                            -(deps[0][0] + deps[0][1]) if d == r - 1 else None
                        )
                    ),
                    # padding[2] < self.size(-2)
                    cp.Size.Gt(lambda deps, r, d: deps[0][2] if d == r - 2 else None),
                    # padding[3] < self.size(-2)
                    cp.Size.Gt(lambda deps, r, d: deps[0][3] if d == r - 2 else None),
                    # padding[2] + padding[3] + self.size(-2) > 0
                    cp.Size.Gt(
                        lambda deps, r, d: (
                            -(deps[0][2] + deps[0][3]) if d == r - 2 else None
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.LengthList,
                name="padding",
                constraints=[
                    cp.Length.Eq(lambda deps: 4),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            )
        ],
    ),
    Spec(
        op="replication_pad3d.default",  # (Tensor self, SymInt[2] padding) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                deps=[1],
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                    cp.Rank.In(lambda deps: [4, 5]),
                    cp.Size.Gt(lambda deps, r, d: None if d == 0 and r == 5 else 0),
                    # padding[0] < self.size(-1)
                    cp.Size.Gt(lambda deps, r, d: deps[0][0] if d == r - 1 else None),
                    # padding[1] < self.size(-1)
                    cp.Size.Gt(lambda deps, r, d: deps[0][1] if d == r - 1 else None),
                    # padding[0] + padding[1] + self.size(-1) > 0
                    cp.Size.Gt(
                        lambda deps, r, d: (
                            -(deps[0][0] + deps[0][1]) if d == r - 1 else None
                        )
                    ),
                    # padding[2] < self.size(-2)
                    cp.Size.Gt(lambda deps, r, d: deps[0][2] if d == r - 2 else None),
                    # padding[3] < self.size(-2)
                    cp.Size.Gt(lambda deps, r, d: deps[0][3] if d == r - 2 else None),
                    # padding[2] + padding[3] + self.size(-2) > 0
                    cp.Size.Gt(
                        lambda deps, r, d: (
                            -(deps[0][2] + deps[0][3]) if d == r - 2 else None
                        )
                    ),
                    # padding[4] < self.size(-3)
                    cp.Size.Gt(lambda deps, r, d: deps[0][4] if d == r - 3 else None),
                    # padding[5] < self.size(-3)
                    cp.Size.Gt(lambda deps, r, d: deps[0][5] if d == r - 3 else None),
                    # padding[4] + padding[5] + self.size(-3) > 0
                    cp.Size.Gt(
                        lambda deps, r, d: (
                            -(deps[0][4] + deps[0][5]) if d == r - 3 else None
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.LengthList,
                name="padding",
                constraints=[
                    cp.Length.Eq(lambda deps: 6),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            )
        ],
    ),
    Spec(
        op="round.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="roll.default",  # (Tensor self, SymInt[1] shifts, int[1] dims=[]) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Rank.Ge(lambda deps: 1),
                ],
            ),
            InPosArg(
                ArgType.LengthList,
                name="shifts",
                constraints=[
                    cp.Length.Ge(lambda deps: 1),
                ],
            ),
            InPosArg(
                ArgType.DimList,
                name="dims",
                deps=[0, 1],
                constraints=[
                    cp.Length.Eq(lambda deps: len(deps[1])),
                    cp.Value.Ge(
                        lambda deps, length, ix: (
                            -deps[0].dim()
                            if deps[0].dim() > 0 and deps[0].numel() != 0
                            else None
                        )
                    ),
                    cp.Value.Ge(
                        lambda deps, length, ix: (
                            -1 if deps[0].dim() == 0 and deps[0].numel() != 0 else None
                        )
                    ),
                    cp.Value.Le(
                        lambda deps, length, ix: (
                            deps[0].dim() - 1
                            if deps[0].dim() > 0 and deps[0].numel() != 0
                            else None
                        )
                    ),
                    cp.Value.Le(
                        lambda deps, length, ix: (
                            0 if deps[0].dim() == 0 and deps[0].numel() != 0 else None
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            )
        ],
    ),
    Spec(
        op="rsqrt.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="rsub.Scalar",  # (Tensor self, Scalar other, Scalar alpha=1) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                ],
            ),
            InPosArg(
                ArgType.Scalar,
                name="other",
                constraints=[
                    cp.Dtype.Ne(lambda deps: ScalarDtype.bool),
                ],
            ),
            InPosArg(
                ArgType.Scalar,
                name="alpha",
                deps=[0, 1],
                constraints=[
                    cp.Dtype.Ne(lambda deps: ScalarDtype.bool),
                    cp.Dtype.In(
                        lambda deps: fn.st_le(
                            fn.dt_to_st(
                                fn.promote_type_with_scalar(deps[0].dtype, deps[1])
                            )
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="scalar_tensor.default",  # (Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Scalar,
                name="s",
                deps=[1],
                constraints=[
                    cp.Value.Ge(lambda deps, dtype: fn.dtype_lower_bound(deps[0])),
                    cp.Value.Le(lambda deps, dtype: fn.dtype_upper_bound(deps[0])),
                ],
            ),
            InKwArg(ArgType.ScalarTypeOpt, name="dtype"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="scatter.src",  # (Tensor self, int dim, Tensor index, Tensor src) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=DimDefault,
            ),
            InPosArg(
                ArgType.Tensor,
                name="index",
                deps=[0, 1],
                constraints=[
                    cp.Dtype.Eq(lambda deps: torch.long),
                    cp.Rank.Eq(
                        lambda deps: deps[0].dim() if deps[0].dim() >= 2 else None
                    ),
                    cp.Rank.In(
                        lambda deps: [0, 1] if deps[0].dim() in [0, 1] else None
                    ),
                    cp.Size.Le(
                        lambda deps, r, d: (
                            fn.safe_size(deps[0], d)
                            if d != fn.normalize(deps[1], deps[0].dim())
                            else None
                        )
                    ),
                    cp.Value.Ge(lambda deps, dtype, struct: 0),
                    cp.Value.Le(
                        lambda deps, dtype, struct: (
                            0
                            if deps[0].dim() == 0
                            else max(0, fn.safe_size(deps[0], deps[1]) - 1)
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="src",
                deps=[0, 2],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                    cp.Rank.Eq(
                        lambda deps: deps[1].dim() if deps[1].numel() != 0 else None
                    ),
                    cp.Size.Ge(
                        lambda deps, r, d: (
                            fn.safe_size(deps[1], d) if deps[1].numel() != 0 else None
                        )
                    ),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="scatter.value",  # (Tensor self, int dim, Tensor index, Scalar value) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=[
                    cp.Value.In(lambda deps: fn.dim_non_zero_size(deps[0])),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="index",
                deps=[0, 1],
                # TODO(mcandales) Handle index.numel() == 0 case
                constraints=[
                    cp.Dtype.Eq(lambda deps: torch.long),
                    cp.Rank.Eq(
                        lambda deps: deps[0].dim() if deps[0].dim() >= 2 else None
                    ),
                    cp.Rank.In(
                        lambda deps: [0, 1] if deps[0].dim() in [0, 1] else None
                    ),
                    cp.Size.Le(
                        lambda deps, r, d: (
                            fn.safe_size(deps[0], d)
                            if d != fn.normalize(deps[1], deps[0].dim())
                            else None
                        )
                    ),
                    cp.Value.Ge(lambda deps, dtype, struct: 0),
                    cp.Value.Le(
                        lambda deps, dtype, struct: (
                            0
                            if deps[0].dim() == 0
                            else max(0, fn.safe_size(deps[0], deps[1]) - 1)
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.Scalar,
                name="value",
                deps=[0],
                constraints=[
                    cp.Value.NotIn(
                        lambda deps, dtype: (
                            [float("-inf"), float("inf")]
                            if deps[0].dtype not in dt._floating
                            else None
                        )
                    ),
                    cp.Value.Ge(
                        lambda deps, dtype: fn.dtype_lower_bound(deps[0].dtype)
                    ),
                    cp.Value.Le(
                        lambda deps, dtype: fn.dtype_upper_bound(deps[0].dtype)
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            ),
        ],
    ),
    Spec(  # TODO(mcandales): Calibrate.
        op="scatter_add.default",  # (Tensor self, int dim, Tensor index, Tensor src) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=[
                    cp.Value.In(lambda deps: fn.dim_non_zero_size(deps[0])),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="index",
                deps=[0, 1],
                # TODO(mcandales) Handle index.numel() == 0 case
                constraints=[
                    cp.Dtype.Eq(lambda deps: torch.long),
                    cp.Rank.Eq(
                        lambda deps: deps[0].dim() if deps[0].dim() >= 2 else None
                    ),
                    cp.Rank.In(
                        lambda deps: [0, 1] if deps[0].dim() in [0, 1] else None
                    ),
                    # cp.Rank.Eq(lambda deps: deps[0].dim()),
                    # cp.Size.Le(lambda deps, r, d: fn.scatter_add_index_size_max(
                    #     deps[0], deps[1], deps[2], d
                    # )),
                    cp.Size.Le(
                        lambda deps, r, d: (
                            fn.safe_size(deps[0], d)
                            if d != fn.normalize(deps[1], deps[0].dim())
                            else None
                        )
                    ),
                    cp.Value.Ge(lambda deps, dtype, struct: 0),
                    cp.Value.Le(
                        lambda deps, dtype, struct: (
                            0
                            if deps[0].dim() == 0
                            else max(0, fn.safe_size(deps[0], deps[1]) - 1)
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="src",
                deps=[0, 2],
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                    cp.Rank.Eq(
                        lambda deps: deps[1].dim() if deps[1].numel() != 0 else None
                    ),
                    cp.Size.Ge(
                        lambda deps, r, d: (
                            fn.safe_size(deps[1], d) if deps[1].numel() != 0 else None
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="select_copy.int",  # (Tensor self, int dim, SymInt index) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Rank.Ge(lambda deps: 1),
                ],
            ),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=[
                    cp.Value.In(lambda deps: fn.dim_non_zero_size(deps[0])),
                ],
            ),
            InPosArg(
                ArgType.Index,
                name="index",
                deps=[0, 1],
                constraints=IndexDefault,
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="select_scatter.default",  # (Tensor self, Tensor src, int dim, SymInt index) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Rank.Ge(lambda deps: 1),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="src",
                deps=[0, 2, 3],
                constraints=[
                    cp.Rank.Eq(
                        lambda deps: torch.ops.aten.select(
                            deps[0], deps[1], deps[2]
                        ).dim()
                    ),
                    cp.Size.Eq(
                        lambda deps, r, d: fn.safe_size(
                            torch.ops.aten.select(deps[0], deps[1], deps[2]), d
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=[
                    cp.Value.In(lambda deps: fn.dim_non_zero_size(deps[0])),
                ],
            ),
            InPosArg(
                ArgType.Index,
                name="index",
                deps=[0, 2],
                constraints=IndexDefault,
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="sigmoid.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="sign.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="sin.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="sinh.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="slice_copy.Tensor",  # (Tensor self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Rank.Ge(lambda deps: 1),
                ],
            ),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=DimDefault,
            ),
            InPosArg(
                ArgType.IndexOpt,
                name="start",
                deps=[0, 1],
                constraints=[
                    cp.Value.Be(
                        lambda deps: list(
                            range(
                                -fn.safe_size(deps[0], deps[1]) - 2,
                                fn.safe_size(deps[0], deps[1]) + 2,
                            )
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.IndexOpt,
                name="end",
                deps=[0, 1],
                constraints=[
                    cp.Value.Be(
                        lambda deps: list(
                            range(
                                -fn.safe_size(deps[0], deps[1]) - 2,
                                fn.safe_size(deps[0], deps[1]) + 2,
                            )
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.Length,
                name="step",
                deps=[0, 1],
                constraints=[
                    cp.Value.Ge(lambda deps: 1),
                    cp.Value.Be(
                        lambda deps: list(range(1, fn.safe_size(deps[0], deps[1]) + 2))
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="slice_scatter.default",  # (Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Rank.Ge(lambda deps: 1),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="src",
                deps=[0, 2, 3, 4, 5],
                constraints=[
                    cp.Rank.Eq(
                        lambda deps: torch.ops.aten.slice_copy.Tensor(
                            deps[0], deps[1], deps[2], deps[3], deps[4]
                        ).dim()
                    ),
                    cp.Size.Eq(
                        lambda deps, r, d: fn.safe_size(
                            torch.ops.aten.slice_copy.Tensor(
                                deps[0], deps[1], deps[2], deps[3], deps[4]
                            ),
                            d,
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=DimDefault,
            ),
            InPosArg(
                ArgType.IndexOpt,
                name="start",
                deps=[0, 2],
                constraints=[
                    cp.Value.Be(
                        lambda deps: list(
                            range(
                                -fn.safe_size(deps[0], deps[1]) - 2,
                                fn.safe_size(deps[0], deps[1]) + 2,
                            )
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.IndexOpt,
                name="end",
                deps=[0, 2],
                constraints=[
                    cp.Value.Be(
                        lambda deps: list(
                            range(
                                -fn.safe_size(deps[0], deps[1]) - 2,
                                fn.safe_size(deps[0], deps[1]) + 2,
                            )
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.Length,
                name="step",
                deps=[0, 2],
                constraints=[
                    cp.Value.Ge(lambda deps: 1),
                    cp.Value.Be(
                        lambda deps: list(range(1, fn.safe_size(deps[0], deps[1]) + 2))
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="split_copy.Tensor",  # (Tensor self, SymInt split_size, int dim=0) -> Tensor[]
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Rank.Ge(lambda deps: 1),
                ],
            ),
            InPosArg(
                ArgType.Length,
                name="split_size",
                deps=[0, 2],
                constraints=[
                    cp.Value.Ge(
                        lambda deps: 0 if fn.safe_size(deps[0], deps[1]) == 0 else 1
                    ),
                ],
            ),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=DimDefault,
            ),
        ],
        outspec=[
            OutArg(ArgType.TensorList),
        ],
    ),
    Spec(
        op="split_with_sizes_copy.default",  # (Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Rank.Ge(lambda deps: 1),
                ],
            ),
            InPosArg(
                ArgType.LengthList,
                name="split_sizes",
                deps=[0, 2],
                constraints=[
                    # cp.Value.Ge(lambda deps, length, ix: 0),
                    # cp.Value.Le(lambda deps, length, ix: fn.safe_size(deps[0], deps[1])),
                    cp.Length.Ge(lambda deps: 1),
                    cp.Value.Gen(
                        lambda deps, length: (
                            fn.valid_split_sizes(
                                fn.safe_size(deps[0], deps[1]), length
                            ),
                            fn.invalid_split_sizes(
                                fn.safe_size(deps[0], deps[1]), length
                            ),
                        )
                    ),
                ],
            ),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=DimDefault,
            ),
        ],
        outspec=[
            OutArg(ArgType.TensorList),
        ],
    ),
    Spec(
        op="sqrt.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="squeeze_copy.dim",  # (Tensor self, int dim) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=DimDefault,
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="squeeze_copy.dims",  # (Tensor self, int[] dim) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.DimList,
                name="dim",
                deps=[0],
                constraints=DimListDefault,
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="sub.Tensor",  # (Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0],
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
            InKwArg(
                ArgType.Scalar,
                name="alpha",
                deps=[0, 1],
                constraints=[
                    cp.Dtype.Ne(lambda deps: ScalarDtype.bool),
                    cp.Dtype.In(
                        lambda deps: fn.st_le(
                            fn.dt_to_st(
                                torch.promote_types(deps[0].dtype, deps[1].dtype)
                            )
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="sub.Scalar",  # (Tensor self, Scalar other, Scalar alpha=1) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                ],
            ),
            InPosArg(
                ArgType.Scalar,
                name="other",
                constraints=[
                    cp.Dtype.Ne(lambda deps: ScalarDtype.bool),
                ],
            ),
            InPosArg(
                ArgType.Scalar,
                name="alpha",
                deps=[0, 1],
                constraints=[
                    cp.Dtype.Ne(lambda deps: ScalarDtype.bool),
                    cp.Dtype.In(
                        lambda deps: fn.st_le(
                            fn.dt_to_st(
                                fn.promote_type_with_scalar(deps[0].dtype, deps[1])
                            )
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="sum.dim_IntList",  # (Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.DimListOpt,
                name="dim",
                deps=[0],
                constraints=DimListDefault,
            ),
            InPosArg(ArgType.Bool, name="keepdim"),
            InKwArg(ArgType.ScalarTypeOpt, name="dtype"),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="t_copy.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Rank.Le(lambda deps: 2),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="tan.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="tanh.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="topk.default",  # (Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[cp.Dtype.Ne(lambda deps: torch.bool)],
            ),
            InPosArg(
                ArgType.Length,
                name="k",
                deps=[0, 2],
                constraints=[
                    cp.Value.Ge(lambda deps: 0),
                    cp.Value.Le(lambda deps: fn.safe_size(deps[0], deps[1])),
                ],
            ),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=DimDefault,
            ),
            InPosArg(ArgType.Bool, name="largest"),
            InPosArg(ArgType.Bool, name="sorted"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                name="values",
                constraints=[cp.Dtype.Eq(lambda deps: deps[0].dtype)],
            ),
            OutArg(
                ArgType.Tensor,
                name="indices",
                constraints=[cp.Dtype.Eq(lambda deps: torch.long)],
            ),
        ],
    ),
    Spec(
        op="transpose_copy.int",  # (Tensor self, int dim0, int dim1) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Dim,
                name="dim0",
                deps=[0],
                constraints=DimDefault,
            ),
            InPosArg(
                ArgType.Dim,
                name="dim1",
                deps=[0],
                constraints=DimDefault,
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="tril.default",  # (Tensor self, int diagonal=0) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Rank.Ge(lambda deps: 2),
                ],
            ),
            InPosArg(
                ArgType.Int,
                name="diagonal",
                deps=[0],
                constraints=[
                    cp.Value.Be(
                        lambda deps: list(
                            range(
                                -fn.safe_size(deps[0], 0) - 1,
                                fn.safe_size(deps[0], 0) + 1,
                            )
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="trunc.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.Ne(lambda deps: torch.bool),
                ],
            ),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.Eq(lambda deps: deps[0].dtype),
                ],
            )
        ],
    ),
    Spec(
        op="unbind_copy.int",  # (Tensor self, int dim=0) -> Tensor[]
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Rank.Ge(lambda deps: 1),
                ],
            ),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=DimDefault,
            ),
        ],
        outspec=[
            OutArg(ArgType.TensorList),
        ],
    ),
    Spec(
        op="unfold_copy.default",  # (Tensor self, int dimension, int size, int step) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Dim,
                name="dimension",
                deps=[0],
                constraints=DimDefault,
            ),
            InPosArg(
                ArgType.Length,
                name="size",
                deps=[0, 1],
                constraints=[
                    cp.Value.Ge(lambda deps: 0),
                    cp.Value.Le(lambda deps: fn.safe_size(deps[0], deps[1])),
                ],
            ),
            InPosArg(
                ArgType.Length,
                name="step",
                constraints=[
                    cp.Value.Ge(lambda deps: 1),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="unsqueeze_copy.default",  # (Tensor self, int dim) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Dim,
                name="dim",
                deps=[0],
                constraints=[
                    cp.Value.Ge(lambda deps: -deps[0].dim() - 1),
                    cp.Value.Le(lambda deps: deps[0].dim()),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="upsample_bilinear2d.vec",  # (Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="input",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                    cp.Rank.Eq(lambda deps: 4),
                ],
            ),
            InPosArg(
                ArgType.ShapeOpt,
                name="output_size",
                constraints=[
                    cp.Length.Eq(lambda deps: 2),
                    cp.Value.Ge(lambda deps, length, ix: 1),
                ],
            ),
            InPosArg(ArgType.Bool, name="align_corners"),
            InPosArg(
                ArgType.FloatListOpt,
                name="scale_factors",
                deps=[0, 1],
                constraints=[
                    cp.Optional.Eq(lambda deps: deps[1] is not None),
                    cp.Length.Eq(lambda deps: 2),
                    cp.Value.Ge(
                        lambda deps, length, ix: 1
                        / float(fn.safe_size(deps[0], 2 + ix))
                    ),
                    cp.Value.Le(
                        lambda deps, length, ix: 10.0
                    ),  # restrict to avoid storage overflow
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="upsample_nearest2d.vec",  # (Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="input",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                    cp.Rank.Eq(lambda deps: 4),
                ],
            ),
            InPosArg(
                ArgType.ShapeOpt,
                name="output_size",
                constraints=[
                    cp.Length.Eq(lambda deps: 2),
                    cp.Value.Ge(lambda deps, length, ix: 1),
                ],
            ),
            InPosArg(
                ArgType.FloatListOpt,
                name="scale_factors",
                deps=[0, 1],
                constraints=[
                    cp.Optional.Eq(lambda deps: deps[1] is not None),
                    cp.Length.Eq(lambda deps: 2),
                    cp.Value.Ge(
                        lambda deps, length, ix: 1
                        / float(fn.safe_size(deps[0], 2 + ix))
                    ),
                    cp.Value.Le(
                        lambda deps, length, ix: 10.0
                    ),  # restrict to avoid storage overflow
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(
        op="var.correction",  # (Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                ],
            ),
            InPosArg(
                ArgType.DimListOpt,
                name="dim",
                deps=[0],
                constraints=DimListDefault,
            ),
            InKwArg(ArgType.ScalarOpt, name="correction"),
            InKwArg(ArgType.Bool, name="keepdim"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                ],
            ),
        ],
    ),
    Spec(
        op="var.dim",  # (Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                ],
            ),
            InPosArg(
                ArgType.DimListOpt,
                name="dim",
                deps=[0],
                constraints=DimListDefault,
            ),
            InPosArg(ArgType.Bool, name="unbiased"),
            InPosArg(ArgType.Bool, name="keepdim"),
        ],
        outspec=[
            OutArg(
                ArgType.Tensor,
                constraints=[
                    cp.Dtype.In(lambda deps: dt._floating),
                ],
            ),
        ],
    ),
    Spec(
        op="view_as_real_copy.default",  # (Tensor self) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="self",
                constraints=[
                    cp.Dtype.In(lambda deps: dt._complex),
                ],
            ),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
    Spec(  # TODO(mcandales) Implement numel constraint prefix
        op="view_copy.default",  # (Tensor self, SymInt[] size) -> Tensor
        inspec=[
            InPosArg(ArgType.Tensor, name="self"),
            InPosArg(
                ArgType.Shape,
                name="size",
                deps=[0],
                constraints=[
                    cp.Length.Ge(lambda deps: 1 if deps[0].numel() != 1 else 0),
                    cp.Value.Gen(
                        lambda deps, length: (
                            fn.valid_view_copy_size(deps[0], length),
                            fn.invalid_view_copy_size(deps[0], length),
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="where.self",  # (Tensor condition, Tensor self, Tensor other) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Tensor,
                name="condition",
                constraints=[
                    cp.Dtype.In(lambda deps: [torch.bool, torch.uint8]),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="self",
                deps=[0],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(deps[0].shape, r, d)
                    ),
                ],
            ),
            InPosArg(
                ArgType.Tensor,
                name="other",
                deps=[0, 1],
                constraints=[
                    cp.Size.In(
                        lambda deps, r, d: fn.broadcast_with(
                            fn.broadcasted_shape(deps[0].shape, deps[1].shape), r, d
                        )
                    ),
                ],
            ),
        ],
        outspec=[
            OutArg(ArgType.Tensor),
        ],
    ),
    Spec(
        op="zeros.default",  # (SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
        inspec=[
            InPosArg(
                ArgType.Shape,
                name="size",
                constraints=ShapeDefault,
            ),
            InKwArg(ArgType.ScalarTypeOpt, "dtype"),
        ],
        outspec=[OutArg(ArgType.Tensor)],
    ),
]


SpecDictDB = {s.op: s for s in SpecDB}
