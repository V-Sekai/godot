# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


class ArgType(str, Enum):
    Tensor = "Tensor"
    TensorOpt = "Tensor?"

    TensorList = "Tensor[]"
    TensorOptList = "Tensor?[]"

    Scalar = "Scalar"
    ScalarOpt = "Scalar?"

    ScalarType = "ScalarType"
    ScalarTypeOpt = "ScalarType?"

    Dim = "Dim"
    DimOpt = "Dim?"
    DimList = "Dim[]"
    DimListOpt = "Dim[]?"

    Shape = "Shape"
    ShapeOpt = "Shape?"
    Index = "Index"
    IndexOpt = "Index?"
    Length = "Length"
    LengthOpt = "Length?"
    LengthList = "Length[]"
    LengthListOpt = "Length[]?"

    Bool = "Bool"
    Int = "Integer"
    IntOpt = "Integer?"
    Float = "Float"
    FloatOpt = "Float?"
    FloatList = "Float[]"
    FloatListOpt = "Float[]?"
    String = "String"
    StringOpt = "String?"
    MemoryFormat = "MemoryFormat"

    def is_tensor(self) -> bool:
        return self in [ArgType.Tensor, ArgType.TensorOpt]

    def is_tensor_list(self) -> bool:
        return self in [ArgType.TensorList, ArgType.TensorOptList]

    def is_scalar(self) -> bool:
        return self in [ArgType.Scalar, ArgType.ScalarOpt]

    def is_scalar_type(self) -> bool:
        return self in [ArgType.ScalarType, ArgType.ScalarTypeOpt]

    def is_dim(self) -> bool:
        return self in [ArgType.Dim, ArgType.DimOpt]

    def is_dim_list(self) -> bool:
        return self in [ArgType.DimList, ArgType.DimListOpt]

    def is_shape(self) -> bool:
        return self in [ArgType.Shape, ArgType.ShapeOpt]

    def is_index(self) -> bool:
        return self in [ArgType.Index, ArgType.IndexOpt]

    def is_length(self) -> bool:
        return self in [ArgType.Length, ArgType.LengthOpt]

    def is_length_list(self) -> bool:
        return self in [ArgType.LengthList, ArgType.LengthListOpt]

    def is_bool(self) -> bool:
        return self in [ArgType.Bool]

    def is_int(self) -> bool:
        return self in [ArgType.Int, ArgType.IntOpt]

    def is_float(self) -> bool:
        return self in [ArgType.Float, ArgType.FloatOpt]

    def is_string(self) -> bool:
        return self in [ArgType.String, ArgType.StringOpt]

    def is_memory_format(self) -> bool:
        return self in [ArgType.MemoryFormat]

    def is_optional(self) -> bool:
        return self in [
            ArgType.TensorOpt,
            ArgType.ScalarOpt,
            ArgType.ScalarTypeOpt,
            ArgType.ShapeOpt,
            ArgType.DimOpt,
            ArgType.DimListOpt,
            ArgType.FloatOpt,
            ArgType.FloatListOpt,
            ArgType.IndexOpt,
            ArgType.IntOpt,
            ArgType.LengthOpt,
            ArgType.LengthListOpt,
            ArgType.StringOpt,
        ]

    def is_list(self) -> bool:
        return self in [
            ArgType.TensorList,
            ArgType.TensorOptList,
            ArgType.DimList,
            ArgType.DimListOpt,
            ArgType.LengthList,
            ArgType.LengthListOpt,
            ArgType.Shape,
            ArgType.ShapeOpt,
            ArgType.FloatList,
            ArgType.FloatListOpt,
        ]

    def has_integer_value(self) -> bool:
        return self in [
            ArgType.Dim,
            ArgType.DimOpt,
            ArgType.DimList,
            ArgType.DimListOpt,
            ArgType.Shape,
            ArgType.ShapeOpt,
            ArgType.Index,
            ArgType.IndexOpt,
            ArgType.Length,
            ArgType.LengthOpt,
            ArgType.LengthList,
            ArgType.LengthListOpt,
            ArgType.Int,
            ArgType.IntOpt,
        ]

    def has_dtype(self) -> bool:
        return (
            self.is_tensor()
            or self.is_tensor_list()
            or self.is_scalar()
            or self.is_scalar_type()
        )
