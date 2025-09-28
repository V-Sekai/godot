# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

_int = [torch.uint8, torch.int8, torch.short, torch.int, torch.long]
_int_and_bool = [torch.bool] + _int
_floating = [torch.float16, torch.bfloat16, torch.float, torch.double]
_real = _int + _floating
_real_and_bool = [torch.bool] + _int + _floating
_complex = [torch.chalf, torch.cfloat, torch.cdouble]
_quant = [torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4]
_all = _real_and_bool + _complex + _quant


def can_cast_from(t):
    return [x for x in _all if torch.can_cast(t, x)]


def can_cast_to(t):
    return [x for x in _all if torch.can_cast(x, t)]
