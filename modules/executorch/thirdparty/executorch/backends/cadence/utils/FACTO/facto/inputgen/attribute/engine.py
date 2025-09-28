# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import List, Optional

from facto.inputgen.argument.type import ArgType
from facto.inputgen.attribute.model import Attribute
from facto.inputgen.attribute.solve import AttributeSolver
from facto.inputgen.specs.model import Constraint
from facto.inputgen.variable.gen import VariableGenerator
from facto.inputgen.variable.type import (
    ScalarDtype,
    sort_values_of_type,
    SUPPORTED_TENSOR_DTYPES,
)


class AttributeEngine(AttributeSolver):
    def __init__(
        self,
        attribute: Attribute,
        constraints: List[Constraint],
        valid: bool,
        argtype: Optional[ArgType] = None,
        scalar_dtype: Optional[ScalarDtype] = None,
    ):
        super().__init__(attribute, argtype, scalar_dtype)
        self.constraints = constraints
        self.valid = valid

    def gen(self, focus: Attribute, *args):
        if self.attribute == Attribute.OPTIONAL:
            num = 2
        elif self.attribute == focus:
            if self.attribute == Attribute.DTYPE:
                num = len(SUPPORTED_TENSOR_DTYPES)
            else:
                num = 6
        else:
            num = 1
        gen_vals = set()
        for variable in self.solve(self.constraints, focus, self.valid, *args):
            vals = []
            if variable.vtype in [bool, int, float]:
                limits = self.attribute.get_custom_limits(self.argtype)
                if limits is not None:
                    v_copy = copy.deepcopy(variable)
                    v_copy.Ge(limits[0])
                    v_copy.Le(limits[1])
                    vals = VariableGenerator(v_copy.space).gen(num)
            if len(vals) == 0:
                vals = VariableGenerator(variable.space).gen(num)
            gen_vals.update(vals)
        return sort_values_of_type(self.vtype, gen_vals)
