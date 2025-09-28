# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Generator, List, Optional, Tuple

import torch
from facto.inputgen.argtuple.engine import MetaArgTupleEngine
from facto.inputgen.argument.engine import MetaArg
from facto.inputgen.argument.gen import ArgumentGenerator
from facto.inputgen.specs.model import ConstraintProducer as cp, Spec
from facto.inputgen.utils.config import Condition, TensorConfig


def compare_tensors(x, y):
    x = x.to(torch.float)
    y = y.to(torch.float)
    c = torch.isclose(x, y, atol=1e-4, rtol=1e-2, equal_nan=True)
    if c.all():
        return True, 0, lambda t: t
    else:
        d = x[c == False] - y[c == False]
        x_d = x[c == False]
        y_d = y[c == False]
        xy = torch.logical_or(
            torch.logical_and(x_d > 1e30, y_d == float("inf")),
            torch.logical_and(x_d == float("inf"), y_d > 1e30),
        )
        if xy.all():
            return True, 0, lambda t: t
        else:
            ix = xy == False
            return False, d[ix].abs().max().item(), lambda t: t[c == False][ix]


class ArgumentTupleGenerator:
    def __init__(self, spec: Spec, config: Optional[TensorConfig] = None):
        self.spec = spec
        self.config = config
        self._modified_spec = self._apply_config_constraints(spec, config)

    def _apply_config_constraints(
        self, spec: Spec, config: Optional[TensorConfig]
    ) -> Spec:
        """Apply TensorConfig constraints to the spec by modifying argument constraints."""

        if config is None:
            return spec

        # Create a copy of the spec with modified constraints
        modified_inspec = []
        for arg in spec.inspec:
            modified_arg = self._apply_constraints_to_arg(arg, config)
            modified_inspec.append(modified_arg)

        modified_outspec = []
        for arg in spec.outspec:
            modified_arg = self._apply_constraints_to_arg(arg, config)
            modified_outspec.append(modified_arg)

        return Spec(spec.op, modified_inspec, modified_outspec)

    def _apply_constraints_to_arg(self, arg, config: TensorConfig):
        """Apply config constraints to a single argument."""
        # Create a copy of the argument with potentially modified constraints
        modified_arg = deepcopy(arg)

        # Add rank constraints for tensor arguments when zerodim tensors are not allowed
        if not config.is_allowed(Condition.ALLOW_ZERODIM):
            if arg.type.is_tensor():
                rank_constraint = cp.Rank.Ge(lambda deps: 1)
                modified_arg.constraints = modified_arg.constraints + [rank_constraint]
            elif arg.type.is_tensor_list():
                rank_constraint = cp.Rank.Ge(lambda deps, length, ix: 1)
                modified_arg.constraints = modified_arg.constraints + [rank_constraint]

        # Add size constraints for tensor arguments when empty tensors are not allowed
        if not config.is_allowed(Condition.ALLOW_EMPTY):
            if arg.type.is_tensor() or arg.type.is_tensor_list():
                size_constraint = cp.Size.Ge(lambda deps, r, d: 1)
                modified_arg.constraints = modified_arg.constraints + [size_constraint]

        # Add dtype constraints for tensor arguments when dtypes are disallowed
        if config.disallow_dtypes:
            if arg.type.is_tensor():
                dtype_constraint = cp.Dtype.NotIn(lambda deps: config.disallow_dtypes)
                modified_arg.constraints = modified_arg.constraints + [dtype_constraint]
            elif arg.type.is_tensor_list():
                dtype_constraint = cp.Dtype.NotIn(
                    lambda deps, length, ix: config.disallow_dtypes
                )
                modified_arg.constraints = modified_arg.constraints + [dtype_constraint]
            elif arg.type.is_scalar_type():
                dtype_constraint = cp.Value.NotIn(lambda deps: config.disallow_dtypes)
                modified_arg.constraints = modified_arg.constraints + [dtype_constraint]

        return modified_arg

    def gen_tuple(
        self, meta_tuple: Tuple[MetaArg], *, out: bool = False
    ) -> Tuple[List[Any], OrderedDict[str, Any], OrderedDict[str, Any]]:
        posargs = []
        inkwargs = OrderedDict()
        outargs = OrderedDict()
        for ix, arg in enumerate(self.spec.inspec):
            m = meta_tuple[ix]
            val = ArgumentGenerator(m, config=self.config).gen()
            if arg.kw:
                inkwargs[arg.name] = val
            else:
                posargs.append(val)
        if out:
            for ix, arg in enumerate(self.spec.outspec):
                m = meta_tuple[ix + len(self.spec.inspec)]
                val = ArgumentGenerator(m, config=self.config).gen()
                outargs[arg.name] = val
        return posargs, inkwargs, outargs

    def gen(
        self, *, valid: bool = True, out: bool = False, verbose: bool = False
    ) -> Generator[
        Tuple[List[Any], OrderedDict[str, Any], OrderedDict[str, Any]], Any, Any
    ]:
        engine = MetaArgTupleEngine(self._modified_spec, out=out)
        for meta_tuple in engine.gen(valid=valid):
            if verbose:
                print(f"Generated meta_tuple: {[str(x) for x in meta_tuple]}")
            yield self.gen_tuple(meta_tuple, out=out)

    def gen_errors(
        self,
        op,
        *,
        valid: bool = True,
        out: bool = False,
        verbose: bool = False,
        check_correctness: bool = False,
    ) -> Generator[
        Tuple[List[Any], OrderedDict[str, Any], OrderedDict[str, Any]], Any, Any
    ]:
        """
        Generate input tuples and yield only those that don't behave as expected.

        This function takes the same signature as gen() but with an additional
        op parameter. It filters inputs based on whether they behave as expected:
        - When valid=True: yields inputs that should be valid but DO error
        - When valid=False: yields inputs that should be invalid but DON'T error

        Args:
            op: The operation/function to test the inputs against
            valid: Whether to generate valid or invalid inputs (same as gen())
            out: Whether to include output arguments (same as gen())

        Yields:
            Tuples of (posargs, inkwargs, outargs) that don't behave as expected
        """

        engine = MetaArgTupleEngine(self._modified_spec, out=out)
        for meta_tuple in engine.gen(valid=valid):
            posargs, inkwargs, outargs = self.gen_tuple(meta_tuple, out=out)

            try:
                # Try to execute the operation with the generated inputs
                if out:
                    # If there are output arguments, include them in the call
                    ret = op(*posargs, **inkwargs, **outargs)
                else:
                    # Otherwise, just call with positional and keyword arguments
                    ret = op(*posargs, **inkwargs)

                # If execution succeeds:
                if valid:
                    # When valid=True, we expect success, so this is NOT a bug
                    if (
                        check_correctness
                        and self.config is not None
                        and self.config.device != "cpu"
                    ):
                        # If correctness=True, and device != cpu we also check if the output is correct
                        # by comparing it to the cpu output
                        cpu_posargs = []
                        cpu_inkwargs = OrderedDict()
                        cpu_outargs = OrderedDict()
                        for arg in posargs:
                            new = arg
                            if isinstance(arg, torch.Tensor):
                                new = arg.to("cpu")
                            cpu_posargs.append(new)
                        for k, v in inkwargs.items():
                            new = v
                            if isinstance(v, torch.Tensor):
                                new = v.to("cpu")
                            cpu_inkwargs[k] = new
                        for k, v in outargs.items():
                            new = v
                            if isinstance(v, torch.Tensor):
                                new = v.to("cpu")
                            cpu_outargs[k] = new

                        try:
                            cpu_ret = op(*cpu_posargs, **cpu_inkwargs, **cpu_outargs)
                        except Exception:
                            continue

                        if isinstance(ret, torch.Tensor) and isinstance(
                            cpu_ret, torch.Tensor
                        ):
                            allclose, max_diff, ix_fun = compare_tensors(
                                cpu_ret, ret.to("cpu")
                            )
                            if not allclose:
                                if verbose:
                                    print(f"Output mismatch: {max_diff}")
                                    print(
                                        op.__name__, str([str(x) for x in meta_tuple])
                                    )
                                    # for cpu_arg in cpu_posargs:
                                    #     if isinstance(cpu_arg, torch.Tensor) and cpu_arg.numel() < 10:
                                    #         print(cpu_arg)
                                    #     elif isinstance(cpu_arg, torch.Tensor) and cpu_arg.shape == cpu_ret.shape:
                                    #         print(ix_fun(cpu_arg))
                                    # if ret.numel() < 10:
                                    #     print(ret)
                                    #     print(cpu_ret)
                                yield posargs, inkwargs, outargs
                else:
                    # When valid=False, we expect failure, so success IS a bug
                    if verbose:
                        print(f"Unexpected success:")
                        print(op.__name__, str([str(x) for x in meta_tuple]))
                    yield posargs, inkwargs, outargs

            except Exception as e:
                # If execution fails:
                if valid:
                    # When valid=True, we expect success, so failure IS a bug
                    if verbose:
                        print(op.__name__, str([str(x) for x in meta_tuple]))
                        print(f"Exception occurred: {e}")
                    yield posargs, inkwargs, outargs
                else:
                    # When valid=False, we expect failure, so this is NOT a bug
                    continue
