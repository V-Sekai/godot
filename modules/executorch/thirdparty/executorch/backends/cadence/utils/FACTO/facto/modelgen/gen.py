# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import traceback
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch.nn as nn

from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.inputgen.specs.model import InArg, Spec
from facto.inputgen.utils.config import TensorConfig


def is_forward_arg(spec: Spec, arg: InArg) -> bool:
    """
    Check if an argument is a forward input.

    Args:
        spec: The operation specification containing argument type information
        arg: The argument to check

    Returns:
        True if the argument is a forward input, False otherwise
    """
    return (
        arg.type.is_tensor() or arg.type.is_tensor_list()
    ) and not arg.type.is_optional()


def separate_forward_and_model_inputs(
    spec: Spec, args: List[Any], kwargs: Dict[str, Any]
) -> Tuple[List[Any], Dict[str, Any], List[Any], Dict[str, Any]]:
    """
    Separate forward inputs from model parameters using FACTO's ArgType system.

    Args:
        spec: The operation specification containing argument type information
        args: All positional arguments
        kwargs: All keyword arguments

    Returns:
        Tuple of (forward_args, forward_kwargs, model_args, model_kwargs)
    """
    forward_args = []
    model_args = []

    forward_kwargs = {}
    model_kwargs = {}

    for i, inarg in enumerate(spec.inspec):
        is_forward_input = is_forward_arg(spec, inarg)
        if inarg.kw:
            if is_forward_input:
                forward_kwargs[inarg.name] = kwargs[inarg.name]
            else:
                model_kwargs[inarg.name] = kwargs[inarg.name]
        else:
            if is_forward_input:
                forward_args.append(args[i])
            else:
                model_args.append(args[i])

    return forward_args, forward_kwargs, model_args, model_kwargs


def combine_forward_and_model_inputs(
    spec: Spec,
    forward_args: Tuple[Any],
    forward_kwargs: Dict[str, Any],
    model_args: Tuple[Any],
    model_kwargs: Dict[str, Any],
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Combine forward inputs with model parameters using FACTO's ArgType system.

    Args:
        spec: The operation specification containing argument type information
        args: All positional arguments
        kwargs: All keyword arguments
        model_args: All model parameters
        model_kwargs: All model keyword parameters

    Returns:
        Tuple of (args, kwargs)
    """
    combined_args = []
    combined_kwargs = {}

    forward_args_ix = 0
    model_args_ix = 0

    # Iterate over the input specification
    for ix, inarg in enumerate(spec.inspec):
        is_forward_input = is_forward_arg(spec, inarg)
        if inarg.kw:
            # If the argument is a keyword argument, check if it's a tensor or tensor list
            if is_forward_input:
                combined_kwargs[inarg.name] = forward_kwargs[inarg.name]
            else:
                combined_kwargs[inarg.name] = model_kwargs[inarg.name]
        else:
            # If the argument is a positional argument, check if it's a tensor or tensor list
            if is_forward_input:
                combined_args.append(forward_args[forward_args_ix])
                forward_args_ix += 1
            else:
                combined_args.append(model_args[model_args_ix])
                model_args_ix += 1

    return combined_args, combined_kwargs


class OpModel(nn.Module):
    """
    A PyTorch model that wraps a torch aten operation.

    This class creates a simple model that applies a given torch operation
    to its inputs in the forward pass.
    """

    def __init__(
        self, op: Any, spec: Spec, op_name: str = "", *model_args, **model_kwargs
    ):
        """
        Initialize the OpModel.

        Args:
            op: The torch aten operation to wrap
            op_name: Optional name for the operation (for debugging/logging)
            *model_args: Positional model parameters
            **model_kwargs: Keyword model parameters
        """
        super().__init__()
        self.op = op
        self.op_name = op_name or str(op)
        self.spec = spec
        self.model_args = model_args
        self.model_kwargs = model_kwargs

    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass that applies the wrapped operation to the inputs.

        Args:
            *args: Positional arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation

        Returns:
            The result of applying the operation to the inputs
        """
        op_args, op_kwargs = combine_forward_and_model_inputs(
            self.spec, args, kwargs, self.model_args, self.model_kwargs
        )
        return self.op(*op_args, **op_kwargs)

    def __repr__(self) -> str:
        return f"OpModel(op={self.op_name})"


class OpModelGenerator:
    """
    Generator that creates OpModel instances with appropriate inputs for testing.

    This class takes a torch operation and its specification, then uses
    ArgumentTupleGenerator to create OpModel instances along with valid
    inputs for the forward function. It automatically separates tensor inputs
    from non-tensor parameters for ExecuTorch compatibility using FACTO's ArgType system.
    """

    def __init__(self, op: Any, spec: Spec, config: Optional[TensorConfig] = None):
        """
        Initialize the OpModelGenerator.

        Args:
            op: The torch aten operation to wrap in models
            spec: The specification for the operation's arguments
            config: Optional tensor configuration for input generation
        """
        self.op = op
        self.spec = spec
        self.config = config
        self.arg_generator = ArgumentTupleGenerator(spec, config)

    def gen(
        self,
        *,
        valid: bool = True,
        verbose: bool = False,
        max_count: Optional[int] = None,
    ) -> Generator[Tuple[OpModel, List[Any], Dict[str, Any]], None, None]:
        """
        Generate OpModel instances with corresponding inputs.

        Args:
            valid: Whether to generate valid inputs (default: True)
            max_count: Maximum number of models to generate (default: None for unlimited)

        Yields:
            Tuple containing:
            - OpModel instance wrapping the operation
            - List of positional arguments for forward()
            - Dict of keyword arguments for forward()
        """
        count = 0
        for args, kwargs, _ in self.arg_generator.gen(
            valid=valid, out=False, verbose=verbose
        ):
            if max_count is not None and count >= max_count:
                break

            # Separate tensor inputs from non-tensor parameters
            forward_args, forward_kwargs, model_args, model_kwargs = (
                separate_forward_and_model_inputs(self.spec, args, kwargs)
            )

            # Create model instance
            model = OpModel(
                self.op, self.spec, self.spec.op, *model_args, **model_kwargs
            )

            yield model, forward_args, forward_kwargs
            count += 1

    def test_model_with_inputs(
        self, model: OpModel, args: List[Any], kwargs: Dict[str, Any]
    ) -> Tuple[bool, Optional[Any], Optional[Exception]]:
        """
        Test a model with given inputs and return the result.

        Args:
            model: The OpModel to test
            args: Positional arguments for the model
            kwargs: Keyword arguments for the model

        Returns:
            Tuple containing:
            - Boolean indicating success/failure
            - The output if successful, None if failed
            - The exception if failed, None if successful
        """
        try:
            output = model(*args, **kwargs)
            return True, output, None
        except Exception as e:
            traceback.print_exc()
            return False, None, e

    def __repr__(self) -> str:
        return f"OpModelGenerator(op={self.spec.op})"
