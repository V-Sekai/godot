# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from facto.inputgen.argtuple.gen import ArgumentTupleGenerator
from facto.inputgen.argument.engine import MetaArg
from facto.inputgen.argument.gen import ArgumentGenerator
from facto.inputgen.argument.type import ArgType
from facto.inputgen.specs.model import ConstraintProducer as cp, InPosArg, Spec
from facto.inputgen.utils.config import TensorConfig
from facto.inputgen.variable.solve import SolvableVariable


class TestTensorConfigIntegration(unittest.TestCase):
    """Integration tests for TensorConfig"""

    def setUp(self):
        """Set up common test fixtures."""
        # Create a basic variable space for tensor values
        self.variable = SolvableVariable(float)
        self.variable.Ge(-10)
        self.variable.Le(10)

        # Create a basic MetaArg for testing transpose
        self.meta_arg = MetaArg(
            argtype=ArgType.Tensor,
            dtype=torch.float32,
            structure=(3, 4),  # 2D tensor for testing transpose
            value=self.variable.space,
        )

    def test_transposed_tensor_generation(self):
        """Test that ALLOW_TRANSPOSED condition affects tensor generation."""
        # Not allow transposition
        config = TensorConfig(transposed=False).set_probability(1.0)

        generator = ArgumentGenerator(self.meta_arg, config=config)
        tensor = generator.gen()

        self.assertIsNotNone(tensor)
        self.assertEqual(tuple(tensor.size()), (3, 4))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.dim_order(), (0, 1))

        # Force transposition to be applied
        config = TensorConfig(transposed=True).set_probability(1.0)

        generator = ArgumentGenerator(self.meta_arg, config=config)
        tensor = generator.gen()

        self.assertIsNotNone(tensor)
        self.assertEqual(tuple(tensor.size()), (3, 4))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.dim_order(), (1, 0))

    def test_permuted_tensor_generation(self):
        """Test that ALLOW_PERMUTED condition affects tensor generation."""
        # Create a 3D tensor MetaArg for better permutation testing
        meta_arg_3d = MetaArg(
            argtype=ArgType.Tensor,
            dtype=torch.float32,
            structure=(2, 3, 4),
            value=self.variable.space,
        )

        # Not allow permutation
        config = TensorConfig(permuted=False)
        generator = ArgumentGenerator(meta_arg_3d, config=config)
        tensor = generator.gen()

        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (2, 3, 4))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.dim_order(), (0, 1, 2))

        # Force permutation to be applied
        config = TensorConfig(permuted=True).set_probability(1.0)
        generator = ArgumentGenerator(meta_arg_3d, config=config)
        tensor = generator.gen()

        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (2, 3, 4))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertNotEqual(tensor.dim_order(), (0, 1, 2))

    def test_strided_tensor_generation(self):
        """Test that ALLOW_STRIDED condition affects tensor generation."""
        # Test with striding allowed and high probability
        config = TensorConfig(strided=True).set_probability(1.0)

        # Not allow strided
        config = TensorConfig(strided=False)
        generator = ArgumentGenerator(self.meta_arg, config=config)
        tensor = generator.gen()

        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (3, 4))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.dim_order(), (0, 1))
        self.assertEqual(tensor.stride(), (4, 1))
        self.assertTrue(tensor.is_contiguous())

        # Force strided tensor to be generated
        config = TensorConfig(strided=True).set_probability(1.0)
        generator = ArgumentGenerator(self.meta_arg, config=config)
        tensor = generator.gen()

        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (3, 4))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.dim_order(), (0, 1))
        self.assertNotEqual(tensor.stride(), (4, 1))
        self.assertFalse(tensor.is_contiguous())

    def test_multiple_conditions_enabled(self):
        """Test tensor generation with multiple conditions enabled."""
        config = TensorConfig(
            transposed=True, permuted=True, strided=True
        ).set_probability(1.0)

        # Use 3D tensor for comprehensive testing
        meta_arg_3d = MetaArg(
            argtype=ArgType.Tensor,
            dtype=torch.float32,
            structure=(4, 5, 6),
            value=self.variable.space,
        )

        generator = ArgumentGenerator(meta_arg_3d, config=config)
        tensor = generator.gen()

        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (4, 5, 6))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertNotEqual(tensor.dim_order(), (0, 1, 2))
        self.assertFalse(tensor.is_contiguous())
        self.assertNotEqual(
            len(tensor.storage()) - tensor.storage_offset(), tensor.numel()
        )

    def test_no_conditions_enabled(self):
        """Test tensor generation with no special conditions enabled."""
        config = TensorConfig()  # All conditions False by default

        meta_arg = MetaArg(
            argtype=ArgType.Tensor,
            dtype=torch.float32,
            structure=(3, 4),
            value=self.variable.space,
        )

        generator = ArgumentGenerator(meta_arg, config=config)
        tensor = generator.gen()

        # Should generate normal contiguous tensor
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (3, 4))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(tensor.is_contiguous())

    def test_empty_tensor_not_allowed_in_tuple_generation(self):
        """Test that ALLOW_EMPTY=False prevents empty tensors in ArgumentTupleGenerator."""
        # Create a spec that could potentially generate empty tensors
        spec = Spec(
            op="test_empty_prevention",
            inspec=[
                InPosArg(
                    ArgType.Tensor,
                    name="input_tensor",
                    constraints=[
                        # Allow rank 0-3 which could include empty tensors
                        cp.Rank.Ge(lambda deps: 0),
                        cp.Rank.Le(lambda deps: 3),
                        # Allow sizes 0-5 which could include empty dimensions
                        cp.Size.Ge(lambda deps, r, d: 0),
                        cp.Size.Le(lambda deps, r, d: 5),
                    ],
                )
            ],
            outspec=[],
        )

        # Test with ALLOW_EMPTY=False (should prevent empty tensors)
        config = TensorConfig(empty=False)
        generator = ArgumentTupleGenerator(spec, config=config)

        # Generate multiple argument tuples and verify none have empty tensors
        generated_count = 0
        for posargs, _, _ in generator.gen(valid=True):
            generated_count += 1
            self.assertEqual(len(posargs), 1)
            tensor = posargs[0]

            # Verify tensor is not empty (all dimensions > 0)
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertNotEqual(tensor.numel(), 0)

            # Stop after checking a reasonable number of generated tuples
            if generated_count >= 10:
                break

        self.assertGreater(generated_count, 0, "No argument tuples were generated")

    def test_empty_tensor_allowed_in_tuple_generation(self):
        """Test that ALLOW_EMPTY=True allows empty tensors in ArgumentTupleGenerator."""
        # Create a spec that can generate empty tensors
        spec = Spec(
            op="test_empty_allowed",
            inspec=[
                InPosArg(
                    ArgType.Tensor,
                    name="input_tensor",
                    constraints=[
                        # Allow rank 1-2 for simpler testing
                        cp.Rank.Ge(lambda deps: 1),
                        cp.Rank.Le(lambda deps: 2),
                        # Explicitly allow size 0 to encourage empty tensor generation
                        cp.Size.Ge(lambda deps, r, d: 0),
                        cp.Size.Le(lambda deps, r, d: 3),
                    ],
                )
            ],
            outspec=[],
        )

        # Test with ALLOW_EMPTY=True (should allow empty tensors)
        config = TensorConfig(empty=True)
        generator = ArgumentTupleGenerator(spec, config=config)

        # Generate multiple argument tuples and check if any have empty tensors
        generated_count = 0
        found_empty_tensor = False

        for posargs, _, _ in generator.gen(valid=True):
            generated_count += 1
            self.assertEqual(len(posargs), 1)
            tensor = posargs[0]

            self.assertIsInstance(tensor, torch.Tensor)

            # Check if this tensor has any empty dimensions
            if any(dim_size == 0 for dim_size in tensor.shape):
                found_empty_tensor = True
                break

            # Stop after checking a reasonable number of generated tuples
            if generated_count >= 50:
                break

        self.assertGreater(generated_count, 0, "No argument tuples were generated")
        self.assertTrue(found_empty_tensor, "No empty tensors were generated")


if __name__ == "__main__":
    unittest.main()
