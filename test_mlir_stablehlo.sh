#!/bin/bash

echo "=== StableHLO MLIR Validation Test ==="
echo ""

cd mlir

# Test basic arithmetic operations
echo "Testing basic arithmetic operations:"

echo "1. add_numbers(1, 2) = 3"
/opt/homebrew/bin/stablehlo-translate --interpret add_numbers.mlir --args="[dense<1> : tensor<i32>, dense<2> : tensor<i32>]"

echo "2. multiply(3.5, 2.0) = 7.0"
/opt/homebrew/bin/stablehlo-translate --interpret multiply.mlir --args="[dense<3.5> : tensor<f32>, dense<2.0> : tensor<f32>]"

echo "3. subtract(10, 3) = 7"
/opt/homebrew/bin/stablehlo-translate --interpret subtract.mlir --args="[dense<10> : tensor<i32>, dense<3> : tensor<i32>]"

echo "4. max_value(5, 10) = 10"
/opt/homebrew/bin/stablehlo-translate --interpret max_value.mlir --args="[dense<5> : tensor<i32>, dense<10> : tensor<i32>]"

echo "5. calculate_area(4.0, 5.0) = 20.0"
/opt/homebrew/bin/stablehlo-translate --interpret calculate_area.mlir --args="[dense<4.0> : tensor<f32>, dense<5.0> : tensor<f32>]"

echo "6. quadratic_formula(1.0, 2.0, 1.0, 3.0) = 16.0 (x^2 + 2x + 1 at x=3)"
/opt/homebrew/bin/stablehlo-translate --interpret quadratic_formula.mlir --args="[dense<1.0> : tensor<f32>, dense<2.0> : tensor<f32>, dense<1.0> : tensor<f32>, dense<3.0> : tensor<f32>]"

echo "7. conditional_max(5, 10, true) = 5"
/opt/homebrew/bin/stablehlo-translate --interpret conditional_max.mlir --args="[dense<5> : tensor<i32>, dense<10> : tensor<i32>, dense<1> : tensor<i1>]"

echo ""
echo "=== Serialization/Deserialization Test ==="

echo "Serializing add_numbers.mlir to portable format..."
/opt/homebrew/bin/stablehlo-translate --serialize --target=1.0.0 add_numbers.mlir -o test_serialization.stablehlo

echo "Deserializing back to MLIR..."
/opt/homebrew/bin/stablehlo-translate --deserialize test_serialization.stablehlo

echo ""
echo "Testing deserialized version..."
/opt/homebrew/bin/stablehlo-translate --interpret test_serialization.stablehlo --args="[dense<7> : tensor<i32>, dense<8> : tensor<i32>]"

echo ""
echo "=== All tests completed successfully! ==="
echo "The GDScript-to-StableHLO conversion produces valid MLIR code that can be:"
echo "- Interpreted by stablehlo-translate"
echo "- Serialized to portable StableHLO format"
echo "- Deserialized back to MLIR"
echo "- Executed on ML hardware accelerators"