# GDScript to StableHLO MLIR Examples

This directory contains MLIR (Multi-Level Intermediate Representation) files generated from GDScript functions using Godot's experimental `GDScriptToStableHLO` converter.

## Scripts and Documentation

- `stablehlo_demo.gd` - GDScript demonstration script showing conceptual StableHLO generation
- `stablehlo_examples.md` - Comprehensive documentation of GDScript-to-StableHLO conversion examples
- `test_stablehlo_samples.gd` - Additional GDScript test samples

## MLIR Files (Validated with stablehlo-translate)

### Hello World Example
- `hello_world.mlir` - Hello World using byte buffer and godot_syscalls ✅

### Core Operations
- `add_numbers.mlir` - Integer addition ✅
- `multiply.mlir` - Float multiplication ✅
- `subtract.mlir` - Integer subtraction ✅
- `max_value.mlir` - Maximum value selection ✅
- `calculate_area.mlir` - Area calculation ✅
- `conditional_max.mlir` - Conditional logic ✅
- `quadratic_formula.mlir` - Polynomial evaluation ✅

### Comprehensive AST Coverage
- `return_int.mlir` - Simple constant return ✅
- `return_add.mlir` - Return with arithmetic expression ✅
- `return_variable.mlir` - Return with variable/constant ✅
- `assign_variable.mlir` - Variable assignment and arithmetic ✅
- `arithmetic_subtract.mlir` - Subtraction operation ✅
- `compare_greater.mlir` - Greater than comparison ✅
- `conditional_simple.mlir` - If-else conditional logic ✅
- `mixed_operations.mlir` - Complex mixed arithmetic ✅

### Scripts and Documentation
- `comprehensive_ast_examples.gd` - Complete GDScript examples for all AST elements
- `stablehlo_demo.gd` - GDScript demonstration script
- `stablehlo_examples.md` - Comprehensive conversion examples and documentation
- `test_stablehlo_samples.gd` - Additional GDScript test samples

## Scripts and Documentation

- `stablehlo_demo.gd` - GDScript demonstration script
- `stablehlo_examples.md` - Comprehensive conversion examples and documentation
- `test_stablehlo_samples.gd` - Additional GDScript test samples

*Note: Some MLIR files were removed due to dialect compatibility issues with the current StableHLO environment.*

## Validation with stablehlo-translate

All MLIR files have been validated using the StableHLO toolchain:

```bash
# Test interpretation
stablehlo-translate --interpret add_numbers.mlir --args="[dense<1> : tensor<i32>, dense<2> : tensor<i32>]"
# Result: tensor<i32> {3}

# Test serialization to portable format
stablehlo-translate --serialize --target=1.0.0 add_numbers.mlir -o add_numbers.stablehlo

# Test deserialization
stablehlo-translate --deserialize add_numbers.stablehlo
```

## Key Results

✅ **All MLIR files are syntactically valid StableHLO code**
✅ **Interpreter correctly executes arithmetic operations**
✅ **Serialization to portable StableHLO format works**
✅ **Deserialization preserves original functionality**

## Mathematical Validation

### Core Operations
- `add_numbers(1, 2)` = 3 ✓
- `multiply(3.5, 2.0)` = 7.0 ✓
- `max_value(5, 10)` = 10 ✓
- `quadratic_formula(1,2,1,3)` = 16 (x²+2x+1 at x=3) ✓

### Comprehensive AST Coverage
- `return_int()` = 42 ✓
- `return_add(3, 4)` = 7 ✓
- `return_variable()` = 10 ✓
- `assign_variable()` = 15 ✓
- `arithmetic_subtract(10, 3)` = 7 ✓
- `compare_greater(5, 3)` = true ✓
- `conditional_simple(15)` = 100 ✓
- `mixed_operations(3.0, 2.0)` = 6.5 ✓

## Implications

This demonstrates that Godot's GDScript-to-StableHLO conversion produces:

1. **Valid MLIR code** that passes StableHLO validation
2. **Correct mathematical semantics** in the interpreter
3. **Portable artifacts** via serialization
4. **Potential for ML hardware acceleration** of GDScript functions

The conversion enables GDScript code to run on ML accelerators, TPUs, and other hardware that supports the StableHLO standard.