# GDScript to StableHLO Examples

This document shows examples of GDScript functions and their corresponding StableHLO (Stable High Level Operations) representations, based on the actual conversion implementation in Godot's `GDScriptToStableHLO` class.

## Hello World Example

### Hello World with Byte Buffer and Syscalls
**GDScript:**
```gdscript
func hello_world() -> int:
    print("Hello, World!")
    return 0
```

**StableHLO:**
```mlir
module {
  func.func @hello_world() -> tensor<i32> {
    // Create byte buffer tensor with "Hello, World!\n" string
    %message = stablehlo.constant dense<[72, 101, 108, 108, 111, 44, 32, 87, 111, 114, 108, 100, 33, 10]> : tensor<14xi8>
    
    // Call godot_syscall_print from the sandbox module to print the byte buffer
    %result = stablehlo.custom_call @godot_syscall_print(%message) : (tensor<14xi8>) -> tensor<i32>
    
    // Return success code (0)
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    return %c0 : tensor<i32>
  }
}
```

**Notes:**
- The string is converted to a byte buffer tensor (`tensor<14xi8>`) containing ASCII values
- `stablehlo.custom_call` is used to invoke `godot_syscall_print` from the sandbox module
- This allows I/O operations in StableHLO by using syscalls, which is necessary since StableHLO itself doesn't support string operations directly

## Basic Arithmetic Operations

### 1. Simple Addition
**GDScript:**
```gdscript
func add_numbers(a: int, b: int) -> int:
    return a + b
```

**StableHLO:**
```mlir
module {
  func.func @add_numbers(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
}
```

### 2. Multiplication
**GDScript:**
```gdscript
func multiply(x: float, y: float) -> float:
    return x * y
```

**StableHLO:**
```mlir
module {
  func.func @multiply(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
```

### 3. Subtraction
**GDScript:**
```gdscript
func subtract(a: int, b: int) -> int:
    return a - b
```

**StableHLO:**
```mlir
module {
  func.func @subtract(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
}
```

## Constants and Variables

### 4. Constant Return
**GDScript:**
```gdscript
func constant_return() -> int:
    return 42
```

**StableHLO:**
```mlir
module {
  func.func @constant_return() -> tensor<i32> {
    %c42 = arith.constant dense<42> : tensor<i32>
    return %c42 : tensor<i32>
  }
}
```

### 5. Variable Assignment
**GDScript:**
```gdscript
func calculate_area(width: float, height: float) -> float:
    var area = width * height
    return area
```

**StableHLO:**
```mlir
module {
  func.func @calculate_area(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
```

## Comparison Operations

### 6. Maximum Value (Conditional)
**GDScript:**
```gdscript
func max_value(x: int, y: int) -> int:
    if x > y:
        return x
    else:
        return y
```

**StableHLO:**
```mlir
module {
  func.func @max_value(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
}
```

### 7. Boolean Logic
**GDScript:**
```gdscript
func is_even(n: int) -> bool:
    return n % 2 == 0
```

**StableHLO:**
```mlir
module {
  func.func @is_even(%arg0: tensor<i32>) -> tensor<i1> {
    %c2 = arith.constant dense<2> : tensor<i32>
    %0 = arith.remsi %arg0, %c2 : tensor<i32>
    %c0 = arith.constant dense<0> : tensor<i32>
    %1 = arith.cmpi eq, %0, %c0 : tensor<i1>
    return %1 : tensor<i1>
  }
}
```

## Complex Operations

### 8. Conditional with Select
**GDScript:**
```gdscript
func conditional_max(a: int, b: int, use_a: bool) -> int:
    if use_a:
        return a
    else:
        return b
```

**StableHLO (conceptual):**
```mlir
module {
  func.func @conditional_max(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i1>) -> tensor<i32> {
    %0 = stablehlo.select %arg2, %arg0, %arg1 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
}
```

### 9. Mathematical Expression
**GDScript:**
```gdscript
func quadratic_formula(a: float, b: float, c: float, x: float) -> float:
    return a * x * x + b * x + c
```

**StableHLO:**
```mlir
module {
  func.func @quadratic_formula(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg3 : tensor<f32>
    %1 = stablehlo.multiply %0, %arg3 : tensor<f32>
    %2 = stablehlo.multiply %arg1, %arg3 : tensor<f32>
    %3 = stablehlo.add %1, %2 : tensor<f32>
    %4 = stablehlo.add %3, %arg2 : tensor<f32>
    return %4 : tensor<f32>
  }
}
```

## Notes

1. **Type System**: GDScript types are mapped to StableHLO tensor types:
   - `int` → `tensor<i32>`
   - `float` → `tensor<f32>`
   - `bool` → `tensor<i1>`

2. **Function Arguments**: All function arguments are mapped to `%arg0`, `%arg1`, etc.

3. **Constants**: Constants in GDScript become `arith.constant` operations in StableHLO.

4. **Operations**: Basic arithmetic operations map directly to StableHLO operations like `stablehlo.add`, `stablehlo.multiply`, etc.

5. **Control Flow**: Complex control flow (loops, conditionals) may be simplified or may not be fully supported depending on the implementation maturity.

6. **Current Limitations**: The current implementation focuses on basic arithmetic and comparison operations. Complex features like loops, object-oriented constructs, and advanced GDScript features may not be fully supported.

## Running the Conversion

To generate StableHLO from GDScript functions, you can use the `GDScriptToStableHLO` class in Godot:

```cpp
String stablehlo_text = GDScriptToStableHLO::convert_function_to_stablehlo_text(function_pointer);
```

The conversion is part of Godot's experimental GDScript-to-StableHLO compilation pipeline, which aims to enable GDScript functions to run on ML accelerators and other StableHLO-compatible runtimes.