# ARITHMETIC OPERATIONS

# Addition
module {
  func.func @arithmetic_add(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
}

# Subtraction
module {
  func.func @arithmetic_subtract(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
}

# Multiplication
module {
  func.func @arithmetic_multiply(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
}

# Division (float)
module {
  func.func @arithmetic_divide(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.divide %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}

# Complex arithmetic: a + b * c - (a / 2)
module {
  func.func @complex_arithmetic(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %0 = stablehlo.multiply %arg1, %arg2 : tensor<i32>
    %1 = stablehlo.add %arg0, %0 : tensor<i32>
    %2 = stablehlo.divide %arg0, %c2 : tensor<i32>
    %3 = stablehlo.subtract %1, %2 : tensor<i32>
    return %3 : tensor<i32>
  }
}