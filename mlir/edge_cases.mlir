# EDGE CASES AND LIMITATIONS

# Large numbers
module {
  func.func @large_numbers() -> tensor<i32> {
    %c1000000 = stablehlo.constant dense<1000000> : tensor<i32>
    %c2000000 = stablehlo.constant dense<2000000> : tensor<i32>
    %0 = stablehlo.add %c1000000, %c2000000 : tensor<i32>
    return %0 : tensor<i32>
  }
}

# Floating point precision
module {
  func.func @float_precision() -> tensor<f32> {
    %c1 = stablehlo.constant dense<1.0> : tensor<f32>
    %c3 = stablehlo.constant dense<3.0> : tensor<f32>
    %c0 = stablehlo.divide %c1, %c3 : tensor<f32>
    %1 = stablehlo.multiply %c0, %c3 : tensor<f32>
    return %1 : tensor<f32>
  }
}

# Zero operations
module {
  func.func @zero_operations() -> tensor<i32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c5 = stablehlo.constant dense<5> : tensor<i32>
    %0 = stablehlo.multiply %c0, %c5 : tensor<i32>
    %1 = stablehlo.add %c0, %0 : tensor<i32>
    %2 = stablehlo.subtract %c1, %c0 : tensor<i32>
    return %2 : tensor<i32>
  }
}

# Identity operations
module {
  func.func @identity_operations(%arg0: tensor<i32>) -> tensor<i32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c5 = stablehlo.constant dense<5> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.multiply %c0, %c5 : tensor<i32>
    %1 = stablehlo.divide %c0, %c1 : tensor<i32>
    %2 = stablehlo.add %arg0, %1 : tensor<i32>
    return %2 : tensor<i32>
  }
}