# RETURN STATEMENTS

# Simple return with constant
module {
  func.func @return_int() -> tensor<i32> {
    %c42 = stablehlo.constant dense<42> : tensor<i32>
    return %c42 : tensor<i32>
  }
}

# Return with addition
module {
  func.func @return_add(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
}

# Return with variable
module {
  func.func @return_variable() -> tensor<i32> {
    %c10 = stablehlo.constant dense<10> : tensor<i32>
    return %c10 : tensor<i32>
  }
}