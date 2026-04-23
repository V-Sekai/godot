# COMPARISON OPERATIONS

# Greater than
module {
  func.func @compare_greater(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
    %0 = stablehlo.compare GT, %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

# Less than
module {
  func.func @compare_less(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
    %0 = stablehlo.compare LT, %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

# Greater equal
module {
  func.func @compare_greater_equal(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
    %0 = stablehlo.compare GE, %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

# Less equal
module {
  func.func @compare_less_equal(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
    %0 = stablehlo.compare LE, %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

# Equal
module {
  func.func @compare_equal(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
    %0 = stablehlo.compare EQ, %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}

# Not equal
module {
  func.func @compare_not_equal(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
    %0 = stablehlo.compare NE, %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}