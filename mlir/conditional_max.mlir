module {
  func.func @conditional_max(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i1>) -> tensor<i32> {
    %0 = stablehlo.select %arg2, %arg0, %arg1 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
}