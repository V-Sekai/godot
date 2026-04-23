module {
  func.func @compare_greater(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i1> {
    %0 = stablehlo.compare GT, %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %0 : tensor<i1>
  }
}