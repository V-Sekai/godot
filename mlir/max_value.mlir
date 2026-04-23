module {
  func.func @max_value(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
}