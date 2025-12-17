module {
  func.func @return_add(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }
}