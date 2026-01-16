module {
  func.func @is_even(%arg0: tensor<i32>) -> tensor<i1> {
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %0 = stablehlo.remainder %arg0, %c2 : tensor<i32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.compare EQ, %0, %c0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %1 : tensor<i1>
  }
}