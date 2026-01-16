module {
  func.func @conditional_simple(%arg0: tensor<i32>) -> tensor<i32> {
    %c10 = stablehlo.constant dense<10> : tensor<i32>
    %c100 = stablehlo.constant dense<100> : tensor<i32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare GT, %arg0, %c10 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1 = stablehlo.select %0, %c100, %c0 : tensor<i1>, tensor<i32>
    return %1 : tensor<i32>
  }
}