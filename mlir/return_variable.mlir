module {
  func.func @return_variable() -> tensor<i32> {
    %c10 = stablehlo.constant dense<10> : tensor<i32>
    return %c10 : tensor<i32>
  }
}