module {
  func.func @constant_return() -> tensor<i32> {
    %c42 = stablehlo.constant dense<42> : tensor<i32>
    return %c42 : tensor<i32>
  }
}