module {
  func.func @test_constant() -> tensor<i32> {
    %c42 = stablehlo.constant dense<42> : tensor<i32>
    return %c42 : tensor<i32>
  }
}