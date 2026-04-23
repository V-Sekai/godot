module {
  func.func @assign_variable() -> tensor<i32> {
    %c5 = stablehlo.constant dense<5> : tensor<i32>
    %c10 = stablehlo.constant dense<10> : tensor<i32>
    %0 = stablehlo.add %c5, %c10 : tensor<i32>
    return %0 : tensor<i32>
  }
}