module {
  func.func @mixed_operations(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %c2 = stablehlo.constant dense<2.0> : tensor<f32>
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<f32>
    %2 = stablehlo.divide %1, %c2 : tensor<f32>
    %3 = stablehlo.add %0, %2 : tensor<f32>
    return %3 : tensor<f32>
  }
}