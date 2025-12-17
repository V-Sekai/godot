module {
  func.func @calculate_area(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}