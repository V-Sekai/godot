module {
  func.func @quadratic_formula(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.multiply %arg0, %arg3 : tensor<f32>
    %1 = stablehlo.multiply %0, %arg3 : tensor<f32>
    %2 = stablehlo.multiply %arg1, %arg3 : tensor<f32>
    %3 = stablehlo.add %1, %2 : tensor<f32>
    %4 = stablehlo.add %3, %arg2 : tensor<f32>
    return %4 : tensor<f32>
  }
}