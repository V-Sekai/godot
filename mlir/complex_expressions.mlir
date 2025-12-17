# COMPLEX EXPRESSIONS

# Mixed operations: a * b + (a - b) / 2.0
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

# Boolean arithmetic with conditionals
module {
  func.func @boolean_arithmetic(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c2 = stablehlo.constant dense<2> : tensor<i32>

    %cond1 = stablehlo.compare GT, %arg0, %arg1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %cond2 = stablehlo.compare GT, %arg0, %c0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %both_true = stablehlo.and %cond1, %cond2 : tensor<i1>

    %x_doubled = stablehlo.multiply %arg0, %c2 : tensor<i32>
    %y_doubled = stablehlo.multiply %arg1, %c2 : tensor<i32>
    %result = stablehlo.select %both_true, %x_doubled, %y_doubled : tensor<i1>, tensor<i32>

    return %result : tensor<i32>
  }
}