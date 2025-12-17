# ASSIGNMENTS

# Variable assignment
module {
  func.func @assign_variable() -> tensor<i32> {
    %c5 = stablehlo.constant dense<5> : tensor<i32>
    %c10 = stablehlo.constant dense<10> : tensor<i32>
    %0 = stablehlo.add %c5, %c10 : tensor<i32>
    return %0 : tensor<i32>
  }
}

# Assignment with expression
module {
  func.func @assign_expression(%arg0: tensor<i32>) -> tensor<i32> {
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.multiply %arg0, %c2 : tensor<i32>
    %1 = stablehlo.add %0, %c1 : tensor<i32>
    return %1 : tensor<i32>
  }
}

# Multiple assignments
module {
  func.func @multiple_assignments() -> tensor<i32> {
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %c3 = stablehlo.constant dense<3> : tensor<i32>
    %0 = stablehlo.add %c1, %c2 : tensor<i32>
    %1 = stablehlo.add %0, %c3 : tensor<i32>
    return %1 : tensor<i32>
  }
}