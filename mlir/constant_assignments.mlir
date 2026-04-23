# CONSTANT ASSIGNMENTS

# Null assignment (represented as zero)
module {
  func.func @assign_null_concept() -> tensor<i32> {
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    return %c0 : tensor<i32>
  }
}

# Boolean assignments
module {
  func.func @assign_booleans() -> tensor<i1> {
    %ctrue = stablehlo.constant dense<1> : tensor<i1>
    %cfalse = stablehlo.constant dense<0> : tensor<i1>
    %0 = stablehlo.and %ctrue, %cfalse : tensor<i1>
    %1 = stablehlo.not %0 : tensor<i1>
    return %1 : tensor<i1>
  }
}