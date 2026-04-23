# CONDITIONAL LOGIC

# Simple if-else: if x > 10 then 100 else 0
module {
  func.func @conditional_simple(%arg0: tensor<i32>) -> tensor<i32> {
    %c10 = stablehlo.constant dense<10> : tensor<i32>
    %c100 = stablehlo.constant dense<100> : tensor<i32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare GT, %arg0, %c10 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1 = stablehlo.select %0, %c100, %c0 : tensor<i1>, tensor<i32>
    return %1 : tensor<i32>
  }
}

# If without else: if x > 5 then 1 else 0
module {
  func.func @conditional_if_only(%arg0: tensor<i32>) -> tensor<i32> {
    %c5 = stablehlo.constant dense<5> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare GT, %arg0, %c5 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1 = stablehlo.select %0, %c1, %c0 : tensor<i1>, tensor<i32>
    return %1 : tensor<i32>
  }
}

# Nested conditionals: if x > 10 then (if x > 20 then 200 else 100) else 0
module {
  func.func @conditional_nested(%arg0: tensor<i32>) -> tensor<i32> {
    %c10 = stablehlo.constant dense<10> : tensor<i32>
    %c20 = stablehlo.constant dense<20> : tensor<i32>
    %c200 = stablehlo.constant dense<200> : tensor<i32>
    %c100 = stablehlo.constant dense<100> : tensor<i32>
    %c0 = stablehlo.constant dense<0> : tensor<i32>

    %cond1 = stablehlo.compare GT, %arg0, %c10 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %cond2 = stablehlo.compare GT, %arg0, %c20 : (tensor<i32>, tensor<i32>) -> tensor<i1>

    %inner_select = stablehlo.select %cond2, %c200, %c100 : tensor<i1>, tensor<i32>
    %outer_select = stablehlo.select %cond1, %inner_select, %c0 : tensor<i1>, tensor<i32>

    return %outer_select : tensor<i32>
  }
}