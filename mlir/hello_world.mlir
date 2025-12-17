module {
  // Hello World in StableHLO using byte buffer and godot_syscalls
  // This demonstrates how to print text from StableHLO by:
  // 1. Creating a tensor of bytes representing the string
  // 2. Using custom_call to invoke godot_syscall_print from the sandbox module
  func.func @hello_world() -> tensor<i32> {
    // Create byte buffer tensor with "Hello, World!\n" string
    // ASCII bytes: H=72, e=101, l=108, l=108, o=111, ,=44, space=32, W=87, o=111, r=114, l=108, d=100, !=33, \n=10
    %message = stablehlo.constant dense<[72, 101, 108, 108, 111, 44, 32, 87, 111, 114, 108, 100, 33, 10]> : tensor<14xi8>
    
    // Call godot_syscall_print from the sandbox module to print the byte buffer
    // The custom_call invokes the syscall which handles the byte buffer and prints it
    %result = stablehlo.custom_call @godot_syscall_print(%message) : (tensor<14xi8>) -> tensor<i32>
    
    // Return success code (0)
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    return %c0 : tensor<i32>
  }
}
