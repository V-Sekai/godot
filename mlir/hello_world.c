// Generated C99 code from StableHLO
// Compile with: gcc -std=c99 hello_world.c -o hello_world

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

// Forward declaration for custom call
extern int32_t godot_syscall_print(int8_t* message, int64_t* message_shape);

// Function: hello_world
// Returns: tensor<i32>
int32_t hello_world() {
  // Allocate message
  int64_t message_size = 1 * 14;
  int8_t* message = (int8_t*)malloc(message_size * sizeof(int8_t));
  int64_t message_shape[] = {14};

  // Constant tensor: message
  // Initialize constant array with values: "Hello, World!\n"
  message[0] = 72;   // 'H'
  message[1] = 101;  // 'e'
  message[2] = 108;  // 'l'
  message[3] = 108;  // 'l'
  message[4] = 111;  // 'o'
  message[5] = 44;   // ','
  message[6] = 32;   // ' '
  message[7] = 87;   // 'W'
  message[8] = 111;  // 'o'
  message[9] = 114;  // 'r'
  message[10] = 108; // 'l'
  message[11] = 100; // 'd'
  message[12] = 33;  // '!'
  message[13] = 10;  // '\n'

  // Call godot_syscall_print from the sandbox module to print the byte buffer
  // The custom_call invokes the syscall which handles the byte buffer and prints it
  // result is tensor<i32> (scalar), so we call the function once
  int32_t result;
  {
    // For a syscall that prints a string, we pass the entire buffer
    // This assumes godot_syscall_print can handle the buffer directly
    // Since result is a scalar (rank 0), we just call the function directly
    result = godot_syscall_print(message, message_shape);
  }
  
  // Return success code (0)
  // Constant tensor: c0 (scalar tensor<i32>)
  int32_t c0 = 0;

  int32_t return_value = c0;
  
  // Cleanup
  free(message);
  
  return return_value;
}

