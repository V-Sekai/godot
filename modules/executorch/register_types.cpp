#include "register_types.h"
#include <iostream>

void initialize_executorch_module() {
    // This is where we would register the class with Godot's ClassDB
    // ClassDB::register_class<ExecuTorchRuntime>();

    // For now, just print initialization message
    std::cout << "ExecuTorch module initialized" << std::endl;
}

void uninitialize_executorch_module() {
    std::cout << "ExecuTorch module uninitialized" << std::endl;
}
