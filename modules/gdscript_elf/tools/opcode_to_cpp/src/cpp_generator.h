#pragma once

#include "stablehlo_parser.h"
#include <string>

class CPPGenerator {
public:
    // Generate C++ code from StableHLO function
    static std::string generate_cpp(const StableHLOFunction &func);
    
    // Generate function signature
    static std::string generate_function_signature(const StableHLOFunction &func);
    
    // Generate function body
    static std::string generate_function_body(const StableHLOFunction &func);
    
    // Generate operation as C++ code
    static std::string generate_operation(const StableHLOOperation &op, int stack_size);

private:
    // Map StableHLO operation to C++ code
    static std::string map_stablehlo_to_cpp(const StableHLOOperation &op, int stack_size);
    
    // Generate stack variable name
    static std::string stack_var(int index);
    
    // Generate constant value
    static std::string generate_constant_value(const std::string &const_id, const std::map<std::string, std::string> &constants);
};

