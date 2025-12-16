#include "cpp_generator.h"
#include <sstream>
#include <algorithm>

std::string CPPGenerator::stack_var(int index) {
    return "stack[" + std::to_string(index) + "]";
}

std::string CPPGenerator::generate_constant_value(const std::string &const_id, const std::map<std::string, std::string> &constants) {
    auto it = constants.find(const_id);
    if (it != constants.end()) {
        return it->second;
    }
    return "0.0"; // Default
}

std::string CPPGenerator::map_stablehlo_to_cpp(const StableHLOOperation &op, int stack_size) {
    std::ostringstream code;
    
    auto extract_index = [](const std::string &id) -> int {
        // Extract number from "v1", "c1", "arg0", etc.
        size_t start = 0;
        while (start < id.length() && !std::isdigit(id[start])) {
            start++;
        }
        if (start < id.length()) {
            return std::stoi(id.substr(start));
        }
        return 0;
    };
    
    if (op.op_name == "stablehlo.constant") {
        // Constant assignment
        if (!op.results.empty()) {
            int result_idx = extract_index(op.results[0]);
            std::string const_value = "0.0"; // Default, would be extracted from constants map
            code << "    " << stack_var(result_idx) << " = Variant(" << const_value << ");\n";
        }
    } else if (op.op_name == "stablehlo.add") {
        // Addition - use Variant operator+
        if (op.operands.size() >= 2 && !op.results.empty()) {
            int a_idx = extract_index(op.operands[0]);
            int b_idx = extract_index(op.operands[1]);
            int result_idx = extract_index(op.results[0]);
            code << "    " << stack_var(result_idx) << " = " << stack_var(a_idx) << " + " << stack_var(b_idx) << ";\n";
        }
    } else if (op.op_name == "stablehlo.subtract") {
        // Subtraction
        if (op.operands.size() >= 2 && !op.results.empty()) {
            int a_idx = extract_index(op.operands[0]);
            int b_idx = extract_index(op.operands[1]);
            int result_idx = extract_index(op.results[0]);
            code << "    " << stack_var(result_idx) << " = " << stack_var(a_idx) << " - " << stack_var(b_idx) << ";\n";
        }
    } else if (op.op_name == "stablehlo.multiply") {
        // Multiplication
        if (op.operands.size() >= 2 && !op.results.empty()) {
            int a_idx = extract_index(op.operands[0]);
            int b_idx = extract_index(op.operands[1]);
            int result_idx = extract_index(op.results[0]);
            code << "    " << stack_var(result_idx) << " = " << stack_var(a_idx) << " * " << stack_var(b_idx) << ";\n";
        }
    } else if (op.op_name == "stablehlo.divide") {
        // Division
        if (op.operands.size() >= 2 && !op.results.empty()) {
            int a_idx = extract_index(op.operands[0]);
            int b_idx = extract_index(op.operands[1]);
            int result_idx = extract_index(op.results[0]);
            code << "    " << stack_var(result_idx) << " = " << stack_var(a_idx) << " / " << stack_var(b_idx) << ";\n";
        }
    } else if (op.op_name == "stablehlo.if") {
        // Conditional
        if (!op.operands.empty()) {
            int cond_idx = extract_index(op.operands[0]);
            code << "    if (" << stack_var(cond_idx) << ".booleanize()) {\n";
            code << "        // true branch\n";
            code << "    } else {\n";
            code << "        // false branch\n";
            code << "    }\n";
        }
    } else if (op.op_name == "stablehlo.return") {
        // Return
        if (!op.operands.empty()) {
            int return_idx = extract_index(op.operands[0]);
            code << "    *result = " << stack_var(return_idx) << ";\n";
        } else {
            code << "    *result = Variant();\n";
        }
        code << "    return;\n";
    } else if (op.op_name == "stablehlo.call") {
        // Function call - use syscall
        code << "    // Function call via syscall\n";
        code << "    ECALL_VCALL(instance, method_name, args, argcount, result);\n";
    } else if (op.op_name == "stablehlo.custom_call") {
        // Custom call (member access) - use syscall
        code << "    // Custom call (member access) via syscall\n";
        if (op.operands.size() >= 1) {
            int obj_idx = extract_index(op.operands[0]);
            code << "    ECALL_VCALL(" << stack_var(obj_idx) << ".get_object(), method_name, args, argcount, result);\n";
        }
    } else if (op.op_name == "stablehlo.copy") {
        // Copy assignment
        if (op.operands.size() >= 1 && !op.results.empty()) {
            int src_idx = extract_index(op.operands[0]);
            int dst_idx = extract_index(op.results[0]);
            code << "    " << stack_var(dst_idx) << " = " << stack_var(src_idx) << ";\n";
        }
    }
    
    return code.str();
}

std::string CPPGenerator::generate_operation(const StableHLOOperation &op, int stack_size) {
    return map_stablehlo_to_cpp(op, stack_size);
}

std::string CPPGenerator::generate_function_signature(const StableHLOFunction &func) {
    std::ostringstream sig;
    // Simplified signature matching sandbox API - operator_funcs handled via syscalls
    sig << "void gdscript_" << func.name << "(void* instance, Variant* args, int argcount, "
        << "Variant* result, Variant* constants)";
    return sig.str();
}

std::string CPPGenerator::generate_function_body(const StableHLOFunction &func) {
    std::ostringstream body;
    
    // Calculate stack size (max value index + 1)
    int max_stack = func.arguments.size();
    for (const auto &op : func.operations) {
        for (const auto &result : op.results) {
            if (result[0] == 'v' || result[0] == 'c') {
                int idx = std::stoi(result.substr(1));
                if (idx >= max_stack) {
                    max_stack = idx + 1;
                }
            }
        }
    }
    
    // Stack declaration
    body << "{\n";
    body << "    Variant stack[" << max_stack << "];\n";
    body << "    int ip = 0;\n\n";
    
    // Initialize arguments
    for (size_t i = 0; i < func.arguments.size(); i++) {
        body << "    stack[" << i << "] = args[" << i << "];\n";
    }
    body << "\n";
    
    // Generate operations
    for (const auto &op : func.operations) {
        body << generate_operation(op, max_stack);
    }
    
    body << "}\n";
    
    return body.str();
}

std::string CPPGenerator::generate_cpp(const StableHLOFunction &func) {
    std::ostringstream code;
    
    // Includes
    code << "#include <stdint.h>\n";
    code << "#include <api.hpp>  // Sandbox API (includes Variant, GuestVariant, syscalls, etc.)\n";
    code << "\n";
    
    // Function
    code << generate_function_signature(func) << "\n";
    code << generate_function_body(func);
    
    return code.str();
}

