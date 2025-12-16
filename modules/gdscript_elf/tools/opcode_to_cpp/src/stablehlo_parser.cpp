#include "stablehlo_parser.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <iostream>

bool StableHLOParser::is_supported_operation(const std::string &op_name) {
    // Supported StableHLO operations mapped from GDScript opcodes
    return op_name == "stablehlo.add" ||
           op_name == "stablehlo.subtract" ||
           op_name == "stablehlo.multiply" ||
           op_name == "stablehlo.divide" ||
           op_name == "stablehlo.constant" ||
           op_name == "stablehlo.if" ||
           op_name == "stablehlo.return" ||
           op_name == "stablehlo.call" ||
           op_name == "stablehlo.custom_call" ||
           op_name == "stablehlo.copy";
}

std::string StableHLOParser::extract_value_id(const std::string &operand) {
    // Extract value ID from "%v1" or "v1" format
    std::regex pattern(R"(%?([cv]\d+))");
    std::smatch match;
    if (std::regex_search(operand, match, pattern)) {
        return match[1].str();
    }
    return operand;
}

StableHLOOperation StableHLOParser::parse_operation(const std::string &line) {
    StableHLOOperation op;
    
    // Remove leading whitespace and comments
    std::string clean_line = line;
    size_t comment_pos = clean_line.find("//");
    if (comment_pos != std::string::npos) {
        clean_line = clean_line.substr(0, comment_pos);
    }
    
    // Trim whitespace
    clean_line.erase(0, clean_line.find_first_not_of(" \t"));
    clean_line.erase(clean_line.find_last_not_of(" \t") + 1);
    
    if (clean_line.empty()) {
        return op;
    }
    
    // Parse result (if any): %v1 = ...
    std::regex result_pattern(R"(%([cv]\d+)\s*=\s*(.+))");
    std::smatch result_match;
    if (std::regex_search(clean_line, result_match, result_pattern)) {
        op.results.push_back(result_match[1].str());
        clean_line = result_match[2].str();
    }
    
    // Parse operation name
    std::regex op_pattern(R"((stablehlo\.\w+))");
    std::smatch op_match;
    if (std::regex_search(clean_line, op_match, op_pattern)) {
        op.op_name = op_match[1].str();
    }
    
    // Parse operands (values in parentheses or after operation)
    std::regex operand_pattern(R"(%([cv]\d+))");
    std::sregex_iterator iter(clean_line.begin(), clean_line.end(), operand_pattern);
    std::sregex_iterator end;
    for (; iter != end; ++iter) {
        op.operands.push_back(iter->str(1));
    }
    
    return op;
}

void StableHLOParser::parse_function_signature(const std::string &line, StableHLOFunction &func) {
    // Parse: func.func @function_name(%arg0: tensor<f32>, ...) -> tensor<f32>
    std::regex func_pattern(R"(func\.func\s+@(\w+)\s*\(([^)]*)\)\s*->\s*(\w+))");
    std::smatch match;
    if (std::regex_search(line, match, func_pattern)) {
        func.name = match[1].str();
        func.return_type = match[3].str();
        
        // Parse arguments
        std::string args_str = match[2].str();
        std::regex arg_pattern(R"(%(\w+):\s*(\w+))");
        std::sregex_iterator iter(args_str.begin(), args_str.end(), arg_pattern);
        std::sregex_iterator end;
        for (; iter != end; ++iter) {
            func.arguments.push_back(iter->str(1));
        }
    }
}

StableHLOFunction StableHLOParser::parse_from_string(const std::string &content) {
    StableHLOFunction func;
    std::istringstream stream(content);
    std::string line;
    bool in_function = false;
    int line_num = 0;
    
    while (std::getline(stream, line)) {
        line_num++;
        
        // Check for module start
        if (line.find("module {") != std::string::npos) {
            continue;
        }
        
        // Check for function definition
        if (line.find("func.func") != std::string::npos) {
            in_function = true;
            parse_function_signature(line, func);
            continue;
        }
        
        // Check for function end
        if (in_function && line.find("}") != std::string::npos && line.find("{") == std::string::npos) {
            break;
        }
        
        // Parse operations inside function
        if (in_function) {
            StableHLOOperation op = parse_operation(line);
            if (!op.op_name.empty()) {
                op.line_number = line_num;
                func.operations.push_back(op);
            }
            
            // Check for constants
            if (op.op_name == "stablehlo.constant") {
                // Extract constant value
                std::regex const_pattern(R"(dense<([^>]+)>)");
                std::smatch const_match;
                if (std::regex_search(line, const_match, const_pattern)) {
                    if (!op.results.empty()) {
                        func.constants[op.results[0]] = const_match[1].str();
                    }
                }
            }
        }
    }
    
    return func;
}

StableHLOFunction StableHLOParser::parse_from_file(const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << file_path << std::endl;
        return StableHLOFunction();
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();
    
    return parse_from_string(content);
}

