#pragma once

#include <string>
#include <vector>
#include <map>

// Represents a StableHLO operation
struct StableHLOOperation {
    std::string op_name;
    std::vector<std::string> operands;
    std::vector<std::string> results;
    std::map<std::string, std::string> attributes;
    int line_number = 0;
};

// Represents a parsed StableHLO function
struct StableHLOFunction {
    std::string name;
    std::vector<std::string> arguments;
    std::string return_type;
    std::vector<StableHLOOperation> operations;
    std::map<std::string, std::string> constants;
};

class StableHLOParser {
public:
    // Parse StableHLO text format (MLIR text) from file
    static StableHLOFunction parse_from_file(const std::string &file_path);
    
    // Parse StableHLO text format from string
    static StableHLOFunction parse_from_string(const std::string &content);
    
    // Check if operation is supported
    static bool is_supported_operation(const std::string &op_name);

private:
    // Parse a single operation line
    static StableHLOOperation parse_operation(const std::string &line);
    
    // Extract value ID from operand string (e.g., "%v1" -> "v1")
    static std::string extract_value_id(const std::string &operand);
    
    // Parse function signature
    static void parse_function_signature(const std::string &line, StableHLOFunction &func);
};

