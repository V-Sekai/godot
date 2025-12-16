#include "stablehlo_parser.h"
#include "cpp_generator.h"
#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_stablehlo_file> <output_cpp_file>" << std::endl;
        std::cerr << "  input_stablehlo_file: Path to StableHLO text or bytecode file (.mlir or .mlir.bc)" << std::endl;
        std::cerr << "  output_cpp_file: Path to output C++ file" << std::endl;
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    // Parse StableHLO file
    StableHLOFunction func = StableHLOParser::parse_from_file(input_file);
    
    if (func.name.empty()) {
        std::cerr << "Error: Failed to parse StableHLO file: " << input_file << std::endl;
        return 1;
    }
    
    // Validate operations
    for (const auto &op : func.operations) {
        if (!StableHLOParser::is_supported_operation(op.op_name)) {
            std::cerr << "Warning: Unsupported operation: " << op.op_name << std::endl;
        }
    }
    
    // Generate C++ code
    std::string cpp_code = CPPGenerator::generate_cpp(func);
    
    // Write to output file
    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        std::cerr << "Error: Cannot open output file: " << output_file << std::endl;
        return 1;
    }
    
    out_file << cpp_code;
    out_file.close();
    
    std::cout << "Successfully generated C++ code: " << output_file << std::endl;
    return 0;
}

