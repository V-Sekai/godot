#pragma once

#include <string>

class RISCVCompiler {
public:
    // Compile C++ source to RISC-V ELF64
    // Returns path to ELF64 binary, or empty string on error
    static std::string compile_to_elf64(const std::string &cpp_source_path, const std::string &output_elf_path);
    
    // Check if RISC-V cross-compiler is available
    static bool is_compiler_available();
    
    // Get detected compiler path
    static std::string get_compiler_path();

private:
    // Detect RISC-V cross-compiler
    static std::string detect_compiler();
};

