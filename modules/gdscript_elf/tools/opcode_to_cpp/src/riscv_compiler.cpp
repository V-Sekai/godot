#include "riscv_compiler.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

std::string RISCVCompiler::detect_compiler() {
    // Try common RISC-V cross-compiler names
    const char* candidates[] = {
        "riscv64-unknown-elf-g++",
        "riscv64-linux-gnu-g++",
        "riscv64-linux-gnu-g++-14",
        "riscv64-linux-gnu-g++-12",
        "zig c++",  // Zig can cross-compile to RISC-V
        nullptr
    };
    
    for (int i = 0; candidates[i] != nullptr; i++) {
        std::string cmd = std::string(candidates[i]) + " --version";
        int result = std::system((cmd + " > /dev/null 2>&1").c_str());
        if (result == 0) {
            return candidates[i];
        }
    }
    
    return "";
}

bool RISCVCompiler::is_compiler_available() {
    return !detect_compiler().empty();
}

std::string RISCVCompiler::get_compiler_path() {
    return detect_compiler();
}

std::string RISCVCompiler::compile_to_elf64(const std::string &cpp_source_path, const std::string &output_elf_path) {
    std::string compiler = detect_compiler();
    if (compiler.empty()) {
        std::cerr << "Error: RISC-V cross-compiler not found" << std::endl;
        return "";
    }
    
    // Check if source file exists
    if (!std::filesystem::exists(cpp_source_path)) {
        std::cerr << "Error: C++ source file not found: " << cpp_source_path << std::endl;
        return "";
    }
    
    // Build compiler command
    std::ostringstream cmd;
    
    if (compiler.find("zig") != std::string::npos) {
        // Zig compiler
        cmd << compiler << " -target riscv64-linux-musl "
            << "-O ReleaseSmall "
            << "-fno-stack-protector "
            << "-static "
            << cpp_source_path << " -o " << output_elf_path;
    } else {
        // GCC/Clang RISC-V cross-compiler
        cmd << compiler << " "
            << "-march=rv64gc_zba_zbb_zbs_zbc "
            << "-mabi=lp64d "
            << "-O3 "
            << "-fno-stack-protector "
            << "-fno-threadsafe-statics "
            << "-static "
            << "-nostdlib "
            << cpp_source_path << " -o " << output_elf_path;
    }
    
    // Execute compilation
    int result = std::system(cmd.str().c_str());
    if (result != 0) {
        std::cerr << "Error: Compilation failed with exit code " << result << std::endl;
        std::cerr << "Command: " << cmd.str() << std::endl;
        return "";
    }
    
    // Verify output file exists
    if (!std::filesystem::exists(output_elf_path)) {
        std::cerr << "Error: ELF64 output file was not created" << std::endl;
        return "";
    }
    
    return output_elf_path;
}

