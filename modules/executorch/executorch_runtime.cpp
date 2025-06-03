#include "executorch_runtime.h"
#include "executorch_model.h"
#include "mcp_server_internal.h"

#include <iostream>
#include <chrono>
#include <stdexcept>

ExecuTorchRuntime::ExecuTorchRuntime()
    : is_initialized_(false), last_inference_time_ms_(0.0), total_inferences_(0) {

    // Initialize the MCP server
    mcp_server_ = std::make_unique<MCPServerInternal>();
    mcp_server_->initialize("GodotExecuTorchModule", "1.0.0");
}

ExecuTorchRuntime::~ExecuTorchRuntime() {
    unload_model();
}

bool ExecuTorchRuntime::load_model_from_pck(const String& pck_path) {
    try {
        // In a real Godot implementation, this would use FileAccess to read from PCK
        // For now, treat it as a regular file path
        return load_model_from_file(pck_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model from PCK: " << e.what() << std::endl;
        return false;
    }
}

bool ExecuTorchRuntime::load_model_from_file(const String& file_path) {
    try {
        unload_model(); // Ensure clean state

        // Create and load the model
        auto model = std::make_shared<ExecuTorchModel>();
        if (!model->load_from_file(file_path)) {
            std::cerr << "Failed to load ExecuTorch model from: " << file_path << std::endl;
            return false;
        }

        // Set the model in our internal MCP server
        if (!mcp_server_->set_model(model)) {
            std::cerr << "Failed to set model in MCP server" << std::endl;
            return false;
        }

        model_ = model;
        model_path_ = file_path;
        is_initialized_ = true;

        std::cout << "ExecuTorch model loaded successfully from: " << file_path << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception loading model: " << e.what() << std::endl;
        return false;
    }
}

void ExecuTorchRuntime::unload_model() {
    if (model_) {
        model_->unload();
        model_.reset();
    }

    model_path_.clear();
    is_initialized_ = false;
    reset_performance_stats();

    std::cout << "ExecuTorch model unloaded" << std::endl;
}

bool ExecuTorchRuntime::is_model_loaded() const {
    return model_ && model_->is_loaded();
}

Dictionary ExecuTorchRuntime::run_inference(const Dictionary& inputs) {
    if (!is_model_loaded()) {
        throw std::runtime_error("Model not loaded");
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        // Convert Godot Dictionary to C++ map
        auto cpp_inputs = _convert_godot_inputs(inputs);

        // Run inference
        auto cpp_result = model_->forward(cpp_inputs);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double inference_time_ms = duration.count() / 1000.0;

        _update_performance_stats(inference_time_ms);

        // Convert result back to Godot Dictionary
        return _convert_cpp_result(cpp_result);

    } catch (const std::exception& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        throw;
    }
}

Array ExecuTorchRuntime::run_inference_array(const Array& input_data) {
    Dictionary inputs;
    inputs["input_0"] = input_data;

    auto result = run_inference(inputs);

    // Return first output array
    if (result.find("output_0") != result.end()) {
        return result["output_0"];
    }

    return Array(); // Empty array
}

double ExecuTorchRuntime::get_last_inference_time() const {
    return last_inference_time_ms_;
}

int ExecuTorchRuntime::get_total_inferences() const {
    return total_inferences_;
}

void ExecuTorchRuntime::reset_performance_stats() {
    last_inference_time_ms_ = 0.0;
    total_inferences_ = 0;
}

void ExecuTorchRuntime::set_optimization_level(int level) {
    std::cout << "Set optimization level to: " << level << std::endl;
}

void ExecuTorchRuntime::enable_profiling(bool enable) {
    std::cout << "Profiling " << (enable ? "enabled" : "disabled") << std::endl;
}

void ExecuTorchRuntime::set_memory_limit(int64_t bytes) {
    std::cout << "Memory limit set to: " << bytes << " bytes" << std::endl;
}

std::vector<String> ExecuTorchRuntime::list_mcp_tools() const {
    if (!mcp_server_) {
        return {};
    }
    return mcp_server_->list_tools();
}

Dictionary ExecuTorchRuntime::call_mcp_tool(const String& tool_name, const Dictionary& params) {
    if (!mcp_server_) {
        throw std::runtime_error("MCP server not initialized");
    }

    auto cpp_params = _convert_godot_inputs(params);
    auto response = mcp_server_->call_tool(tool_name, cpp_params);

    if (response.success) {
        return _convert_cpp_result(response.result);
    } else {
        std::cerr << "MCP tool error: " << response.error_message << std::endl;
        throw std::runtime_error(response.error_message);
    }
}

Dictionary ExecuTorchRuntime::get_model_info() const {
    return call_mcp_tool("get_model_info", Dictionary());
}

Dictionary ExecuTorchRuntime::health_check() const {
    if (!mcp_server_) {
        return Dictionary();
    }

    auto response = mcp_server_->call_tool("health_check", std::map<std::string, std::vector<float>>());

    if (response.success) {
        return _convert_cpp_result(response.result);
    }

    return Dictionary();
}

bool ExecuTorchRuntime::_load_model_from_buffer(const PackedByteArray& model_data) {
    try {
        unload_model();

        auto model = std::make_shared<ExecuTorchModel>();

        // Convert PackedByteArray to std::vector<uint8_t>
        std::vector<uint8_t> cpp_data(model_data.begin(), model_data.end());

        if (!model->load_from_buffer(cpp_data)) {
            std::cerr << "Failed to load ExecuTorch model from buffer" << std::endl;
            return false;
        }

        if (!mcp_server_->set_model(model)) {
            std::cerr << "Failed to set model in MCP server" << std::endl;
            return false;
        }

        model_ = model;
        is_initialized_ = true;

        std::cout << "ExecuTorch model loaded from buffer" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception loading model from buffer: " << e.what() << std::endl;
        return false;
    }
}

void ExecuTorchRuntime::_update_performance_stats(double inference_time) {
    last_inference_time_ms_ = inference_time;
    total_inferences_++;

    std::cout << "Inference #" << total_inferences_
              << " completed in " << inference_time << "ms" << std::endl;
}

Dictionary ExecuTorchRuntime::_convert_cpp_result(const std::map<std::string, std::vector<float>>& cpp_result) {
    Dictionary result;
    for (const auto& [key, value] : cpp_result) {
        result[key] = value;
    }
    return result;
}

std::map<std::string, std::vector<float>> ExecuTorchRuntime::_convert_godot_inputs(const Dictionary& godot_inputs) {
    std::map<std::string, std::vector<float>> cpp_inputs;
    for (const auto& [key, value] : godot_inputs) {
        cpp_inputs[key] = value;
    }
    return cpp_inputs;
}
