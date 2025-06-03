#ifndef EXECUTORCH_RUNTIME_H
#define EXECUTORCH_RUNTIME_H

// Simplified header for module compilation
// This will be replaced with proper Godot headers when building within Godot

#include <string>
#include <map>
#include <vector>
#include <memory>

// Forward declarations
class ExecuTorchModel;
class MCPServerInternal;

// Simplified types for standalone compilation
using String = std::string;
using Dictionary = std::map<std::string, std::vector<float>>;
using Array = std::vector<float>;
using PackedByteArray = std::vector<uint8_t>;

class ExecuTorchRuntime {
private:
    std::shared_ptr<ExecuTorchModel> model_;
    std::unique_ptr<MCPServerInternal> mcp_server_;
    bool is_initialized_;
    String model_path_;

    // Performance metrics
    double last_inference_time_ms_;
    int total_inferences_;

public:
    ExecuTorchRuntime();
    virtual ~ExecuTorchRuntime();

    // Core model management
    bool load_model_from_pck(const String& pck_path);
    bool load_model_from_file(const String& file_path);
    void unload_model();
    bool is_model_loaded() const;

    // Inference methods
    Dictionary run_inference(const Dictionary& inputs);
    Array run_inference_array(const Array& input_data);

    // Performance and diagnostics
    double get_last_inference_time() const;
    int get_total_inferences() const;
    void reset_performance_stats();

    // Configuration
    void set_optimization_level(int level);
    void enable_profiling(bool enable);
    void set_memory_limit(int64_t bytes);

    // MCP tools interface
    std::vector<String> list_mcp_tools() const;
    Dictionary call_mcp_tool(const String& tool_name, const Dictionary& params);
    Dictionary get_model_info() const;
    Dictionary health_check() const;

private:
    // Internal helpers
    bool _load_model_from_buffer(const PackedByteArray& model_data);
    void _update_performance_stats(double inference_time);
    Dictionary _convert_cpp_result(const std::map<std::string, std::vector<float>>& cpp_result);
    std::map<std::string, std::vector<float>> _convert_godot_inputs(const Dictionary& godot_inputs);
};

#endif // EXECUTORCH_RUNTIME_H
