#ifndef EXECUTORCH_MODEL_H
#define EXECUTORCH_MODEL_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstdint>

class ExecuTorchModel {
private:
    bool is_loaded_;
    std::map<std::string, std::vector<int64_t>> input_shapes_;
    std::map<std::string, std::vector<int64_t>> output_shapes_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

public:
    ExecuTorchModel();
    ~ExecuTorchModel();

    // Core loading functionality
    bool load_from_file(const std::string& file_path);
    bool load_from_buffer(const std::vector<uint8_t>& model_data);
    void unload();
    bool is_loaded() const { return is_loaded_; }

    // Model metadata
    std::vector<std::string> get_input_names() const { return input_names_; }
    std::vector<std::string> get_output_names() const { return output_names_; }
    std::vector<int64_t> get_input_shape(const std::string& name) const;
    std::vector<int64_t> get_output_shape(const std::string& name) const;

    // Inference
    std::map<std::string, std::vector<float>> forward(const std::map<std::string, std::vector<float>>& inputs);
    std::vector<float> forward_single(const std::vector<float>& input);
    std::map<std::string, std::vector<float>> forward_named(const std::map<std::string, std::vector<float>>& inputs);

private:
    bool _initialize_metadata();
};

#endif // EXECUTORCH_MODEL_H
