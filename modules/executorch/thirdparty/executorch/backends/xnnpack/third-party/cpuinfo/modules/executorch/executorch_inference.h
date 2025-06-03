/**************************************************************************/
/*  executorch_inference.h                                               */
/**************************************************************************/

#pragma once

#include "executorch_model.h"
#include "executorch_runtime.h"
#include <memory>

// Convenience wrapper for simple inference use cases
class ExecuTorchInference {
private:
    std::unique_ptr<ExecuTorchRuntime> runtime_;
    std::unique_ptr<ExecuTorchModel> model_;
    bool auto_manage_runtime_;

public:
    ExecuTorchInference(bool auto_manage = true);
    ~ExecuTorchInference();

    // Simple API for quick setup
    bool load_model(const std::string &file_path);
    std::vector<float> predict(const std::vector<float> &input);
    
    // Advanced API for when you need more control
    ExecuTorchRuntime* get_runtime() { return runtime_.get(); }
    ExecuTorchModel* get_model() { return model_.get(); }
    
    // Allow using external runtime (for sharing between models)
    void set_runtime(ExecuTorchRuntime* external_runtime);
};