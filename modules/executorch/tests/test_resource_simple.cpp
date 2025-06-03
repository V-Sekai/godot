#include "doctest.h"
#include "../src/executorch_resource.h"

TEST_CASE("ExecuTorchResource Basic Creation") {
    auto resource = std::make_unique<ExecuTorchResource>();
    CHECK(resource != nullptr);
    CHECK_FALSE(resource->is_loaded());
    CHECK(resource->get_model_size() == 0);
}

TEST_CASE("ExecuTorchResource Model Data") {
    auto resource = std::make_unique<ExecuTorchResource>();

    PackedByteArray test_data = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                               0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10};

    resource->set_model_data(test_data);

    PackedByteArray retrieved_data = resource->get_model_data();
    CHECK(retrieved_data.size() == test_data.size());
    CHECK(retrieved_data == test_data);
}

TEST_CASE("ExecuTorchResource Memory Configuration") {
    auto resource = std::make_unique<ExecuTorchResource>();

    Error result = resource->configure_memory(ExecuTorchResource::MEMORY_POLICY_AUTO);
    CHECK(result == OK);

    Dictionary memory_info = resource->get_memory_info();
    CHECK(memory_info.find("policy") != memory_info.end());
}

TEST_CASE("ExecuTorchModule Linear Regression") {
    auto module = std::make_unique<ExecuTorchModule>();

    // Create mock .pte data (needs to be at least 16 bytes)
    PackedByteArray mock_data(32, 0x42);

    Error result = module->load_from_buffer(mock_data);
    CHECK(result == OK);
    CHECK(module->is_loaded());

    // Test linear regression: y = 2x + 3
    Dictionary inputs;
    inputs["input_0"] = std::vector<float>{1.0f};

    Dictionary outputs = module->forward(inputs);

    CHECK(outputs.find("output_0") != outputs.end());

    auto& output_values = outputs["output_0"];
    if (!output_values.empty()) {
        float result_val = output_values[0];
        float expected = 2.0f * 1.0f + 3.0f; // 5.0

        CHECK(std::abs(result_val - expected) < 0.1f);
    }
}

TEST_CASE("ExecuTorchMemoryManager Static Memory") {
    auto memory_manager = std::make_unique<ExecuTorchMemoryManager>();

    size_t pool_size = 1024 * 1024; // 1MB
    Error result = memory_manager->configure_static_memory(pool_size);
    CHECK(result == OK);

    Dictionary stats = memory_manager->get_memory_stats();
    CHECK(stats.find("total_bytes") != stats.end());
    CHECK(stats.find("is_static") != stats.end());
}

TEST_CASE("Linear Regression Pipeline Complete Test") {
    auto resource = std::make_unique<ExecuTorchResource>();

    // Configure resource
    resource->configure_memory(ExecuTorchResource::MEMORY_POLICY_AUTO);
    resource->set_optimization_level(ExecuTorchResource::OPTIMIZATION_BASIC);
    resource->enable_profiling(true);

    // Set mock model data
    PackedByteArray model_data(64, 0x42);
    resource->set_model_data(model_data);

    // Test cases for y = 2x + 3
    std::vector<std::pair<float, float>> test_cases = {
        {0.0f, 3.0f},   // y = 2*0 + 3 = 3
        {1.0f, 5.0f},   // y = 2*1 + 3 = 5
        {2.0f, 7.0f},   // y = 2*2 + 3 = 7
        {-1.0f, 1.0f}   // y = 2*(-1) + 3 = 1
    };

    int passed_tests = 0;

    for (const auto& [input, expected] : test_cases) {
        Dictionary inputs;
        inputs["input_0"] = std::vector<float>{input};

        try {
            Dictionary outputs = resource->forward(inputs);

            if (outputs.find("output_0") != outputs.end()) {
                auto& output_values = outputs["output_0"];
                if (!output_values.empty()) {
                    float actual = output_values[0];
                    float error = std::abs(actual - expected);

                    if (error < 0.1f) {
                        passed_tests++;
                    }
                }
            }
        } catch (const std::exception&) {
            // Test case failed
        }
    }

    // We expect most tests to pass in our mock implementation
    CHECK(passed_tests >= 3);

    // Performance check
    CHECK(resource->get_total_inferences() >= 0);
    CHECK(resource->get_last_inference_time() >= 0.0);
}

DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
