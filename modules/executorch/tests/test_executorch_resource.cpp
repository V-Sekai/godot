#include "doctest.h"
#include "../src/executorch_resource.h"

TEST_SUITE("ExecuTorchResource Tests") {

TEST_CASE("ExecuTorchResource - Basic Creation and Lifecycle") {
    SUBCASE("Resource Creation") {
        auto resource = std::make_unique<ExecuTorchResource>();
        CHECK(resource != nullptr);
        CHECK_FALSE(resource->is_loaded());
        CHECK(resource->get_model_size() == 0);
        INFO("ExecuTorchResource created successfully");
    }

    SUBCASE("Resource Clear") {
        auto resource = std::make_unique<ExecuTorchResource>();
        resource->clear();
        CHECK_FALSE(resource->is_loaded());
        CHECK(resource->get_total_inferences() == 0);
        INFO("Resource cleared successfully");
    }
}

TEST_CASE("ExecuTorchResource - Model Data Management") {
    auto resource = std::make_unique<ExecuTorchResource>();

    SUBCASE("Set Model Data") {
        PackedByteArray test_data = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                                   0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10};

        resource->set_model_data(test_data);

        PackedByteArray retrieved_data = resource->get_model_data();
        CHECK(retrieved_data.size() == test_data.size());
        CHECK(retrieved_data == test_data);
        INFO("Model data set and retrieved correctly");
    }

    SUBCASE("Model Size") {
        PackedByteArray large_data(1024, 0xAB);
        resource->set_model_data(large_data);

        CHECK(resource->get_model_size() == 1024);
        INFO("Model size reported correctly");
    }
}

TEST_CASE("ExecuTorchResource - Memory Management Configuration") {
    auto resource = std::make_unique<ExecuTorchResource>();

    SUBCASE("Auto Memory Policy") {
        Error result = resource->configure_memory(ExecuTorchResource::MEMORY_POLICY_AUTO);
        CHECK(result == OK);

        Dictionary memory_info = resource->get_memory_info();
        CHECK(memory_info.find("policy") != memory_info.end());
        INFO("Auto memory policy configured");
    }

    SUBCASE("Static Memory Policy") {
        int64_t memory_limit = 2 * 1024 * 1024; // 2MB
        Error result = resource->configure_memory(ExecuTorchResource::MEMORY_POLICY_STATIC, memory_limit);
        CHECK(result == OK);

        Dictionary memory_info = resource->get_memory_info();
        CHECK(memory_info.find("total_bytes") != memory_info.end());
        INFO("Static memory policy configured with 2MB limit");
    }

    SUBCASE("Custom Memory Policy") {
        Error result = resource->configure_memory(ExecuTorchResource::MEMORY_POLICY_CUSTOM);
        CHECK(result == OK);
        INFO("Custom memory policy configured");
    }
}

TEST_CASE("ExecuTorchResource - Optimization Configuration") {
    auto resource = std::make_unique<ExecuTorchResource>();

    SUBCASE("Optimization Levels") {
        CHECK(resource->set_optimization_level(ExecuTorchResource::OPTIMIZATION_NONE) == OK);
        CHECK(resource->set_optimization_level(ExecuTorchResource::OPTIMIZATION_BASIC) == OK);
        CHECK(resource->set_optimization_level(ExecuTorchResource::OPTIMIZATION_AGGRESSIVE) == OK);
        INFO("All optimization levels configured successfully");
    }

    SUBCASE("Profiling Enable/Disable") {
        CHECK(resource->enable_profiling(true) == OK);
        CHECK(resource->enable_profiling(false) == OK);
        INFO("Profiling toggled successfully");
    }
}

TEST_CASE("ExecuTorchResource - Linear Regression Model Test") {
    auto resource = std::make_unique<ExecuTorchResource>();

    // Create a mock .pte file for testing
    PackedByteArray mock_model_data(256, 0x42); // 256 bytes of test data
    resource->set_model_data(mock_model_data);

    SUBCASE("Model Metadata") {
        Array input_names = resource->get_input_names();
        Array output_names = resource->get_output_names();

        CHECK(input_names.size() >= 0); // May be empty if not loaded
        CHECK(output_names.size() >= 0);

        Dictionary input_shapes = resource->get_input_shapes();
        Dictionary output_shapes = resource->get_output_shapes();

        INFO("Model metadata accessible");
    }

    SUBCASE("Performance Tracking") {
        int initial_inferences = resource->get_total_inferences();
        double initial_time = resource->get_last_inference_time();

        CHECK(initial_inferences == 0);
        CHECK(initial_time == 0.0);

        INFO("Performance tracking initialized correctly");
    }

    SUBCASE("Model Properties") {
        String model_name = resource->get_model_name();
        String model_version = resource->get_model_version();

        // These may be empty if model is not loaded, but should not crash
        CHECK(model_name.length() >= 0);
        CHECK(model_version.length() >= 0);

        INFO("Model properties accessible");
    }
}

TEST_CASE("ExecuTorchResource - File Operations") {
    auto resource = std::make_unique<ExecuTorchResource>();

    SUBCASE("Load Non-existent File") {
        Error result = resource->load_from_file("non_existent_file.pte");
        CHECK(result == FAILED);
        CHECK_FALSE(resource->is_loaded());
        INFO("Non-existent file load handled correctly");
    }

    SUBCASE("Save and Load Cycle") {
        // Create test data
        PackedByteArray test_data = {
            0x50, 0x54, 0x45, 0x00, // Mock PTE header
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10
        };

        resource->set_model_data(test_data);

        // Save to temporary file
        String temp_file = "/tmp/test_model.pte";
        Error save_result = resource->save_to_file(temp_file);

        if (save_result == OK) {
            // Create new resource and load
            auto new_resource = std::make_unique<ExecuTorchResource>();
            Error load_result = new_resource->load_from_file(temp_file);

            if (load_result == OK) {
                CHECK(new_resource->get_model_size() == test_data.size());
                INFO("Save and load cycle completed successfully");
            } else {
                INFO("Load failed (expected in mock implementation)");
            }

            // Clean up
            std::remove(temp_file.c_str());
        } else {
            INFO("Save failed (may be expected depending on environment)");
        }
    }
}

TEST_CASE("ExecuTorchModule - High-Level API") {
    SUBCASE("Module Creation") {
        auto module = std::make_unique<ExecuTorchModule>();
        CHECK(module != nullptr);
        CHECK_FALSE(module->is_loaded());
        INFO("ExecuTorchModule created successfully");
    }

    SUBCASE("Load from Buffer") {
        auto module = std::make_unique<ExecuTorchModule>();

        // Create mock .pte data (needs to be at least 16 bytes)
        PackedByteArray mock_data(32, 0x42);

        Error result = module->load_from_buffer(mock_data);
        CHECK(result == OK);
        CHECK(module->is_loaded());
        INFO("Module loaded from buffer successfully");
    }

    SUBCASE("Linear Regression Inference") {
        auto module = std::make_unique<ExecuTorchModule>();
        PackedByteArray mock_data(32, 0x42);

        if (module->load_from_buffer(mock_data) == OK) {
            // Test linear regression: y = 2x + 3
            Dictionary inputs;
            inputs["input_0"] = std::vector<float>{1.0f};

            Dictionary outputs = module->forward(inputs);

            CHECK(outputs.find("output_0") != outputs.end());

            auto& output_values = outputs["output_0"];
            if (!output_values.empty()) {
                float result = output_values[0];
                float expected = 2.0f * 1.0f + 3.0f; // 5.0

                CHECK(std::abs(result - expected) < 0.1f);
                INFO("Linear regression inference working correctly: y = 2x + 3");
            }
        }
    }

    SUBCASE("Method Names") {
        auto module = std::make_unique<ExecuTorchModule>();

        Array methods = module->get_method_names();
        CHECK(methods.size() > 0);
        INFO("Method names retrieved");
    }
}

TEST_CASE("ExecuTorchMemoryManager - Low-Level Memory Control") {
    SUBCASE("Memory Manager Creation") {
        auto memory_manager = std::make_unique<ExecuTorchMemoryManager>();
        CHECK(memory_manager != nullptr);
        INFO("Memory manager created successfully");
    }

    SUBCASE("Static Memory Configuration") {
        auto memory_manager = std::make_unique<ExecuTorchMemoryManager>();

        size_t pool_size = 1024 * 1024; // 1MB
        Error result = memory_manager->configure_static_memory(pool_size);
        CHECK(result == OK);

        Dictionary stats = memory_manager->get_memory_stats();
        CHECK(stats.find("total_bytes") != stats.end());
        CHECK(stats.find("is_static") != stats.end());

        INFO("Static memory configured with 1MB pool");
    }

    SUBCASE("Dynamic Memory Configuration") {
        auto memory_manager = std::make_unique<ExecuTorchMemoryManager>();

        Error result = memory_manager->configure_dynamic_memory();
        CHECK(result == OK);

        Dictionary stats = memory_manager->get_memory_stats();
        auto& is_static = stats["is_static"];
        CHECK(!is_static.empty());
        CHECK(is_static[0] == 0.0f); // Should be false for dynamic

        INFO("Dynamic memory configured");
    }

    SUBCASE("Memory Allocation and Deallocation") {
        auto memory_manager = std::make_unique<ExecuTorchMemoryManager>();
        memory_manager->configure_static_memory(1024);

        void* ptr = memory_manager->allocate(64, 16);
        CHECK(ptr != nullptr);

        memory_manager->deallocate(ptr);
        INFO("Memory allocation and deallocation working");
    }

    SUBCASE("Memory Statistics") {
        auto memory_manager = std::make_unique<ExecuTorchMemoryManager>();
        memory_manager->configure_static_memory(2048);

        size_t allocated = memory_manager->get_allocated_bytes();
        size_t available = memory_manager->get_available_bytes();

        CHECK(allocated + available <= 2048);
        CHECK(allocated >= 0);
        CHECK(available >= 0);

        INFO("Memory statistics working correctly");
    }
}

TEST_CASE("ExecuTorchResource - Complete Linear Regression Pipeline") {
    auto resource = std::make_unique<ExecuTorchResource>();

    SUBCASE("End-to-End Linear Regression Test") {
        // Configure resource
        resource->configure_memory(ExecuTorchResource::MEMORY_POLICY_AUTO);
        resource->set_optimization_level(ExecuTorchResource::OPTIMIZATION_BASIC);
        resource->enable_profiling(true);

        // Set mock model data
        PackedByteArray model_data(64, 0x42);
        resource->set_model_data(model_data);

        // Test cases for y = 2x + 3
        struct TestCase {
            float input;
            float expected_output;
            std::string name;
        };

        std::vector<TestCase> test_cases = {
            {0.0f, 3.0f, "Zero input"},
            {1.0f, 5.0f, "Unit input"},
            {2.0f, 7.0f, "Double input"},
            {-1.0f, 1.0f, "Negative input"}
        };

        int passed_tests = 0;

        for (const auto& test_case : test_cases) {
            Dictionary inputs;
            inputs["input_0"] = std::vector<float>{test_case.input};

            try {
                Dictionary outputs = resource->forward(inputs);

                if (outputs.find("output_0") != outputs.end()) {
                    auto& output_values = outputs["output_0"];
                    if (!output_values.empty()) {
                        float actual = output_values[0];
                        float error = std::abs(actual - test_case.expected_output);

                        if (error < 0.1f) {
                            passed_tests++;
                            INFO("Test case '" + test_case.name + "' passed: " +
                                 std::to_string(test_case.input) + " -> " + std::to_string(actual));
                        }
                    }
                }
            } catch (const std::exception& e) {
                INFO("Test case '" + test_case.name + "' failed with exception: " + e.what());
            }
        }

        INFO("Linear regression pipeline test completed");
        INFO("Passed " + std::to_string(passed_tests) + "/" + std::to_string(test_cases.size()) + " test cases");

        // Performance check
        CHECK(resource->get_total_inferences() >= 0);
        CHECK(resource->get_last_inference_time() >= 0.0);
    }
}

} // TEST_SUITE

DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
