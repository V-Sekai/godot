// Minimal doctest implementation for the ExecuTorch module
// This is a simplified version of doctest for integration with Godot's build system

#ifndef DOCTEST_H
#define DOCTEST_H

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <functional>

namespace doctest {

    struct TestResult {
        bool passed;
        std::string name;
        std::string error_message;
    };

    class TestRunner {
    private:
        std::vector<std::function<void()>> tests_;
        std::vector<TestResult> results_;
        std::string current_test_name_;
        std::string current_subcase_;
        bool current_test_failed_;

        TestRunner() = default;

    public:
        static TestRunner& getInstance() {
            static TestRunner instance;
            return instance;
        }

        void addTest(const std::string& name, std::function<void()> test) {
            tests_.push_back([this, name, test]() {
                current_test_name_ = name;
                current_test_failed_ = false;
                current_subcase_.clear();

                try {
                    test();
                    if (!current_test_failed_) {
                        results_.push_back({true, name, ""});
                    }
                } catch (const std::exception& e) {
                    results_.push_back({false, name, e.what()});
                }
            });
        }

        void fail(const std::string& message, const std::string& file, int line) {
            if (!current_test_failed_) {
                std::stringstream ss;
                ss << file << ":" << line << " - " << message;
                if (!current_subcase_.empty()) {
                    ss << " (in subcase: " << current_subcase_ << ")";
                }
                results_.push_back({false, current_test_name_, ss.str()});
                current_test_failed_ = true;
            }
        }

        void setSubcase(const std::string& name) {
            current_subcase_ = name;
        }

        int runTests() {
            std::cout << "Running " << tests_.size() << " tests...\n\n";

            for (auto& test : tests_) {
                test();
            }

            // Print results
            int passed = 0;
            int failed = 0;

            for (const auto& result : results_) {
                if (result.passed) {
                    std::cout << "✓ " << result.name << std::endl;
                    passed++;
                } else {
                    std::cout << "✗ " << result.name << ": " << result.error_message << std::endl;
                    failed++;
                }
            }

            std::cout << "\n--- Test Summary ---" << std::endl;
            std::cout << "Total: " << (passed + failed) << std::endl;
            std::cout << "Passed: " << passed << std::endl;
            std::cout << "Failed: " << failed << std::endl;

            return failed;
        }
    };
}

// Macros
#define TEST_CASE(name) \
    static void DOCTEST_ANONYMOUS(test_func_)(); \
    static bool DOCTEST_ANONYMOUS(registered_) = []() { \
        doctest::TestRunner::getInstance().addTest(name, DOCTEST_ANONYMOUS(test_func_)); \
        return true; \
    }(); \
    static void DOCTEST_ANONYMOUS(test_func_)()

#define DOCTEST_ANONYMOUS(prefix) DOCTEST_CAT(prefix, __COUNTER__)
#define DOCTEST_CAT(a, b) DOCTEST_CAT_IMPL(a, b)
#define DOCTEST_CAT_IMPL(a, b) a##b

#define TEST_SUITE(name) namespace test_suite_##__LINE__

#define SUBCASE(name) doctest::TestRunner::getInstance().setSubcase(name);

#define CHECK(condition) \
    if (!(condition)) { \
        doctest::TestRunner::getInstance().fail("CHECK failed: " #condition, __FILE__, __LINE__); \
    }

#define CHECK_FALSE(condition) \
    if (condition) { \
        doctest::TestRunner::getInstance().fail("CHECK_FALSE failed: " #condition, __FILE__, __LINE__); \
    }

#define REQUIRE(condition) \
    if (!(condition)) { \
        doctest::TestRunner::getInstance().fail("REQUIRE failed: " #condition, __FILE__, __LINE__); \
        throw std::runtime_error("REQUIRE failed"); \
    }

#define CHECK_THROWS(expression) \
    { \
        bool threw = false; \
        try { \
            expression; \
        } catch (...) { \
            threw = true; \
        } \
        if (!threw) { \
            doctest::TestRunner::getInstance().fail("CHECK_THROWS failed: " #expression " did not throw", __FILE__, __LINE__); \
        } \
    }

#define INFO(message) \
    std::cout << "[INFO] " << message << std::endl;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN \
    int main() { \
        return doctest::TestRunner::getInstance().runTests(); \
    }

#endif // DOCTEST_H
