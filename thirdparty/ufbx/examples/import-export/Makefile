CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

# Source files
UFBX_SRC = ../../ufbx.c ../../ufbx_export.c  ../../ufbx_export_writer.c
MODULE_SRC = error_utils.c scene_analysis.c scene_copy.c main.c
ALL_SRC = $(MODULE_SRC) $(UFBX_SRC)

# Output executables
ROUNDTRIP_BIN = roundtrip
VALIDATE_BIN = validate

# Default target
all: $(ROUNDTRIP_BIN) $(VALIDATE_BIN)

# Build roundtrip test from modular components
$(ROUNDTRIP_BIN): $(MODULE_SRC) $(UFBX_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Build validation tool
$(VALIDATE_BIN): validate.c $(UFBX_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Test with huesitos.fbx
test-huesitos: $(ROUNDTRIP_BIN)
	./$(ROUNDTRIP_BIN) ./data/huesitos.fbx huesitos_roundtrip.fbx

# Test all FBX files systematically (excluding Blender files)
test-all: $(ROUNDTRIP_BIN)
	@echo "=== Systematic FBX Roundtrip Testing ==="
	@echo "Testing all FBX files in ../../data directory (excluding Blender files)..."
	@mkdir -p test_results
	@success_count=0; \
	total_count=0; \
	failed_files=""; \
	for fbx_file in $$(find ../../data ./data -name "*.fbx" 2>/dev/null | grep -v -i blender | sort); do \
		total_count=$$((total_count + 1)); \
		filename=$$(basename "$$fbx_file" .fbx); \
		output_file="test_results/$${filename}_roundtrip.fbx"; \
		echo ""; \
		echo "Testing $$total_count: $$fbx_file"; \
		echo "Output: $$output_file"; \
		if ./$(ROUNDTRIP_BIN) "$$fbx_file" "$$output_file" > "test_results/$${filename}.log" 2>&1; then \
			echo "✓ SUCCESS: $$filename"; \
			success_count=$$((success_count + 1)); \
		else \
			echo "✗ FAILED: $$filename"; \
			failed_files="$$failed_files $$filename"; \
		fi; \
	done; \
	echo ""; \
	echo "=== Test Summary ==="; \
	echo "Total files tested: $$total_count"; \
	echo "Successful: $$success_count"; \
	echo "Failed: $$((total_count - success_count))"; \
	if [ $$success_count -eq $$total_count ]; then \
		echo "🎉 ALL TESTS PASSED!"; \
	else \
		echo "❌ Some tests failed:$$failed_files"; \
		echo "Check test_results/*.log for details"; \
	fi

# Test golden path cases only (exclude known problematic files and Blender files)
test-golden: $(ROUNDTRIP_BIN)
	@echo "=== Golden Path FBX Roundtrip Testing ==="
	@echo "Testing well-formed FBX files (excluding fuzz, fail, edge cases, and Blender files)..."
	@mkdir -p test_results
	@success_count=0; \
	total_count=0; \
	failed_files=""; \
	for fbx_file in $$(find ../../data ./data -name "*.fbx" 2>/dev/null | grep -v -E "(fuzz_|_fail_|edge_|cursed|maya_bad_face)" | grep -v -i blender | sort); do \
		total_count=$$((total_count + 1)); \
		filename=$$(basename "$$fbx_file" .fbx); \
		output_file="test_results/$${filename}_roundtrip.fbx"; \
		echo ""; \
		echo "Testing $$total_count: $$fbx_file"; \
		echo "Output: $$output_file"; \
		if ./$(ROUNDTRIP_BIN) "$$fbx_file" "$$output_file" > "test_results/$${filename}.log" 2>&1; then \
			echo "✓ SUCCESS: $$filename"; \
			success_count=$$((success_count + 1)); \
		else \
			echo "✗ FAILED: $$filename"; \
			failed_files="$$failed_files $$filename"; \
		fi; \
	done; \
	echo ""; \
	echo "=== Golden Path Test Summary ==="; \
	echo "Total files tested: $$total_count"; \
	echo "Successful: $$success_count"; \
	echo "Failed: $$((total_count - success_count))"; \
	if [ $$success_count -eq $$total_count ]; then \
		echo "🎉 ALL GOLDEN PATH TESTS PASSED!"; \
	else \
		echo "❌ Some golden path tests failed:$$failed_files"; \
		echo "Check test_results/*.log for details"; \
	fi

# Test golden path with automatic validation
test-golden-validated: $(ROUNDTRIP_BIN) $(VALIDATE_BIN)
	@echo "=== Golden Path Testing with Validation ==="
	@echo "Testing well-formed FBX files with automatic data preservation validation (excluding Blender files)..."
	@mkdir -p test_results
	@success_count=0; \
	validation_pass=0; \
	total_count=0; \
	failed_files=""; \
	validation_failed=""; \
	for fbx_file in $$(find ../../data ./data -name "*.fbx" 2>/dev/null | grep -v -E "(fuzz_|_fail_|edge_|cursed|maya_bad_face)" | grep -v -i blender | sort | head -20); do \
		total_count=$$((total_count + 1)); \
		filename=$$(basename "$$fbx_file" .fbx); \
		output_file="test_results/$${filename}_roundtrip.fbx"; \
		echo ""; \
		echo "Testing $$total_count: $$fbx_file"; \
		echo "Output: $$output_file"; \
		if ./$(ROUNDTRIP_BIN) "$$fbx_file" "$$output_file" > "test_results/$${filename}.log" 2>&1; then \
			echo "✓ ROUNDTRIP SUCCESS: $$filename"; \
			success_count=$$((success_count + 1)); \
			echo "  Validating data preservation..."; \
			if ./$(VALIDATE_BIN) "$$fbx_file" "$$output_file" > "test_results/$${filename}_validation.log" 2>&1; then \
				echo "  ✓ VALIDATION PASSED: Data preserved correctly"; \
				validation_pass=$$((validation_pass + 1)); \
			else \
				echo "  ⚠️  VALIDATION FAILED: Data loss detected"; \
				validation_failed="$$validation_failed $$filename"; \
			fi; \
		else \
			echo "✗ ROUNDTRIP FAILED: $$filename"; \
			failed_files="$$failed_files $$filename"; \
		fi; \
	done; \
	echo ""; \
	echo "=== Golden Path Validation Summary ==="; \
	echo "Total files tested: $$total_count"; \
	echo "Roundtrip successful: $$success_count"; \
	echo "Validation passed: $$validation_pass"; \
	echo "Roundtrip failed: $$((total_count - success_count))"; \
	echo "Validation failed: $$((success_count - validation_pass))"; \
	if [ $$success_count -eq $$total_count ] && [ $$validation_pass -eq $$success_count ]; then \
		echo "🎉 ALL TESTS PASSED WITH PERFECT DATA PRESERVATION!"; \
	else \
		if [ $$success_count -ne $$total_count ]; then \
			echo "❌ Roundtrip failures:$$failed_files"; \
		fi; \
		if [ $$validation_pass -ne $$success_count ]; then \
			echo "⚠️  Data validation failures:$$validation_failed"; \
		fi; \
		echo "Check test_results/*.log and *_validation.log for details"; \
	fi

# Test specific animations and complex models
test-complex: $(ROUNDTRIP_BIN)
	@echo "=== Testing Complex FBX Files ==="
	@mkdir -p test_results
	@for file in "../../data/blender_293_barbarian_7400_binary.fbx" \
	             "../../data/blender440_shape_weight_anim_7400_binary.fbx" \
	             "../../data/max7_skin_5000_binary.fbx" \
	             "../../data/max2009_cube_anim_6100_binary.fbx"; do \
		if [ -f "$$file" ]; then \
			filename=$$(basename "$$file" .fbx); \
			echo "Testing: $$file"; \
			./$(ROUNDTRIP_BIN) "$$file" "test_results/$${filename}_roundtrip.fbx"; \
		fi; \
	done

# Quick verification test with a few representative files
test-verify: $(ROUNDTRIP_BIN)
	@echo "=== Quick Verification Test ==="
	@mkdir -p test_results
	@for file in "./data/huesitos.fbx" \
	             "../../data/blender_279_default_7400_binary.fbx" \
	             "../../data/max7_cube_5000_binary.fbx"; do \
		if [ -f "$$file" ]; then \
			filename=$$(basename "$$file" .fbx); \
			echo "Testing: $$file"; \
			./$(ROUNDTRIP_BIN) "$$file" "test_results/$${filename}_roundtrip.fbx"; \
		fi; \
	done

# Clean build artifacts and test results
clean:
	rm -f $(ROUNDTRIP_BIN) $(VALIDATE_BIN) *.fbx
	rm -rf test_results

# Show help
help:
	@echo "Available targets:"
	@echo "  all                 - Build roundtrip and validation tools"
	@echo "  test-huesitos       - Test roundtrip with huesitos.fbx"
	@echo "  test-all            - Test ALL FBX files in ../../data (comprehensive)"
	@echo "  test-golden         - Test golden path files (exclude fuzz/fail/edge cases)"
	@echo "  test-golden-validated - Test golden path with automatic validation"
	@echo "  test-complex        - Test complex models with animations/skinning"
	@echo "  test-verify         - Quick verification with representative files"
	@echo "  clean               - Clean build artifacts and test results"
	@echo "  help                - Show this help"
	@echo ""
	@echo "Test targets:"
	@echo "  test-all            - Systematic testing of all ufbx data files"
	@echo "  test-golden         - Golden path testing (well-formed files only)"
	@echo "  test-golden-validated - Golden path with data preservation validation"
	@echo "  test-complex        - Focus on barbarian, animations, skinning"
	@echo "  test-verify         - Quick test with 3 representative files"
	@echo ""
	@echo "Source files:"
	@echo "  main.c              - Main roundtrip test program"
	@echo "  validate.c          - Roundtrip validation tool"
	@echo "  error_utils.c       - Error handling utilities"
	@echo "  scene_analysis.c    - Scene analysis and printing"
	@echo "  scene_copy.c        - Scene data copying functionality"
	@echo "  scene_utils.h       - Header for all utilities"
	@echo ""
	@echo "Manual usage:"
	@echo "  ./roundtrip <input.fbx> <output.fbx>"
	@echo "  ./validate <input.fbx> <roundtrip.fbx>"
	@echo "  ./roundtrip ./data/huesitos.fbx huesitos_test.fbx"

.PHONY: all test-huesitos test-all test-golden test-golden-validated test-complex test-verify clean help
