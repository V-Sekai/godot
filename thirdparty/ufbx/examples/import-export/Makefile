CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

# Source files
UFBX_SRC = ../../ufbx.c ../../ufbx_export.c  ../../ufbx_export_writer.c
MODULE_SRC = error_utils.c scene_analysis.c scene_copy.c main.c
ALL_SRC = $(MODULE_SRC) $(UFBX_SRC)

# Output executable
ROUNDTRIP_BIN = roundtrip

# Default target
all: $(ROUNDTRIP_BIN)

# Build roundtrip test from modular components
$(ROUNDTRIP_BIN): $(MODULE_SRC) $(UFBX_SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Test with huesitos.fbx
test-huesitos: $(ROUNDTRIP_BIN)
	./$(ROUNDTRIP_BIN) ./data/huesitos.fbx huesitos_roundtrip.fbx

# Clean build artifacts
clean:
	rm -f $(ROUNDTRIP_BIN) *.fbx

# Show help
help:
	@echo "Available targets:"
	@echo "  all           - Build modular roundtrip test program"
	@echo "  test-huesitos - Test roundtrip with huesitos.fbx"
	@echo "  clean         - Clean build artifacts"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Source files:"
	@echo "  main.c        - Main roundtrip test program"
	@echo "  error_utils.c - Error handling utilities"
	@echo "  scene_analysis.c - Scene analysis and printing"
	@echo "  scene_copy.c  - Scene data copying functionality"
	@echo "  scene_utils.h - Header for all utilities"
	@echo ""
	@echo "Manual usage:"
	@echo "  ./roundtrip <input.fbx> <output.fbx>"
	@echo "  ./roundtrip ./data/huesitos.fbx huesitos_test.fbx"

.PHONY: all test-huesitos clean help
