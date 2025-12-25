# Makefile for Godot CNode
# Used by elixir_make to compile the CNode using SCons

# Default target - use SCons to build
all:
	@echo "Building CNode using SCons..."
	cd cnode && python3 -c "import os; os.chdir('..'); exec(open('cnode/SConstruct').read())" || cd cnode && scons -f SConstruct

# Clean build artifacts
clean:
	@echo "Cleaning CNode build artifacts..."
	cd cnode && scons -f SConstruct -c
	rm -rf priv/godot_cnode priv/godot_cnode.exe cnode/*.o

.PHONY: all clean
