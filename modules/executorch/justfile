# ExecuTorch Module Commands
# Run with: just <command>

install:
    pip install torch torchvision executorch
    @echo "✅ Dependencies installed"

setup:
    mkdir -p models
    mkdir -p scripts
    @echo "✅ Directory structure created"

convert:
    cd scripts && python convert_model.py
    @echo "✅ Model conversion completed"

test:
    cd scripts && python run.py
    @echo "✅ Model testing completed"

all: install setup convert test
    @echo "🎉 Full pipeline completed successfully!"

clean:
    rm -f models/*.pte
    rm -f models/*.pth
    @echo "✅ Cleaned up generated files"

info:
    @echo "=== ExecuTorch Models ==="
    @ls -la models/ 2>/dev/null || echo "No models directory found"
    @echo ""
    @echo "=== Available Commands ==="
    @just --list

quick-test:
    cd scripts && python run.py

help:
    @echo "ExecuTorch Module Build System"
    @echo ""
    @echo "Available commands:"
    @just --list
