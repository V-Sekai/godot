# ExecuTorch Stories110M Demo C++ Application

This is a native C++ demo application that demonstrates text generation using the ExecuTorch framework for stories110M model.

## Build instructions

0. Export the stories110M model. See [../python/README.md](../python/README.md) for Python export scripts.

1. Build the project using SConstruct (replaces CMake):
   ```bash
   scons
   ```

2. Run the demo application:
   ```bash
   ./executorch_stories_demo_app
   ```

## Dependencies

- SCons 4.0 or higher
- C++17 compatible compiler
- ExecuTorch library (release/0.6)

## Notes

- This demo uses simulated text generation since the actual model requires separate export and loading
- In a real implementation, you would:
  - Export stories110M model using the Python export script
  - Load the `.pte` model file using Module class
  - Run autoregressive text generation with Metal backend for Apple Silicon GPU acceleration
- The demo includes a simple tokenizer interface (simplified for demo purposes; real apps would use sentencepiece or tiktoken)
