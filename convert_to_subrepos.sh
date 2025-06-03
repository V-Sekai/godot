#!/bin/bash
# Convert ExecuTorch submodules to git subrepos

cd /Users/ernest.lee/Documents/godot

echo "Converting submodules to git subrepos..."

# Define arrays of URLs and paths (relative to godot repo root)
declare -a urls=(
    "https://git.gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-core-driver.git"
    "https://git.gitlab.arm.com/tosa/tosa-serialization.git"
    "https://github.com/KhronosGroup/Vulkan-Headers"
    "https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git"
    "https://github.com/zeux/volk"
    "https://github.com/Maratyszcza/FP16.git"
    "https://github.com/Maratyszcza/FXdiv.git"
    "https://github.com/google/XNNPACK.git"
    "https://github.com/pytorch/cpuinfo.git"
    "https://github.com/Maratyszcza/pthreadpool.git"
    "https://github.com/pytorch-labs/tokenizers.git"
    "https://gitlab.com/libeigen/eigen.git"
    "https://github.com/google/flatbuffers.git"
    "https://github.com/dvidelabs/flatcc.git"
    "https://github.com/gflags/gflags.git"
    "https://github.com/google/googletest.git"
    "https://github.com/leetal/ios-cmake"
    "https://github.com/facebook/buck2-prelude.git"
    "https://github.com/pybind/pybind11.git"
    "https://github.com/pytorch/ao.git"
    "https://github.com/pytorch-labs/FACTO.git"
    "https://github.com/mreineck/pocketfft"
    "https://github.com/facebook/buck2-shims-meta"
    "https://github.com/nlohmann/json.git"
)

declare -a paths=(
    "modules/executorch/thirdparty/executorch/backends/arm/third-party/ethos-u-core-driver"
    "modules/executorch/thirdparty/executorch/backends/arm/third-party/serialization_lib"
    "modules/executorch/thirdparty/executorch/backends/vulkan/third-party/Vulkan-Headers"
    "modules/executorch/thirdparty/executorch/backends/vulkan/third-party/VulkanMemoryAllocator"
    "modules/executorch/thirdparty/executorch/backends/vulkan/third-party/volk"
    "modules/executorch/thirdparty/executorch/backends/xnnpack/third-party/FP16"
    "modules/executorch/thirdparty/executorch/backends/xnnpack/third-party/FXdiv"
    "modules/executorch/thirdparty/executorch/backends/xnnpack/third-party/XNNPACK"
    "modules/executorch/thirdparty/executorch/backends/xnnpack/third-party/cpuinfo"
    "modules/executorch/thirdparty/executorch/backends/xnnpack/third-party/pthreadpool"
    "modules/executorch/thirdparty/executorch/extension/llm/tokenizers"
    "modules/executorch/thirdparty/executorch/kernels/optimized/third-party/eigen"
    "modules/executorch/thirdparty/executorch/third-party/flatbuffers"
    "modules/executorch/thirdparty/executorch/third-party/flatcc"
    "modules/executorch/thirdparty/executorch/third-party/gflags"
    "modules/executorch/thirdparty/executorch/third-party/googletest"
    "modules/executorch/thirdparty/executorch/third-party/ios-cmake"
    "modules/executorch/thirdparty/executorch/third-party/prelude"
    "modules/executorch/thirdparty/executorch/third-party/pybind11"
    "modules/executorch/thirdparty/executorch/third-party/ao"
    "modules/executorch/thirdparty/executorch/backends/cadence/utils/FACTO"
    "modules/executorch/thirdparty/executorch/third-party/pocketfft"
    "modules/executorch/thirdparty/executorch/shim"
    "modules/executorch/thirdparty/executorch/third-party/json"
)

# Loop through and clone each subrepo
for i in "${!urls[@]}"; do
    echo "Cloning ${urls[$i]} to ${paths[$i]}"
    git subrepo clone "${urls[$i]}" "${paths[$i]}"
    if [ $? -eq 0 ]; then
        echo "✓ Successfully cloned ${paths[$i]}"
    else
        echo "✗ Failed to clone ${paths[$i]}"
    fi
done

echo "All subrepos cloned. Committing changes..."
git add .
git commit -m "Convert all submodules to git subrepos"

echo "Conversion complete!"