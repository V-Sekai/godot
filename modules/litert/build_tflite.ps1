# Helper script to build TensorFlow Lite for Windows (PowerShell)
# This builds TFLite separately with CMake, then SCons will find it automatically

# Don't stop on warnings - CMake may output warnings that are not fatal
$ErrorActionPreference = "Continue"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Join-Path (Split-Path -Parent $SCRIPT_DIR) ".."
$TFLITE_SOURCE = Join-Path (Join-Path $PROJECT_ROOT "thirdparty") "tensorflow-lite"
$TFLITE_BUILD = Join-Path $TFLITE_SOURCE "build"

Write-Host "Building TensorFlow Lite..."
Write-Host "Source: $TFLITE_SOURCE"
Write-Host "Build: $TFLITE_BUILD"
Write-Host "Note: SCons will automatically find the library in the build directory"
Write-Host ""

# Check for CMake
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Host "Error: CMake is not installed"
    Write-Host "Install with: scoop install cmake  (Scoop) or download from https://cmake.org/download/"
    exit 1
}

# Check if source exists
if (-not (Test-Path $TFLITE_SOURCE)) {
    Write-Host "Error: TensorFlow Lite source not found at $TFLITE_SOURCE"
    Write-Host "Please ensure thirdparty\tensorflow-lite exists (git subrepo)"
    exit 1
}

# Create build directory
if (-not (Test-Path $TFLITE_BUILD)) {
    New-Item -ItemType Directory -Path $TFLITE_BUILD | Out-Null
}
Push-Location $TFLITE_BUILD

try {
    # Configure CMake
    Write-Host "Configuring CMake..."
    $tflitePath = Join-Path (Join-Path $TFLITE_SOURCE "tensorflow") "lite"
    
    # Try Visual Studio generator first (for Windows)
    $cmakeArgs = @(
        $tflitePath,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DTFLITE_ENABLE_GPU=OFF",
        "-DTFLITE_ENABLE_XNNPACK=OFF",
        "-Wno-dev"  # Suppress developer warnings
    )
    
    # Check if Visual Studio is available, if so use it
    # Try to find VS2022, VS2019, or VS2017
    $vsGenerator = $null
    if (Test-Path "C:\Program Files\Microsoft Visual Studio\2022\*\MSBuild\Current\Bin\MSBuild.exe") {
        $vsGenerator = "Visual Studio 17 2022"
    } elseif (Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\2019\*\MSBuild\Current\Bin\MSBuild.exe") {
        $vsGenerator = "Visual Studio 16 2019"
    } elseif (Test-Path "C:\Program Files (x86)\Microsoft Visual Studio\2017\*\MSBuild\15.0\Bin\MSBuild.exe") {
        $vsGenerator = "Visual Studio 15 2017"
    }
    
    if ($vsGenerator) {
        Write-Host "Using Visual Studio generator: $vsGenerator"
        $cmakeArgs += @("-G", $vsGenerator, "-A", "x64")
    } else {
        Write-Host "Using Ninja generator (Visual Studio not found)"
        $cmakeArgs += @("-G", "Ninja")
        # Add compiler flags to define max() macro for Windows compatibility
        # This fixes cpuinfo max() issue without editing third-party code
        $cmakeArgs += @("-DCMAKE_C_FLAGS=-Dmax(a,b)=((a)>(b)?(a):(b))")
    }
    
    Write-Host "Running: cmake $($cmakeArgs -join ' ')"
    & cmake $cmakeArgs 2>&1 | ForEach-Object { 
        if ($_ -match "CMake Error") { 
            Write-Host $_ -ForegroundColor Red
        } elseif ($_ -match "CMake Warning") {
            Write-Host $_ -ForegroundColor Yellow
        } else {
            Write-Host $_
        }
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: CMake configuration failed"
        Write-Host "Trying alternative: building from tensorflow root..."
        Pop-Location
        Push-Location $TFLITE_SOURCE
        if (-not (Test-Path "build")) {
            New-Item -ItemType Directory -Path "build" | Out-Null
        }
        Push-Location "build"
        
        $cmakeArgs = @(
            "..",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTFLITE_ENABLE_GPU=OFF",
            "-DTFLITE_ENABLE_XNNPACK=OFF",
            "-Wno-dev",
            # Fix cpuinfo max() issue on Windows with Clang/GCC
            "-DCMAKE_C_FLAGS=-Dmax(a,b)=((a)>(b)?(a):(b))",
            "-DCMAKE_CXX_FLAGS=-Dmax(a,b)=((a)>(b)?(a):(b))"
        )
        
        if (Get-Command "cl" -ErrorAction SilentlyContinue) {
            $cmakeArgs += @("-G", "Visual Studio 17 2022", "-A", "x64")
        } else {
            $cmakeArgs += @("-G", "Ninja")
            # Add compiler flags to define max() macro for Windows compatibility
            $cmakeArgs += @("-DCMAKE_C_FLAGS=-Dmax(a,b)=((a)>(b)?(a):(b))")
        }
        
        Write-Host "Running: cmake $($cmakeArgs -join ' ')"
        & cmake $cmakeArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: CMake configuration failed. You may need to:"
            Write-Host "1. Install CMake (scoop install cmake)"
            Write-Host "2. Check TensorFlow Lite build documentation"
            exit 1
        }
    }
    
    # Build
    $cores = $env:NUMBER_OF_PROCESSORS
    Write-Host "Building TensorFlow Lite (this may take 10-30 minutes)..."
    Write-Host "Using $cores parallel jobs"
    
    # Check which generator was used
    $usingNinja = Test-Path "build.ninja"
    
    if ($usingNinja) {
        $buildArgs = @(
            "--build", ".",
            "--target", "tensorflow-lite",
            "-j", $cores
        )
    } else {
        $buildArgs = @(
            "--build", ".",
            "--config", "Release",
            "--target", "tensorflow-lite",
            "-j", $cores
        )
    }
    
    & cmake $buildArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Build failed"
        exit 1
    }
    
    # Find and verify the library
    $library = $null
    $possiblePaths = @(
        "libtensorflow-lite.lib",
        "Release\libtensorflow-lite.lib",
        "tensorflow\lite\libtensorflow-lite.lib",
        "tensorflow\lite\Release\libtensorflow-lite.lib"
    )
    
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $library = $path
            break
        }
    }
    
    if (-not $library) {
        Write-Host "Warning: libtensorflow-lite.lib not found in expected location"
        Write-Host "Searching for it..."
        $found = Get-ChildItem -Recurse -Filter "libtensorflow-lite.lib" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) {
            $library = $found.FullName
        } else {
            Write-Host "Error: Could not find libtensorflow-lite.lib"
            exit 1
        }
    }
    
    $libraryPath = if ([System.IO.Path]::IsPathRooted($library)) {
        $library
    } else {
        Join-Path $PWD $library
    }
    
    Write-Host ""
    Write-Host "Success! libtensorflow-lite.lib built at: $libraryPath"
    Write-Host ""
    Write-Host "SCons will automatically find and link this library when building Godot."
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "1. Build Godot with: scons platform=windows target=template_debug dev_build=yes tests=yes"
    Write-Host "2. The module should now link successfully"
    
} finally {
    Pop-Location
}

