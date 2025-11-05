@echo off
REM Helper script to build TensorFlow Lite for Windows
REM This builds TFLite separately with CMake, then SCons will find it automatically

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..
set TFLITE_SOURCE=%PROJECT_ROOT%\thirdparty\tensorflow-lite
set TFLITE_BUILD=%PROJECT_ROOT%\thirdparty\tensorflow-lite\build

echo Building TensorFlow Lite...
echo Source: %TFLITE_SOURCE%
echo Build: %TFLITE_BUILD%
echo Note: SCons will automatically find the library in the build directory
echo.

REM Check for CMake
where cmake >nul 2>&1
if errorlevel 1 (
    echo Error: CMake is not installed
    echo Install with: choco install cmake  (Chocolatey) or download from https://cmake.org/download/
    exit /b 1
)

REM Check if source exists
if not exist "%TFLITE_SOURCE%" (
    echo Error: TensorFlow Lite source not found at %TFLITE_SOURCE%
    echo Please ensure thirdparty\tensorflow-lite exists (git subrepo)
    exit /b 1
)

REM Create build directory
if not exist "%TFLITE_BUILD%" mkdir "%TFLITE_BUILD%"
cd /d "%TFLITE_BUILD%"

REM Configure CMake
echo Configuring CMake...
cmake "%TFLITE_SOURCE%\tensorflow\lite" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DBUILD_SHARED_LIBS=OFF ^
    -DTFLITE_ENABLE_GPU=OFF ^
    -DTFLITE_ENABLE_XNNPACK=OFF ^
    -A x64
if errorlevel 1 (
    echo Error: CMake configuration failed
    echo Trying alternative: building from tensorflow root...
    cd /d "%TFLITE_SOURCE%"
    if not exist build mkdir build
    cd build
    cmake .. ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DTFLITE_ENABLE_GPU=OFF ^
        -DTFLITE_ENABLE_XNNPACK=OFF ^
        -A x64
    if errorlevel 1 (
        echo Error: CMake configuration failed. You may need to:
        echo 1. Install CMake (download from https://cmake.org/download/)
        echo 2. Check TensorFlow Lite build documentation
        exit /b 1
    )
)

REM Build
echo Building TensorFlow Lite (this may take 10-30 minutes)...
cmake --build . --config Release --target tensorflow-lite -j %NUMBER_OF_PROCESSORS%
if errorlevel 1 (
    echo Error: Build failed
    exit /b 1
)

REM Find and verify the library
set LIBRARY=
if exist "libtensorflow-lite.lib" (
    set LIBRARY=libtensorflow-lite.lib
) else if exist "Release\libtensorflow-lite.lib" (
    set LIBRARY=Release\libtensorflow-lite.lib
) else if exist "tensorflow\lite\libtensorflow-lite.lib" (
    set LIBRARY=tensorflow\lite\libtensorflow-lite.lib
) else if exist "tensorflow\lite\Release\libtensorflow-lite.lib" (
    set LIBRARY=tensorflow\lite\Release\libtensorflow-lite.lib
) else (
    echo Warning: libtensorflow-lite.lib not found in expected location
    echo Searching for it...
    for /r . %%f in (libtensorflow-lite.lib) do (
        set LIBRARY=%%f
        goto :found
    )
    echo Error: Could not find libtensorflow-lite.lib
    exit /b 1
)

:found
set LIBRARY_PATH=%CD%\%LIBRARY%
echo.
echo Success! libtensorflow-lite.lib built at: %LIBRARY_PATH%
echo.
echo SCons will automatically find and link this library when building Godot.
echo.
echo Next steps:
echo 1. Build Godot with: scons platform=windows target=template_debug
echo 2. The module should now link successfully

endlocal

