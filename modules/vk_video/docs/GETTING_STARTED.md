# Getting Started with VK Video Module

**Quick Start Guide for Current Implementation**

## What Works Right Now

The VK Video module currently provides:

✅ **MKV/WebM Container Support**: Full parsing and metadata extraction  
✅ **Opus Audio Playback**: High-quality audio decoding  
✅ **OneEuroFilter**: Production-ready signal smoothing  
✅ **Test Infrastructure**: Comprehensive TDD framework  

❌ **AV1 Video Decoding**: Shows placeholder textures only  

## Building the Module

### Prerequisites
- Godot 4.x source code
- C++ compiler (GCC, Clang, or MSVC)
- SCons build system

### Build Commands
```bash
# Clone Godot with VK Video module
git clone https://github.com/V-Sekai/godot.git
cd godot

# Build with VK Video module enabled
scons platform=linuxbsd target=editor

# For other platforms
scons platform=windows target=editor
scons platform=macos target=editor
```

### Verify Build Success
```bash
# Run OneEuroFilter tests to verify module loaded correctly
./bin/godot --test --test-case="*OneEuroFilter*"

# Expected output:
# [doctest] test cases: 6 | 6 passed | 0 failed
# [doctest] assertions: 8 | 8 passed | 0 failed
```

## Current Usage

### 1. Audio-Only MKV Playback

**What works**: You can play MKV/WebM files with Opus audio. Video shows placeholder frames.

```gdscript
# In your scene
extends Control

@onready var video_player = $VideoStreamPlayer

func _ready():
    # Load MKV file with Opus audio
    var video_stream = VideoStreamMKV.new()
    video_stream.file = "res://path/to/your/video.mkv"
    
    video_player.stream = video_stream
    video_player.play()
    
    # Audio will play correctly
    # Video will show placeholder texture
```

### 2. OneEuroFilter for Signal Smoothing

**What works**: Production-ready signal filtering for any noisy data.

```gdscript
# Create filter for smoothing noisy signals
var filter = OneEuroFilter.new()
filter.set_min_cutoff(0.1)  # Baseline smoothing
filter.set_beta(5.0)        # Responsiveness

func _process(delta):
    var noisy_value = get_some_noisy_signal()
    var smooth_value = filter.filter(noisy_value, delta)
    use_smooth_value(smooth_value)
```

### 3. Testing the Implementation

```bash
# Run all OneEuroFilter tests
./bin/godot --test --test-case="*OneEuroFilter*"

# Run with detailed output
./bin/godot --test --test-case="*OneEuroFilter*" --success

# Test specific functionality
./bin/godot --test --test-case="*OneEuroFilter*Constructor*"
```

## File Format Support

### ✅ **Supported Formats**
- **Container**: MKV (.mkv), WebM (.webm)
- **Audio**: Opus, Uncompressed PCM
- **Video**: None (placeholder textures only)

### ❌ **Unsupported Formats**
- **Containers**: MP4, AVI, MOV, etc.
- **Audio**: MP3, AAC, Vorbis, etc.
- **Video**: AV1, H.264, H.265, VP9, etc.

### Creating Compatible Files

Use FFmpeg to create compatible MKV files:

```bash
# Convert any video to MKV with Opus audio (video will be placeholder)
ffmpeg -i input.mp4 -c:a libopus -c:v copy output.mkv

# Create audio-only MKV
ffmpeg -i input.mp4 -c:a libopus -vn output.mkv
```

## Troubleshooting

### Build Issues

**Problem**: Module not found during build
```bash
# Solution: Verify module directory structure
ls modules/vk_video/
# Should show: SCsub, config.py, register_types.cpp, etc.
```

**Problem**: Compilation errors
```bash
# Solution: Clean build and retry
scons --clean
scons platform=linuxbsd target=editor
```

### Runtime Issues

**Problem**: VideoStreamMKV class not found
```gdscript
# Check if module loaded correctly
if ClassDB.class_exists("VideoStreamMKV"):
    print("VK Video module loaded successfully")
else:
    print("VK Video module not loaded")
```

**Problem**: MKV file won't load
```gdscript
# Verify file format
var file = FileAccess.open("res://video.mkv", FileAccess.READ)
if file:
    print("File exists and is readable")
    file.close()
else:
    print("File not found or not readable")
```

### Audio Playback Issues

**Problem**: No audio from MKV file
- Verify the MKV contains Opus audio track
- Check Godot's audio settings and volume levels
- Ensure the file path is correct

**Problem**: Audio stuttering
- Check system audio latency settings
- Verify sufficient CPU resources available
- Try different Opus bitrate settings

## Current Limitations

### Video Decoding
- **No AV1 hardware decoding**: Vulkan Video integration not implemented
- **Placeholder textures only**: Video track is parsed but not decoded
- **No seeking support**: Video position controls don't work

### Container Support
- **MKV/WebM only**: No support for MP4, AVI, or other containers
- **Limited codec support**: Only Opus audio, no video codecs

### Platform Support
- **Vulkan required**: Module designed for Vulkan-capable systems
- **No software fallback**: Hardware-only approach

## Development and Testing

### Running Tests
```bash
# All VK Video tests
./bin/godot --test --test-case="*vk_video*"

# OneEuroFilter specific tests
./bin/godot --test --test-case="*OneEuroFilter*"

# Verbose test output
./bin/godot --test --test-case="*OneEuroFilter*" --success
```

### Contributing
1. **Follow TDD approach**: Write tests before implementation
2. **Update documentation**: Keep docs in sync with changes
3. **Test on multiple platforms**: Verify cross-platform compatibility

### Next Steps for Development
1. **Add VideoStreamMKV tests**: Unit tests for container parsing
2. **Implement AVSynchronizer**: Integration layer for OneEuroFilter
3. **Add hardware detection**: Check for Vulkan Video support
4. **Implement AV1 decoding**: Replace placeholder textures

## Getting Help

### Documentation
- **Implementation Status**: `docs/IMPLEMENTATION_STATUS.md`
- **Architecture Overview**: `docs/00_VideoStreamAV1.md`
- **Audio-Video Sync**: `docs/06_Audio_Video_Sync.md`

### Code Examples
- **OneEuroFilter Tests**: `tests/test_one_euro_filter.h`
- **Module Registration**: `register_types.cpp`
- **Build Configuration**: `SCsub`, `config.py`

### Community
- **GitHub Issues**: Report bugs and feature requests
- **Godot Discord**: #engine-development channel
- **Godot Contributors Chat**: For development discussions

---

**Remember**: This module is currently in development. The audio playback and OneEuroFilter components are production-ready, but AV1 video decoding is not yet implemented. Use this guide to explore the current capabilities while development continues.
