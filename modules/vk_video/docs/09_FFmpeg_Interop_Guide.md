# FFmpeg Interop Guide - DEPRECATED

> **⚠️ DEPRECATED**: This document is **no longer applicable** to the vk_video module design.
>
> **Current Approach**: The vk_video module is **self-contained** with embedded thirdparty dependencies, **NOT FFmpeg**.
>
> **Scope**: AV1-in-MKV-with-Opus only, hardware-only decoding via Vulkan Video API.

## Why No FFmpeg?

The vk_video module intentionally avoids FFmpeg dependency by:
1. **Self-contained architecture**: All dependencies embedded in `modules/vk_video/thirdparty/`
2. **Focused scope**: AV1-only codec support eliminates need for FFmpeg's broad codec library
3. **Minimal dependencies**: Keeps the module lightweight and reduces build complexity
4. **Hardware-only approach**: No software fallback eliminates need for FFmpeg's software decoders

## Self-Contained Architecture

The vk_video module is completely independent with embedded dependencies:

```
┌──────────────────────────────────────────┐
│           modules/vk_video               │
│                                          │
│ ┌─────────────────┐ ┌─────────────────┐  │
│ │   thirdparty/   │ │  Core Module    │  │
│ │                 │ │                 │  │
│ │ ✅ libsimplewebm│ │ ✅ AV1 Decoding │  │
│ │    (container   │ │    (Vulkan)     │  │
│ │     parsing)    │ │                 │  │
│ │ ✅ libopus      │ │ ✅ GPU textures │  │
│ │    (audio       │ │    (zero-copy)  │  │
│ │     decoding)   │ │                 │  │
│ │ ✅ vk_video_    │ │ ✅ Hardware     │  │
│ │    samples      │ │    encoding     │  │
│ └─────────────────┘ └─────────────────┘  │
└──────────────────────────────────────────┘
```

**Embedded Dependencies:**
- **libsimplewebm**: Lightweight MKV/WebM container parsing (no FFmpeg)
- **libopus**: Opus audio decoding
- **vk_video_samples**: NVIDIA Vulkan Video reference implementation

**Benefits of self-contained approach:**
- No external module dependencies
- Simpler build system and configuration
- Faster compilation times
- Reduced binary size
- Focused, maintainable codebase
- Complete independence from other Godot modules

## Implementation Strategy

### Thirdparty Integration

```cpp
// Self-contained container parsing
#include "thirdparty/libsimplewebm/webm_parser.h"

class MKVDemuxer {
public:
    Error open_container(const String& file_path) {
        // Use embedded libsimplewebm
        webm_parser = new WebMParser();
        return webm_parser->open(file_path.utf8().get_data());
    }

    Error read_av1_packet(PacketData& packet) {
        return webm_parser->read_video_packet(packet);
    }

private:
    WebMParser* webm_parser = nullptr;
};
```

### Audio Processing

```cpp
// Self-contained Opus decoding
#include "thirdparty/libopus/opus.h"

class OpusAudioDecoder {
public:
    Error initialize(const OpusHeader& header) {
        opus_decoder = opus_decoder_create(
            header.sample_rate,
            header.channels,
            &error
        );
        return error == OPUS_OK ? OK : ERR_CANT_CREATE;
    }

    Error decode_frame(const PacketData& packet, AudioFrame& output) {
        int samples = opus_decode(
            opus_decoder,
            packet.data,
            packet.size,
            output.samples,
            output.max_samples,
            0
        );
        return samples > 0 ? OK : ERR_CANT_DECODE;
    }

private:
    OpusDecoder* opus_decoder = nullptr;
};
```

### Complete Pipeline

```cpp
// Self-contained AV1 video processing
class VideoStreamAV1 : public VideoStream {
public:
    Error load(const String& file_path) override {
        // Use embedded demuxer
        Error err = mkv_demuxer.open_container(file_path);
        if (err != OK) return err;

        // Initialize embedded audio decoder
        OpusHeader opus_header = mkv_demuxer.get_opus_header();
        err = opus_decoder.initialize(opus_header);
        if (err != OK) return err;

        // Initialize Vulkan Video for AV1
        AV1Header av1_header = mkv_demuxer.get_av1_header();
        return vulkan_av1_decoder.initialize(av1_header);
    }

    Ref<VideoStreamPlayback> instantiate_playback() override {
        Ref<VideoStreamPlaybackAV1> playback;
        playback.instantiate();

        // Pass embedded components
        playback->set_demuxer(&mkv_demuxer);
        playback->set_audio_decoder(&opus_decoder);
        playback->set_video_decoder(&vulkan_av1_decoder);

        return playback;
    }

private:
    MKVDemuxer mkv_demuxer;
    OpusAudioDecoder opus_decoder;
    VulkanAV1Decoder vulkan_av1_decoder;
};
```

## Build System Integration

### SCsub Configuration

```python
#!/usr/bin/env python

Import("env")
Import("env_modules")

env_vk_video = env_modules.Clone()

# Add thirdparty include paths
env_vk_video.Prepend(CPPPATH=[
    "thirdparty/libsimplewebm/include",
    "thirdparty/libopus/include",
    "thirdparty/vk_video_samples/include"
])

# Build thirdparty libraries
SConscript("thirdparty/SCsub")

# Build main module
env_vk_video.add_source_files(env.modules_sources, "*.cpp")
```

### Thirdparty SCsub

```python
#!/usr/bin/env python

Import("env")
Import("env_modules")

env_thirdparty = env_modules.Clone()

# Build libsimplewebm
env_thirdparty.add_source_files(env.modules_sources, [
    "libsimplewebm/src/webm_parser.cpp",
    "libsimplewebm/src/mkv_reader.cpp"
])

# Build libopus
env_thirdparty.add_source_files(env.modules_sources, [
    "libopus/src/opus_decoder.c",
    "libopus/src/opus_multistream.c"
])

# Include vk_video_samples headers only
env_thirdparty.Prepend(CPPPATH=["vk_video_samples/include"])
```

## Advantages Over FFmpeg Integration

### 1. **Simplified Dependencies**
- No complex FFmpeg build system
- No licensing concerns with FFmpeg's GPL components
- Reduced attack surface from fewer external libraries

### 2. **Focused Functionality**
- Only includes exactly what's needed for AV1-in-MKV-with-Opus
- No unused codec support bloating the binary
- Optimized for the specific use case

### 3. **Better Integration**
- Direct control over all components
- Consistent error handling across the pipeline
- Unified memory management strategy

### 4. **Maintenance Benefits**
- Fewer moving parts to maintain
- Clear ownership of all code
- Easier debugging and profiling

---

**Note**: This document is retained for historical reference but the FFmpeg integration approach has been superseded by the self-contained architecture described above.
