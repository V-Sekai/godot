# VK_Video Module - Hardware-Accelerated AV1 Video Processing

> **⚠️ CURRENT STATUS: DOCUMENTATION ONLY**
>
> This module currently contains **NO IMPLEMENTATION** - only comprehensive documentation and architectural planning for hardware-accelerated AV1 video processing via Vulkan Video API.

## Scope and Limitations

**Intentionally Limited to:**
- **Video Codec**: AV1 only
- **Audio Codecs**: Opus and uncompressed audio only
- **Container**: MKV/WebM only
- **Architecture**: Self-contained with embedded thirdparty dependencies

**NOT Supported:**
- Other video codecs (H.264, H.265, VP9, etc.)
- Other audio codecs (Vorbis, AAC, MP3, etc.)
- Other containers (MP4, AVI, etc.)
- **Software fallback decoding** (hardware-only approach)
- Raw .av1 files
- CPU-based AV1 decoding (any form)

## Current Module Contents

```
modules/vk_video/
├── docs/                           # Comprehensive documentation
│   ├── 00_Current_State_and_Limitations.md  # Implementation status
│   ├── 00_VideoStreamAV1.md        # Main architectural document
│   ├── 01_Architecture_Overview.md
│   ├── 02_RenderingDevice_Extensions.md
│   └── ...                         # Additional planning docs
├── thirdparty/                     # Self-contained dependencies
│   ├── libsimplewebm/              # MKV/WebM container parsing (planned)
│   ├── libopus/                    # Opus audio decoding (planned)
│   └── vk_video_samples/           # NVIDIA Vulkan Video samples
├── LICENSE
└── README.md                       # This file
```

**Missing (Needs Implementation):**
- ❌ `SCsub` (build configuration)
- ❌ `config.py` (module configuration)
- ❌ `register_types.cpp/.h` (Godot registration)
- ❌ `video_stream_av1.cpp/.h` (main classes)
- ❌ `thirdparty/libsimplewebm/` (container parsing library)
- ❌ `thirdparty/libopus/` (audio decoding library)
- ❌ Any functional C++ code

## Self-Contained Architecture (No External Dependencies)

The vk_video module is **completely self-contained** with all dependencies in `thirdparty/`:

```
┌──────────────────────────────────────────┐
│           modules/vk_video               │
│                                          │
│ ┌─────────────────┐ ┌─────────────────┐  │
│ │   thirdparty/   │ │  Core Module    │  │
│ │                 │ │                 │  │
│ │ ✅ libsimplewebm│ │ ❌ AV1 Decoding │  │
│ │    (container   │ │    (planned)    │  │
│ │     parsing)    │ │                 │  │
│ │ ✅ libopus      │ │ ❌ Vulkan Video │  │
│ │    (audio       │ │    (planned)    │  │
│ │     decoding)   │ │                 │  │
│ │                 │ │ ❌ GPU textures │  │
│ │                 │ │    (planned)    │  │
│ └─────────────────┘ └─────────────────┘  │
└──────────────────────────────────────────┘
```

**Self-contained capabilities:**
- ✅ **libsimplewebm**: Lightweight MKV/WebM container parsing (no FFmpeg)
- ✅ **libopus**: Opus audio decoding
- ✅ **Uncompressed audio**: PCM/WAV audio support
- ❌ **AV1 video decoding**: Hardware-accelerated via Vulkan Video (planned)

**No external module dependencies**: The vk_video module includes all necessary third-party libraries in its own `thirdparty/` directory, making it completely independent.

## Hardware Requirements (When Implemented)

**STRICT HARDWARE-ONLY POLICY:**
- **GPU**: NVIDIA RTX 30+, AMD RX 6000+, or Intel Arc
- **Drivers**: Recent drivers with Vulkan Video support
- **Extensions**: `VK_KHR_video_queue`, `VK_KHR_video_decode_av1`, `VK_KHR_video_encode_av1`

**⚠️ NO SOFTWARE FALLBACK**: Systems without compatible hardware cannot play AV1 videos - they must use alternative formats (Theora).

## Implementation Roadmap

### Phase 1: Core Infrastructure ❌ Not Started
- [ ] Create module build system (SCsub, config.py)
- [ ] Implement Godot registration (register_types.cpp)
- [ ] Extend RenderingDevice API for Vulkan Video
- [ ] Hardware capability detection

### Phase 2: Basic AV1 Decoding ❌ Not Started
- [ ] VideoStreamAV1 and VideoStreamPlaybackAV1 classes
- [ ] Integration with embedded libsimplewebm and libopus
- [ ] Basic decode pipeline implementation

### Phase 3: Advanced Features ❌ Not Started
- [ ] Multi-buffered decode pipeline
- [ ] Audio-video synchronization
- [ ] Seeking support
- [ ] Error handling and fallbacks

### Phase 4: Encoding Support ❌ Not Started
- [ ] AV1 encoding pipeline
- [ ] Movie Maker integration
- [ ] Real-time capture capabilities

## Documentation

For detailed architectural information, see:
- [Current State and Limitations](docs/00_Current_State_and_Limitations.md)
- [Main Architecture Document](docs/00_VideoStreamAV1.md)

## Development Status

**Priority**: Complete implementation from scratch required.

**Expertise Needed**:
- Vulkan Video API knowledge
- AV1 codec specifications
- Godot engine internals
- GPU programming and synchronization

**Timeline**: Significant undertaking - months of development for a complete implementation.

## License

See [LICENSE](LICENSE) file for details.
