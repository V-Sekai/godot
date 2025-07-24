# VK_Video Module - Hardware-Accelerated AV1 Video Processing

> **⚠️ CURRENT STATUS: DOCUMENTATION ONLY**
> 
> This module currently contains **NO IMPLEMENTATION** - only comprehensive documentation and architectural planning for hardware-accelerated AV1 video processing via Vulkan Video API.

## Scope and Limitations

**Intentionally Limited to:**
- **Video Codec**: AV1 only
- **Audio Codecs**: Opus and uncompressed audio only
- **Container**: MKV/WebM only  
- **Integration**: Works with existing `modules/mkv` for container parsing

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
├── thirdparty/                     # Reference implementations
│   └── vk_video_samples/           # NVIDIA Vulkan Video samples
├── LICENSE
└── README.md                       # This file
```

**Missing (Needs Implementation):**
- ❌ `SCsub` (build configuration)
- ❌ `config.py` (module configuration)
- ❌ `register_types.cpp/.h` (Godot registration)
- ❌ `video_stream_av1.cpp/.h` (main classes)
- ❌ Any functional C++ code

## Integration with modules/mkv (No FFmpeg Required)

The design leverages the existing MKV module's **libsimplewebm-based** container parsing:

```
┌─────────────────┐    ┌──────────────────┐
│   modules/mkv   │    │  modules/vk_video │
│                 │    │                  │
│ ✅ Container    │◄──►│ ❌ AV1 Decoding  │
│    parsing      │    │    (planned)     │
│    (libsimple   │    │ ❌ Vulkan Video  │
│     webm)       │    │    (planned)     │
│ ✅ Audio decode │    │ ❌ GPU textures  │
│ ✅ Metadata     │    │    (planned)     │
│ ❌ Video decode │    │                  │
│    (placeholder)│    │                  │
└─────────────────┘    └──────────────────┘
```

**Current mkv capabilities (FFmpeg-free):**
- ✅ Parses MKV/WebM containers via **libsimplewebm** (lightweight, no FFmpeg)
- ✅ Extracts video metadata (width, height, duration)  
- ✅ Decodes audio streams (Opus and uncompressed audio)
- ❌ **Does NOT decode video** - provides black placeholder textures

**No FFmpeg dependency**: The vk_video module will use the existing mkv module's container parsing, which is based on libsimplewebm rather than FFmpeg, keeping dependencies minimal.

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
- [ ] Integration with modules/mkv demuxer
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
