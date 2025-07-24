# Current State and Design Limitations for VK_Video Module

## Current Implementation Status

**As of January 2025, the vk_video module contains:**

-   ✅ Comprehensive documentation and architectural planning
-   ✅ Reference implementations (vk_video_samples thirdparty)
-   ✅ **WORKING VideoStreamMKV implementation**
-   ✅ Build system integration (SCsub, config.py, register_types.cpp)
-   ✅ VideoStreamMKV and VideoStreamPlaybackMKV classes
-   ✅ MKV/WebM container parsing (via embedded libsimplewebm)
-   ✅ Opus audio decoding (via embedded libopus)
-   ❌ **Vulkan Video AV1 decoding** (currently uses placeholder textures)

**The module provides working MKV container + Opus audio playback, but needs Vulkan Video integration for actual AV1 video decoding.**

## Design Limitation: AV1-in-MKV Only

### Scope Restriction

The vk_video module is **intentionally limited** to handle:

-   **Video Codec**: AV1 only (no H.264, H.265, VP9, etc.)
-   **Audio Codecs**: Opus and uncompressed audio only (leveraging existing mkv module capabilities)
-   **Container**: MKV/WebM containers only (no MP4, AVI, etc.)
-   **Integration**: Works exclusively with the existing `modules/mkv` for container parsing

### Rationale for Limitation

1. **Hardware-Only Approach**: The module is **strictly hardware-accelerated only**:

    - **No software fallback** for AV1 decoding - if hardware support is unavailable, playback fails gracefully
    - Avoids the severe performance penalties of CPU-based AV1 decoding
    - Maintains consistent, predictable performance characteristics
    - Users without compatible hardware must use alternative formats (e.g., Theora)

2. **Focused Implementation**: By limiting scope to AV1-in-MKV, the initial implementation can be:

    - More thoroughly tested and optimized
    - Easier to maintain and debug
    - Faster to develop and deploy

3. **Leverages Existing Infrastructure**: The `modules/mkv` already provides:

    - Container parsing and demuxing capabilities
    - Audio decoding (Opus and uncompressed audio formats)
    - File I/O and seeking infrastructure
    - Integration with Godot's resource system

4. **Modern Codec Focus**: AV1 represents the future of video compression:
    - Superior compression efficiency vs older codecs
    - Royalty-free licensing
    - Growing hardware support across GPU vendors

### Integration with modules/mkv

The design calls for tight integration between vk_video and mkv modules:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   modules/mkv   │    │  modules/vk_video │    │ VideoStreamPlayer│
│                 │    │                  │    │                 │
│ • Container     │◄──►│ • AV1 Decoding   │◄──►│ • UI Controls   │
│   parsing       │    │ • Vulkan Video   │    │ • Playback      │
│ • Audio decode  │    │ • GPU textures   │    │   management    │
│ • Metadata      │    │ • Sync with mkv  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

**Current mkv module capabilities:**

-   ✅ Parses MKV/WebM containers
-   ✅ Extracts video metadata (width, height, duration)
-   ✅ Decodes audio streams (Opus and uncompressed audio)
-   ❌ **Does NOT decode video** - provides placeholder textures only

**Planned vk_video integration:**

-   vk_video will receive compressed AV1 frames from mkv's demuxer
-   vk_video will decode frames using Vulkan Video API
-   vk_video will provide decoded textures back to VideoStreamPlayer
-   mkv will continue handling audio and synchronization

## Implementation Roadmap

### Phase 1: Core Infrastructure (✅ COMPLETED)

-   [x] Create basic module structure (SCsub, config.py, register_types.cpp)
-   [x] VideoStreamMKV and VideoStreamPlaybackMKV classes
-   [x] MKV/WebM container parsing integration
-   [x] Opus audio decoding integration
-   [ ] **REMAINING**: Vulkan Video API integration for AV1 decoding

### Phase 2: Vulkan Video AV1 Decoding (🚧 IN PROGRESS)

-   [x] Basic VideoStreamMKV framework (placeholder textures)
-   [ ] **NEXT**: Hardware capability detection for AV1 decode support
-   [ ] **NEXT**: VkVideoSession creation for AV1 decoding
-   [ ] **NEXT**: AV1 bitstream parsing and GPU submission
-   [ ] **NEXT**: Replace placeholder textures with decoded frames

### Phase 3: Advanced Features (Not Started)

-   [ ] Multi-buffered decode pipeline
-   [ ] Enhanced audio-video synchronization
-   [ ] Optimized seeking support
-   [ ] Comprehensive error handling and fallbacks

### Phase 4: Encoding Support (Not Started)

-   [ ] AV1 encoding pipeline
-   [ ] Movie Maker integration

## Comparison with Other Modules

| Module       | Container | Video Codec | Audio Codec         | Implementation Status   |
| ------------ | --------- | ----------- | ------------------- | ----------------------- |
| **theora**   | OGG       | Theora      | Vorbis              | ✅ Complete (CPU-based) |
| **mkv**      | MKV/WebM  | None        | Opus/Uncompressed   | ✅ Audio only           |
| **vk_video** | MKV/WebM  | AV1*        | Opus                | 🚧 **Partial** (Audio + Container working, AV1 placeholder) |

\*AV1 decoding currently shows placeholder textures - Vulkan Video integration needed

## Future Considerations

### Potential Expansion Beyond AV1-in-MKV

While the initial implementation is limited to AV1-in-MKV, the architectural design allows for future expansion:

1. **Additional Codecs**: The generic RenderingDevice API design could support H.264, H.265, VP9 (hardware-accelerated only)
2. **Additional Containers**: Support for MP4, AVI could be added with new demuxer modules

**Explicitly NOT Supported (Now or Future):**

-   **Software fallbacks**: No CPU-based AV1 decoding will be implemented
-   **Mixed hardware/software pipelines**: The module maintains a strict hardware-only policy

However, **these expansions are explicitly out of scope** for the initial implementation to maintain focus and deliverability.

### Hardware Requirements

The vk_video module will require:

-   **GPU**: NVIDIA RTX 30+, AMD RX 6000+, or Intel Arc
-   **Drivers**: Recent drivers with Vulkan Video support
-   **Extensions**: VK_KHR_video_queue, VK_KHR_video_decode_av1, VK_KHR_video_encode_av1

**Systems without these requirements cannot use AV1 playback** - they must use alternative video formats (Theora). No software fallback is provided.

## Development Status

**Current Priority**: The vk_video module has a solid foundation with working MKV container parsing and Opus audio decoding. The remaining work is implementing Vulkan Video API integration for AV1 hardware decoding.

**Next Steps**:

1. **Add AV1 hardware detection** - Check for VK_KHR_video_decode_av1 support
2. **Implement Vulkan Video context** - Create VkVideoSession for AV1 decoding  
3. **Add AV1 bitstream parsing** - Extract decode parameters from MKV frames
4. **Replace placeholder textures** - Connect decoded GPU frames to Godot textures

**Current State**: The module provides a working foundation for MKV+Opus playback with placeholder video. Adding Vulkan Video integration will complete the AV1 hardware decoding functionality.

**Timeline**: With the existing foundation, implementing Vulkan Video AV1 decoding is a focused effort requiring expertise in Vulkan Video API and AV1 codec specifications.
