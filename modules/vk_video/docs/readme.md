# Current State and Design Limitations for VK_Video Module

## Current Implementation Status

**As of January 2025, the vk_video module contains:**
- ✅ Comprehensive documentation and architectural planning
- ✅ Reference implementations (vk_video_samples thirdparty)
- ❌ **NO ACTUAL GODOT IMPLEMENTATION CODE**
- ❌ No build system integration (SCsub, config.py, register_types.cpp)
- ❌ No VideoStreamAV1 or VideoStreamPlaybackAV1 classes

**The module currently exists as a planning and documentation repository only.**

## Design Limitation: AV1-in-MKV Only

### Scope Restriction

The vk_video module is **intentionally limited** to handle:
- **Video Codec**: AV1 only (no H.264, H.265, VP9, etc.)
- **Audio Codecs**: Opus and uncompressed audio only (leveraging existing mkv module capabilities)
- **Container**: MKV/WebM containers only (no MP4, AVI, etc.)
- **Integration**: Works exclusively with the existing `modules/mkv` for container parsing

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
- ✅ Parses MKV/WebM containers
- ✅ Extracts video metadata (width, height, duration)
- ✅ Decodes audio streams (Opus and uncompressed audio)
- ❌ **Does NOT decode video** - provides placeholder textures only

**Planned vk_video integration:**
- vk_video will receive compressed AV1 frames from mkv's demuxer
- vk_video will decode frames using Vulkan Video API
- vk_video will provide decoded textures back to VideoStreamPlayer
- mkv will continue handling audio and synchronization

## Implementation Roadmap

### Phase 1: Core Infrastructure (Not Started)
- [ ] Create basic module structure (SCsub, config.py, register_types.cpp)
- [ ] Extend RenderingDevice API for Vulkan Video operations
- [ ] Implement hardware capability detection

### Phase 2: Basic AV1 Decoding (Not Started)
- [ ] Create VideoStreamAV1 and VideoStreamPlaybackAV1 classes
- [ ] Integrate with modules/mkv for container parsing
- [ ] Implement basic decode pipeline

### Phase 3: Advanced Features (Not Started)
- [ ] Multi-buffered decode pipeline
- [ ] Audio-video synchronization
- [ ] Seeking support
- [ ] Error handling and fallbacks

### Phase 4: Encoding Support (Not Started)
- [ ] AV1 encoding pipeline
- [ ] Movie Maker integration
- [ ] Real-time capture capabilities

## Comparison with Other Modules

| Module | Container | Video Codec | Audio Codec | Implementation Status |
|--------|-----------|-------------|-------------|----------------------|
| **theora** | OGG | Theora | Vorbis | ✅ Complete (CPU-based) |
| **mkv** | MKV/WebM | None | Opus/Uncompressed | ✅ Audio only |
| **vk_video** | MKV/WebM | AV1 | Opus/Uncompressed* | ❌ **Not implemented** |

*Audio handled by mkv module

## Future Considerations

### Potential Expansion Beyond AV1-in-MKV

While the initial implementation is limited to AV1-in-MKV, the architectural design allows for future expansion:

1. **Additional Codecs**: The generic RenderingDevice API design could support H.264, H.265, VP9 (hardware-accelerated only)
2. **Additional Containers**: Support for MP4, AVI could be added with new demuxer modules

**Explicitly NOT Supported (Now or Future):**
- **Software fallbacks**: No CPU-based AV1 decoding will be implemented
- **Mixed hardware/software pipelines**: The module maintains a strict hardware-only policy

However, **these expansions are explicitly out of scope** for the initial implementation to maintain focus and deliverability.

### Hardware Requirements

The vk_video module will require:
- **GPU**: NVIDIA RTX 30+, AMD RX 6000+, or Intel Arc
- **Drivers**: Recent drivers with Vulkan Video support
- **Extensions**: VK_KHR_video_queue, VK_KHR_video_decode_av1, VK_KHR_video_encode_av1

**Systems without these requirements cannot use AV1 playback** - they must use alternative video formats (Theora). No software fallback is provided.

## Development Status

**Current Priority**: The vk_video module requires complete implementation from scratch. The existing documentation provides excellent architectural guidance, but no functional code exists.

**Next Steps**: 
1. Create basic module infrastructure
2. Implement RenderingDevice extensions
3. Begin basic AV1 decode pipeline development

**Timeline**: Implementation is a significant undertaking requiring expertise in Vulkan Video API, AV1 codec specifications, and Godot engine internals.
