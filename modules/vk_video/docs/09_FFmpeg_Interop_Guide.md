# FFmpeg Interop Guide

## Overview
Comprehensive guide for FFmpeg integration with the vk_video module, covering container demuxing, YCbCr processing, format conversion, and fallback strategies for alpha-enabled content.

## FFmpeg Integration Architecture

### Core Integration Components

```
┌─────────────────────────────────────────────────────────────┐
│  Godot vk_video Module                                     │
├─────────────────────────────────────────────────────────────┤
│  FFmpeg Integration Layer                                   │
│  ├── Container Demuxing (libavformat)                      │
│  ├── YCbCr Processing Pipeline                             │
│  ├── Format Detection & Conversion                         │
│  └── Hardware/Software Decision Engine                     │
├─────────────────────────────────────────────────────────────┤
│  VulkanFilterYuvCompute                                     │
│  ├── Hardware YCbCr Processing                             │
│  ├── Multi-plane Image Handling                            │
│  ├── Bit Depth Conversion (8/10/12/16-bit)                │
│  └── Chroma Subsampling Support                            │
├─────────────────────────────────────────────────────────────┤
│  Vulkan Video Hardware Path                                │
│  ├── H.264/H.265/AV1 Decode/Encode                         │
│  ├── GPU-accelerated YCbCr conversion                      │
│  └── Zero-copy texture operations                          │
├─────────────────────────────────────────────────────────────┤
│  FFmpeg Software Fallback Path                             │
│  ├── Software decode for unsupported formats               │
│  ├── Alpha channel processing (YUVA420P, etc.)             │
│  ├── Legacy codec support                                  │
│  └── CPU-based format conversion                           │
└─────────────────────────────────────────────────────────────┘
```

## Supported Container Formats

### Primary Container Support
| Container | Extension | Demuxing | Muxing | Hardware Path | Notes |
|-----------|-----------|----------|--------|---------------|-------|
| MP4       | .mp4      | ✅       | ✅     | ✅            | Primary format |
| MOV       | .mov      | ✅       | ✅     | ✅            | QuickTime format |
| MKV       | .mkv      | ✅       | ✅     | ✅            | Matroska container |
| WebM      | .webm     | ✅       | ✅     | ✅            | Web-optimized |
| AVI       | .avi      | ✅       | ❌     | ⚠️            | Legacy support |

### FFmpeg Integration Code
```cpp
// Container demuxing with FFmpeg
class FFmpegDemuxer {
public:
    Error open_container(const String& file_path) {
        // Initialize FFmpeg context
        if (avformat_open_input(&format_ctx, file_path.utf8().get_data(), nullptr, nullptr) < 0) {
            return ERR_FILE_CANT_OPEN;
        }
        
        // Find stream information
        if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
            return ERR_FILE_CORRUPT;
        }
        
        // Locate video and audio streams
        video_stream_index = find_best_stream(AVMEDIA_TYPE_VIDEO);
        audio_stream_index = find_best_stream(AVMEDIA_TYPE_AUDIO);
        
        return OK;
    }
    
    Error read_packet(AVPacket* packet) {
        return av_read_frame(format_ctx, packet) >= 0 ? OK : ERR_FILE_EOF;
    }
};
```

## YCbCr Format Processing

### Supported YCbCr Formats

#### Hardware-Accelerated Formats
```cpp
// VulkanFilterYuvCompute supported formats
enum class YCbCrFormat {
    // 8-bit formats
    NV12,           // 2-plane, 4:2:0, 8-bit
    NV21,           // 2-plane, 4:2:0, 8-bit (swapped UV)
    I420,           // 3-plane, 4:2:0, 8-bit
    YV12,           // 3-plane, 4:2:0, 8-bit (swapped UV)
    
    // 10-bit formats
    P010,           // 2-plane, 4:2:0, 10-bit MSB
    P210,           // 2-plane, 4:2:2, 10-bit MSB
    
    // 12-bit formats
    P012,           // 2-plane, 4:2:0, 12-bit MSB
    P212,           // 2-plane, 4:2:2, 12-bit MSB
    
    // 16-bit formats
    P016,           // 2-plane, 4:2:0, 16-bit
    P216,           // 2-plane, 4:2:2, 16-bit
    
    // RGBA (for compatibility)
    RGBA8,          // 1-plane, 4:4:4, 8-bit with alpha
};
```

#### Format Conversion Pipeline
```cpp
// YCbCr format conversion with VulkanFilterYuvCompute
class YCbCrProcessor {
public:
    Error convert_format(const YCbCrFrame& input, YCbCrFormat target_format, YCbCrFrame& output) {
        // Determine conversion path
        if (requires_bit_depth_conversion(input.format, target_format)) {
            return convert_bit_depth(input, target_format, output);
        }
        
        if (requires_plane_conversion(input.format, target_format)) {
            return convert_plane_layout(input, target_format, output);
        }
        
        // Direct copy if formats are compatible
        return copy_frame(input, output);
    }
    
private:
    Error convert_bit_depth(const YCbCrFrame& input, YCbCrFormat target_format, YCbCrFrame& output) {
        // Use VulkanFilterYuvCompute for hardware conversion
        VkSharedBaseObj<VulkanFilterYuvCompute> filter;
        VulkanFilterYuvCompute::Create(
            vk_dev_ctx,
            queue_family_index,
            queue_index,
            VulkanFilterYuvCompute::YCBCRCOPY,
            max_frames,
            get_vulkan_format(input.format),
            get_vulkan_format(target_format),
            input.enable_msb_to_lsb_shift,
            output.enable_lsb_to_msb_shift,
            &ycbcr_conversion_info,
            &ycbcr_primaries_constants,
            &sampler_info,
            filter
        );
        
        return filter->process_frame(input, output);
    }
};
```

### Chroma Subsampling Support

#### Subsampling Formats
| Format | Horizontal | Vertical | Description | Hardware Support |
|--------|------------|----------|-------------|------------------|
| 4:4:4  | 1:1        | 1:1      | No subsampling | ✅ |
| 4:2:2  | 2:1        | 1:1      | Horizontal subsampling | ✅ |
| 4:2:0  | 2:1        | 2:1      | Both directions | ✅ |
| 4:1:1  | 4:1        | 1:1      | Heavy horizontal | ⚠️ |

#### Subsampling Conversion Code
```cpp
// Chroma subsampling handling in shaders
void GenHandleChromaPosition(std::stringstream& shaderStr,
                            uint32_t chromaHorzRatio,
                            uint32_t chromaVertRatio) {
    if (chromaHorzRatio <= 1 && chromaVertRatio <= 1) {
        // 4:4:4 - no subsampling
        shaderStr << "    bool processChroma = true;\n";
        return;
    }
    
    // Generate condition for subsampled formats
    shaderStr << "    bool processChroma = ";
    if (chromaHorzRatio > 1) {
        shaderStr << "(pos.x % " << chromaHorzRatio << " == 0)";
    }
    if (chromaHorzRatio > 1 && chromaVertRatio > 1) {
        shaderStr << " && ";
    }
    if (chromaVertRatio > 1) {
        shaderStr << "(pos.y % " << chromaVertRatio << " == 0)";
    }
    shaderStr << ";\n";
}
```

## Alpha Channel Handling

### Current Alpha Support Status

#### ❌ Hardware Limitations
```cpp
// Current VulkanFilterYuvCompute limitations
class VulkanFilterYuvCompute {
    // Limited to 3-plane maximum (Y, Cb, Cr)
    static constexpr uint32_t MAX_PLANES = 3;
    
    // Alpha hardcoded in YCbCr to RGBA conversion
    void InitYCBCR2RGBA() {
        shaderStr << "vec4 rgba = vec4(rgb, 1.0);\n";  // Alpha = 1.0 (opaque)
    }
    
    // No YCbCrA format support
    VkImageAspectFlags supported_aspects = 
        VK_IMAGE_ASPECT_PLANE_0_BIT |  // Y plane
        VK_IMAGE_ASPECT_PLANE_1_BIT |  // Cb/CbCr plane  
        VK_IMAGE_ASPECT_PLANE_2_BIT;   // Cr plane (no alpha plane)
};
```

#### ✅ Software Fallback for Alpha
```cpp
// Alpha-enabled format detection and fallback
class AlphaFormatHandler {
public:
    bool requires_alpha_fallback(AVPixelFormat pix_fmt) {
        switch (pix_fmt) {
            case AV_PIX_FMT_YUVA420P:   // YUV 4:2:0 with alpha
            case AV_PIX_FMT_YUVA422P:   // YUV 4:2:2 with alpha
            case AV_PIX_FMT_YUVA444P:   // YUV 4:4:4 with alpha
            case AV_PIX_FMT_RGBA:       // RGB with alpha
            case AV_PIX_FMT_BGRA:       // BGR with alpha
                return true;
            default:
                return false;
        }
    }
    
    Error process_alpha_content(const AVFrame* frame, RID& output_texture) {
        // Use FFmpeg software conversion for alpha content
        SwsContext* sws_ctx = sws_getContext(
            frame->width, frame->height, (AVPixelFormat)frame->format,
            frame->width, frame->height, AV_PIX_FMT_RGBA,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
        
        if (!sws_ctx) {
            return ERR_CANT_CREATE;
        }
        
        // Convert to RGBA with preserved alpha
        AVFrame* rgba_frame = av_frame_alloc();
        rgba_frame->format = AV_PIX_FMT_RGBA;
        rgba_frame->width = frame->width;
        rgba_frame->height = frame->height;
        
        if (av_frame_get_buffer(rgba_frame, 32) < 0) {
            av_frame_free(&rgba_frame);
            sws_freeContext(sws_ctx);
            return ERR_OUT_OF_MEMORY;
        }
        
        sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
                  rgba_frame->data, rgba_frame->linesize);
        
        // Upload to GPU texture
        output_texture = upload_rgba_to_texture(rgba_frame);
        
        av_frame_free(&rgba_frame);
        sws_freeContext(sws_ctx);
        return OK;
    }
};
```

### Alpha Format Support Matrix
| Format | Planes | Alpha Support | Hardware Path | Software Path | Notes |
|--------|--------|---------------|---------------|---------------|-------|
| YUVA420P | 4 | ✅ | ❌ | ✅ | Requires FFmpeg fallback |
| YUVA422P | 4 | ✅ | ❌ | ✅ | Requires FFmpeg fallback |
| YUVA444P | 4 | ✅ | ❌ | ✅ | Requires FFmpeg fallback |
| RGBA | 1 | ✅ | ✅ | ✅ | Native support |
| BGRA | 1 | ✅ | ✅ | ✅ | Native support |
| NV12 | 2 | ❌ | ✅ | ✅ | No alpha channel |
| I420 | 3 | ❌ | ✅ | ✅ | No alpha channel |

## Hardware/Software Decision Engine

### Decision Matrix
```cpp
// Automatic path selection based on format and capabilities
class VideoPathSelector {
public:
    enum class ProcessingPath {
        VULKAN_HARDWARE,    // Full hardware acceleration
        VULKAN_HYBRID,      // Hardware decode + software conversion
        FFMPEG_SOFTWARE     // Full software processing
    };
    
    ProcessingPath select_path(const VideoStreamInfo& stream_info) {
        // Check hardware capabilities
        if (!has_vulkan_video_support()) {
            return ProcessingPath::FFMPEG_SOFTWARE;
        }
        
        // Check codec support
        if (!is_codec_supported_hardware(stream_info.codec)) {
            return ProcessingPath::FFMPEG_SOFTWARE;
        }
        
        // Check format support
        if (requires_alpha_fallback(stream_info.pixel_format)) {
            return ProcessingPath::FFMPEG_SOFTWARE;
        }
        
        // Check performance thresholds
        if (stream_info.width * stream_info.height > get_hardware_threshold()) {
            return ProcessingPath::VULKAN_HYBRID;
        }
        
        return ProcessingPath::VULKAN_HARDWARE;
    }
    
private:
    bool has_vulkan_video_support() {
        RenderingDevice* rd = RenderingDevice::get_singleton();
        return rd && rd->has_feature(RenderingDevice::FEATURE_VULKAN_VIDEO);
    }
    
    bool is_codec_supported_hardware(VideoCodec codec) {
        switch (codec) {
            case VideoCodec::H264:
                return check_vulkan_extension("VK_KHR_video_decode_h264");
            case VideoCodec::H265:
                return check_vulkan_extension("VK_KHR_video_decode_h265");
            case VideoCodec::AV1:
                return check_vulkan_extension("VK_KHR_video_decode_av1");
            default:
                return false;
        }
    }
};
```

### Performance Monitoring
```cpp
// Performance-based path switching
class PerformanceMonitor {
public:
    void update_performance_metrics(ProcessingPath path, double frame_time) {
        performance_history[path].push_back(frame_time);
        
        // Keep only recent samples
        if (performance_history[path].size() > MAX_SAMPLES) {
            performance_history[path].pop_front();
        }
        
        // Check if we should switch paths
        if (should_switch_path(path)) {
            recommend_path_switch(path);
        }
    }
    
private:
    bool should_switch_path(ProcessingPath current_path) {
        double avg_time = calculate_average_time(current_path);
        
        // Switch to software if hardware is too slow
        if (current_path == ProcessingPath::VULKAN_HARDWARE && avg_time > HARDWARE_THRESHOLD) {
            return true;
        }
        
        // Switch back to hardware if software was temporary
        if (current_path == ProcessingPath::FFMPEG_SOFTWARE && avg_time < SOFTWARE_THRESHOLD) {
            return true;
        }
        
        return false;
    }
    
    static constexpr double HARDWARE_THRESHOLD = 16.67; // 60 FPS target
    static constexpr double SOFTWARE_THRESHOLD = 33.33; // 30 FPS minimum
    static constexpr size_t MAX_SAMPLES = 60;
    
    std::map<ProcessingPath, std::deque<double>> performance_history;
};
```

## Integration Examples

### Basic Video Playback with FFmpeg
```cpp
// Complete video playback pipeline
class VideoPlayer {
public:
    Error load_video(const String& file_path) {
        // Open container with FFmpeg
        Error err = demuxer.open_container(file_path);
        if (err != OK) return err;
        
        // Analyze stream format
        VideoStreamInfo stream_info = demuxer.get_video_stream_info();
        
        // Select processing path
        ProcessingPath path = path_selector.select_path(stream_info);
        
        // Initialize appropriate decoder
        switch (path) {
            case ProcessingPath::VULKAN_HARDWARE:
                return initialize_vulkan_decoder(stream_info);
            case ProcessingPath::VULKAN_HYBRID:
                return initialize_hybrid_decoder(stream_info);
            case ProcessingPath::FFMPEG_SOFTWARE:
                return initialize_software_decoder(stream_info);
        }
        
        return ERR_CANT_CREATE;
    }
    
    Error decode_frame(RID& output_texture) {
        AVPacket packet;
        Error err = demuxer.read_packet(&packet);
        if (err != OK) return err;
        
        switch (current_path) {
            case ProcessingPath::VULKAN_HARDWARE:
                return vulkan_decoder.decode_frame(&packet, output_texture);
            case ProcessingPath::FFMPEG_SOFTWARE:
                return software_decoder.decode_frame(&packet, output_texture);
            default:
                return ERR_UNAVAILABLE;
        }
    }
    
private:
    FFmpegDemuxer demuxer;
    VideoPathSelector path_selector;
    ProcessingPath current_path;
    
    VulkanVideoDecoder vulkan_decoder;
    FFmpegSoftwareDecoder software_decoder;
};
```

### Movie Maker Encoding with FFmpeg
```cpp
// Hardware encoding with FFmpeg muxing
class MovieMakerFFmpeg {
public:
    Error initialize_encoding(const String& output_path, const EncodingSettings& settings) {
        // Initialize FFmpeg muxer
        avformat_alloc_output_context2(&format_ctx, nullptr, nullptr, output_path.utf8().get_data());
        if (!format_ctx) return ERR_CANT_CREATE;
        
        // Add video stream
        video_stream = avformat_new_stream(format_ctx, nullptr);
        if (!video_stream) return ERR_CANT_CREATE;
        
        // Configure codec parameters
        video_stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
        video_stream->codecpar->codec_id = get_codec_id(settings.codec);
        video_stream->codecpar->width = settings.width;
        video_stream->codecpar->height = settings.height;
        video_stream->codecpar->format = get_pixel_format(settings.format);
        
        // Initialize Vulkan encoder
        return vulkan_encoder.initialize(settings);
    }
    
    Error encode_frame(RID source_texture) {
        // Convert texture to YCbCr if needed
        YCbCrFrame ycbcr_frame;
        Error err = convert_texture_to_ycbcr(source_texture, ycbcr_frame);
        if (err != OK) return err;
        
        // Encode with Vulkan
        BitstreamBuffer encoded_data;
        err = vulkan_encoder.encode_frame(ycbcr_frame, encoded_data);
        if (err != OK) return err;
        
        // Package into AVPacket
        AVPacket packet;
        av_init_packet(&packet);
        packet.data = encoded_data.data;
        packet.size = encoded_data.size;
        packet.stream_index = video_stream->index;
        
        // Write to container
        return av_interleaved_write_frame(format_ctx, &packet) >= 0 ? OK : ERR_FILE_CANT_WRITE;
    }
    
private:
    AVFormatContext* format_ctx = nullptr;
    AVStream* video_stream = nullptr;
    VulkanVideoEncoder vulkan_encoder;
};
```

## Best Practices

### Format Selection Guidelines
1. **Use hardware path for standard formats**: NV12, I420, P010 for best performance
2. **Fallback to software for alpha content**: YUVA420P, RGBA with alpha
3. **Monitor performance**: Switch paths based on real-time metrics
4. **Validate format support**: Check hardware capabilities before initialization

### Error Handling
```cpp
// Robust error handling with fallback
Error try_hardware_decode_with_fallback(const VideoStreamInfo& info, RID& output) {
    // Try hardware path first
    Error err = try_vulkan_decode(info, output);
    if (err == OK) {
        return OK;
    }
    
    // Log hardware failure and try software
    print_line("Hardware decode failed, falling back to software: " + error_string(err));
    return try_software_decode(info, output);
}
```

### Memory Management
```cpp
// Efficient memory handling for both paths
class VideoMemoryManager {
public:
    void configure_for_path(ProcessingPath path) {
        switch (path) {
            case ProcessingPath::VULKAN_HARDWARE:
                // Use GPU memory pools
                configure_vulkan_pools();
                break;
            case ProcessingPath::FFMPEG_SOFTWARE:
                // Use system memory with alignment
                configure_system_pools();
                break;
        }
    }
    
private:
    void configure_vulkan_pools() {
        // Pre-allocate GPU buffers for zero-copy operations
        vulkan_pool.allocate_dpb_images(MAX_REFERENCE_FRAMES);
        vulkan_pool.allocate_bitstream_buffers(MAX_BITSTREAM_SIZE);
    }
};
```

This comprehensive FFmpeg interop guide provides the foundation for robust video processing with automatic hardware/software path selection and proper alpha channel handling.
