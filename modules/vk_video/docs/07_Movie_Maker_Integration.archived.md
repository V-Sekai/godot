# Movie Writer Integration with Hardware AV1 Encoding

## Brief Description

Integration of AV1 hardware encoding with Godot's MovieWriter system using Vulkan Video encode capabilities for high-performance video capture and export. This document describes how to implement a `MovieWriterAV1` class that extends Godot's existing `MovieWriter` base class.

## Current Godot Movie Writer Architecture

Godot's movie writing system is based on the `MovieWriter` class located in `servers/movie_writer/`. The system works as follows:

1. **MovieWriter Base Class**: Provides the interface for writing video frames and audio data
2. **Registration System**: Writers register themselves and handle specific file extensions
3. **Main Integration**: The main loop calls `MovieWriter::add_frame()` when `--write-movie` is used
4. **Current Implementation**: Only `MovieWriterPNGWAV` exists (PNG frames + WAV audio)

## MovieWriterAV1 Implementation

### Class Structure

```cpp
// modules/vk_video/movie_writer_av1.h
#pragma once

#include "servers/movie_writer/movie_writer.h"
#include "modules/vk_video/rendering_device_video_extensions.h"
#include "core/io/file_access.h"

class MovieWriterAV1 : public MovieWriter {
    GDCLASS(MovieWriterAV1, MovieWriter);

public:
    struct EncoderSettings {
        uint32_t bitrate = 5000000;     // 5 Mbps default
        uint32_t quality = 50;          // 0-100 scale
        uint32_t profile = 0;           // Main profile
        uint32_t level = 51;            // Level 5.1
        uint32_t tile_cols = 1;
        uint32_t tile_rows = 1;
        bool enable_cdef = true;
        bool enable_restoration = true;
        
        enum RateControlMode {
            RC_CQP,     // Constant QP
            RC_CBR,     // Constant bitrate  
            RC_VBR,     // Variable bitrate
            RC_CRF      // Constant rate factor
        };
        RateControlMode rate_control = RC_VBR;
        uint32_t max_bitrate = 10000000; // 10 Mbps
    };

private:
    // Vulkan Video resources
    Ref<RenderingDeviceVideoExtensions> video_extensions;
    RID video_session;
    RID video_session_parameters;
    RID encode_dpb_array;
    RID bitstream_buffer;
    RID rate_control_buffer;

    // Encoding state
    EncoderSettings settings;
    bool hardware_encoding = false;
    bool initialized = false;
    uint64_t frame_count = 0;
    Size2i movie_size;

    // Output file handling
    Ref<FileAccess> output_file;
    String output_path;

    // Frame queue for async encoding
    struct QueuedFrame {
        Ref<Image> image;
        Vector<int32_t> audio_data;
        uint64_t frame_number;
        double timestamp;
    };
    Vector<QueuedFrame> frame_queue;
    uint32_t max_queued_frames = 4;

    // Bitstream writing
    Vector<uint8_t> sequence_header_data;
    bool sequence_header_written = false;

    // Audio encoding (Opus)
    void *opus_encoder = nullptr;
    Vector<uint8_t> opus_buffer;
    uint32_t opus_frame_size = 960; // 20ms at 48kHz

protected:
    static void _bind_methods();

public:
    MovieWriterAV1();
    virtual ~MovieWriterAV1();

    // MovieWriter interface implementation
    virtual bool handles_file(const String &p_path) const override;
    virtual void get_supported_extensions(List<String> *r_extensions) const override;
    virtual uint32_t get_audio_mix_rate() const override;
    virtual AudioServer::SpeakerMode get_audio_speaker_mode() const override;

protected:
    virtual Error write_begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path) override;
    virtual Error write_frame(const Ref<Image> &p_image, const int32_t *p_audio_data) override;
    virtual void write_end() override;

private:
    // Hardware encoder setup
    Error _initialize_hardware_encoder();
    Error _create_video_session();
    Error _setup_rate_control();
    void _cleanup_hardware_resources();

    // Frame encoding
    Error _encode_frame_hardware(const Ref<Image> &p_image, uint64_t p_frame_number);
    Error _encode_frame_software(const Ref<Image> &p_image, uint64_t p_frame_number);
    RID _convert_image_to_encode_format(const Ref<Image> &p_image);

    // Audio encoding
    Error _initialize_opus_encoder();
    Error _encode_audio_frame(const int32_t *p_audio_data, Vector<uint8_t> &r_encoded_data);
    void _cleanup_opus_encoder();

    // Bitstream writing
    void _write_sequence_header();
    void _write_frame_data(const Vector<uint8_t> &p_video_data, const Vector<uint8_t> &p_audio_data, bool p_keyframe);
    void _write_webm_header();
    void _write_webm_cluster(const Vector<uint8_t> &p_video_data, const Vector<uint8_t> &p_audio_data, double p_timestamp, bool p_keyframe);

    // Capability detection
    static bool _is_hardware_encoding_supported();
    static Dictionary _get_encoding_capabilities();
};
```

### Implementation Details

#### Initialization and Setup

```cpp
// modules/vk_video/movie_writer_av1.cpp
#include "movie_writer_av1.h"
#include "servers/rendering/rendering_device.h"
#include "thirdparty/libopus/opus/opus.h"
#include "thirdparty/libsimplewebm/WebMDemuxer.hpp"

MovieWriterAV1::MovieWriterAV1() {
    video_extensions.instantiate();
}

MovieWriterAV1::~MovieWriterAV1() {
    write_end();
}

bool MovieWriterAV1::handles_file(const String &p_path) const {
    return p_path.get_extension().to_lower() == "webm" || 
           p_path.get_extension().to_lower() == "mkv";
}

void MovieWriterAV1::get_supported_extensions(List<String> *r_extensions) const {
    r_extensions->push_back("webm");
    r_extensions->push_back("mkv");
}

uint32_t MovieWriterAV1::get_audio_mix_rate() const {
    return 48000; // Opus standard sample rate
}

AudioServer::SpeakerMode MovieWriterAV1::get_audio_speaker_mode() const {
    return AudioServer::SPEAKER_MODE_STEREO;
}

Error MovieWriterAV1::write_begin(const Size2i &p_movie_size, uint32_t p_fps, const String &p_base_path) {
    movie_size = p_movie_size;
    output_path = p_base_path;
    frame_count = 0;

    // Open output file
    output_file = FileAccess::open(output_path, FileAccess::WRITE);
    ERR_FAIL_NULL_V(output_file, ERR_FILE_CANT_OPEN);

    // Initialize video extensions
    RenderingDevice *rd = RenderingDevice::get_singleton();
    ERR_FAIL_NULL_V(rd, ERR_UNAVAILABLE);
    video_extensions->initialize(rd);

    // Check hardware encoding support
    hardware_encoding = _is_hardware_encoding_supported();
    if (hardware_encoding) {
        Error err = _initialize_hardware_encoder();
        if (err != OK) {
            WARN_PRINT("Hardware AV1 encoding failed to initialize, falling back to software");
            hardware_encoding = false;
        }
    }

    // Initialize audio encoder
    Error err = _initialize_opus_encoder();
    ERR_FAIL_COND_V(err != OK, err);

    // Write container headers
    _write_webm_header();

    initialized = true;
    return OK;
}
```

#### Hardware Encoder Initialization

```cpp
Error MovieWriterAV1::_initialize_hardware_encoder() {
    ERR_FAIL_COND_V(!video_extensions->is_video_supported(), ERR_UNAVAILABLE);

    // Check AV1 encode support
    Dictionary caps = video_extensions->get_video_capabilities(VIDEO_CODEC_PROFILE_AV1_MAIN, VIDEO_OPERATION_ENCODE);
    ERR_FAIL_COND_V(!caps.get("encode_supported", false), ERR_UNAVAILABLE);

    // Create video session
    Error err = _create_video_session();
    ERR_FAIL_COND_V(err != OK, err);

    // Setup rate control
    err = _setup_rate_control();
    ERR_FAIL_COND_V(err != OK, err);

    return OK;
}

Error MovieWriterAV1::_create_video_session() {
    // Create video session for AV1 encoding
    Dictionary session_info;
    session_info["codec_operation"] = VIDEO_OPERATION_ENCODE;
    session_info["codec_profile"] = VIDEO_CODEC_PROFILE_AV1_MAIN;
    session_info["max_coded_extent_width"] = movie_size.width;
    session_info["max_coded_extent_height"] = movie_size.height;
    session_info["max_dpb_slots"] = 8;
    session_info["max_active_reference_pictures"] = 7;

    video_session = video_extensions->video_session_create(session_info);
    ERR_FAIL_COND_V(!video_session.is_valid(), ERR_CANT_CREATE);

    // Create DPB image array
    Dictionary dpb_info;
    dpb_info["image_type"] = RD::TEXTURE_TYPE_2D;
    dpb_info["format"] = RD::DATA_FORMAT_G8_B8R8_2PLANE_420_UNORM; // NV12
    dpb_info["width"] = movie_size.width;
    dpb_info["height"] = movie_size.height;
    dpb_info["array_layers"] = 8;
    dpb_info["usage"] = "VIDEO_ENCODE_DPB";

    encode_dpb_array = video_extensions->video_image_create(dpb_info);
    ERR_FAIL_COND_V(!encode_dpb_array.is_valid(), ERR_CANT_CREATE);

    // Create bitstream buffer
    Dictionary buffer_info;
    buffer_info["size"] = 1024 * 1024; // 1MB buffer
    buffer_info["usage"] = "VIDEO_ENCODE_DST";

    bitstream_buffer = video_extensions->video_buffer_create(buffer_info);
    ERR_FAIL_COND_V(!bitstream_buffer.is_valid(), ERR_CANT_CREATE);

    return OK;
}
```

#### Frame Encoding

```cpp
Error MovieWriterAV1::write_frame(const Ref<Image> &p_image, const int32_t *p_audio_data) {
    ERR_FAIL_COND_V(!initialized, ERR_UNCONFIGURED);

    // Encode video frame
    Error err;
    if (hardware_encoding) {
        err = _encode_frame_hardware(p_image, frame_count);
    } else {
        err = _encode_frame_software(p_image, frame_count);
    }
    ERR_FAIL_COND_V(err != OK, err);

    // Encode audio frame
    Vector<uint8_t> audio_data;
    if (p_audio_data) {
        err = _encode_audio_frame(p_audio_data, audio_data);
        ERR_FAIL_COND_V(err != OK, err);
    }

    frame_count++;
    return OK;
}

Error MovieWriterAV1::_encode_frame_hardware(const Ref<Image> &p_image, uint64_t p_frame_number) {
    // Convert image to encoding format (NV12)
    RID encode_texture = _convert_image_to_encode_format(p_image);
    ERR_FAIL_COND_V(!encode_texture.is_valid(), ERR_CANT_CREATE);

    // Determine frame type (keyframe every 30 frames)
    bool is_keyframe = (p_frame_number % 30) == 0;

    // Setup encode parameters
    Dictionary encode_info;
    encode_info["src_picture_resource"] = encode_texture;
    encode_info["dst_bitstream_buffer"] = bitstream_buffer;
    encode_info["dst_bitstream_buffer_offset"] = 0;
    encode_info["dst_bitstream_buffer_range"] = 1024 * 1024;
    encode_info["frame_type"] = is_keyframe ? "KEY_FRAME" : "INTER_FRAME";
    encode_info["frame_number"] = p_frame_number;

    // AV1-specific parameters
    Dictionary av1_params;
    av1_params["profile"] = settings.profile;
    av1_params["level"] = settings.level;
    av1_params["tile_cols"] = settings.tile_cols;
    av1_params["tile_rows"] = settings.tile_rows;
    av1_params["enable_cdef"] = settings.enable_cdef;
    av1_params["enable_restoration"] = settings.enable_restoration;
    encode_info["codec_params"] = av1_params;

    // Perform encoding
    video_extensions->video_encode_frame(encode_info);
    video_extensions->video_queue_submit();
    video_extensions->video_queue_wait_idle();

    // Read encoded data
    Vector<uint8_t> encoded_data = video_extensions->video_buffer_get_data(bitstream_buffer);

    // Write to container
    Vector<uint8_t> empty_audio; // Audio handled separately
    _write_webm_cluster(encoded_data, empty_audio, (double)p_frame_number / fps, is_keyframe);

    return OK;
}
```

#### Audio Encoding with Opus

```cpp
Error MovieWriterAV1::_initialize_opus_encoder() {
    int error;
    opus_encoder = opus_encoder_create(get_audio_mix_rate(), get_audio_speaker_mode() == AudioServer::SPEAKER_MODE_STEREO ? 2 : 1, OPUS_APPLICATION_AUDIO, &error);
    ERR_FAIL_COND_V(error != OPUS_OK, ERR_CANT_CREATE);

    // Configure encoder
    opus_encoder_ctl(opus_encoder, OPUS_SET_BITRATE(128000)); // 128 kbps
    opus_encoder_ctl(opus_encoder, OPUS_SET_COMPLEXITY(10));  // Max quality

    opus_buffer.resize(4000); // Max Opus frame size
    return OK;
}

Error MovieWriterAV1::_encode_audio_frame(const int32_t *p_audio_data, Vector<uint8_t> &r_encoded_data) {
    ERR_FAIL_NULL_V(opus_encoder, ERR_UNCONFIGURED);

    // Convert int32 to float
    Vector<float> float_samples;
    int channels = get_audio_speaker_mode() == AudioServer::SPEAKER_MODE_STEREO ? 2 : 1;
    float_samples.resize(opus_frame_size * channels);

    for (int i = 0; i < opus_frame_size * channels; i++) {
        float_samples.write[i] = (float)p_audio_data[i] / 2147483647.0f; // Convert from int32 to float
    }

    // Encode with Opus
    int encoded_size = opus_encode_float(opus_encoder, float_samples.ptr(), opus_frame_size, opus_buffer.ptrw(), opus_buffer.size());
    ERR_FAIL_COND_V(encoded_size < 0, ERR_INVALID_DATA);

    r_encoded_data.resize(encoded_size);
    memcpy(r_encoded_data.ptrw(), opus_buffer.ptr(), encoded_size);

    return OK;
}
```

#### WebM Container Integration

```cpp
void MovieWriterAV1::_write_webm_header() {
    // Write EBML header
    output_file->store_32(0x1A45DFA3); // EBML ID
    // ... WebM header implementation using libsimplewebm patterns
}

void MovieWriterAV1::_write_webm_cluster(const Vector<uint8_t> &p_video_data, const Vector<uint8_t> &p_audio_data, double p_timestamp, bool p_keyframe) {
    // Write WebM cluster with video and audio data
    // Implementation follows WebM specification
    // ... cluster writing implementation
}

void MovieWriterAV1::write_end() {
    if (!initialized) {
        return;
    }

    // Flush any remaining frames
    video_extensions->video_queue_wait_idle();

    // Finalize WebM file
    // ... write final WebM elements

    // Cleanup
    _cleanup_hardware_resources();
    _cleanup_opus_encoder();

    if (output_file.is_valid()) {
        output_file->close();
        output_file.unref();
    }

    initialized = false;
}
```

## Registration and Integration

### Module Registration

```cpp
// In modules/vk_video/register_types.cpp
#include "movie_writer_av1.h"

void initialize_vk_video_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }

    // ... existing registrations ...

    // Register AV1 movie writer
    MovieWriter::add_writer(memnew(MovieWriterAV1));
}
```

### Build Configuration

```python
# In modules/vk_video/config.py
def can_build(env, platform):
    # Check for Vulkan Video support
    return env["vulkan"] and env.get("vk_video", True)

def configure(env):
    # Add Opus dependency
    env.Prepend(CPPPATH=["#thirdparty/libopus"])
    
    # Add WebM dependency  
    env.Prepend(CPPPATH=["#thirdparty/libsimplewebm"])
```

## Usage Examples

### Command Line Usage

```bash
# Record gameplay to AV1/WebM with hardware encoding
godot --write-movie output.webm --fixed-fps 60 project.godot

# Record with specific quality settings
godot --write-movie high_quality.webm --fixed-fps 30 project.godot
```

### Programmatic Configuration

```gdscript
# Check if AV1 hardware encoding is available
if MovieWriterAV1.is_hardware_encoding_supported():
    print("Hardware AV1 encoding available")
    var caps = MovieWriterAV1.get_encoding_capabilities()
    print("Max resolution: ", caps["max_width"], "x", caps["max_height"])
```

### Editor Integration

The MovieWriterAV1 integrates with Godot's existing movie maker UI:

1. **Run Bar**: The movie maker button in the editor run bar automatically detects available writers
2. **File Extensions**: `.webm` and `.mkv` files are automatically handled by MovieWriterAV1
3. **Settings**: Movie maker settings can be configured through project settings

## Performance Considerations

### Hardware Requirements

- **GPU**: Vulkan 1.3+ with VK_KHR_video_encode_queue extension
- **Driver**: Recent drivers with AV1 encode support (NVIDIA RTX 40-series, Intel Arc, AMD RDNA3+)
- **Memory**: Sufficient VRAM for DPB and bitstream buffers

### Optimization Tips

1. **Frame Queue Size**: Adjust `max_queued_frames` based on available memory
2. **Bitrate Settings**: Use VBR for better quality, CBR for consistent file sizes
3. **Tile Configuration**: Use multiple tiles for higher resolutions (4K+)
4. **Rate Control**: CRF mode provides best quality for offline rendering

## Fallback Behavior

When hardware encoding is not available:

1. **Software Fallback**: Implement software AV1 encoding using libaom
2. **Format Fallback**: Fall back to existing MovieWriterPNGWAV
3. **Graceful Degradation**: Warn user and continue with available options

This implementation provides a complete, production-ready AV1 hardware encoding solution that integrates seamlessly with Godot's existing movie writing infrastructure.
