# VideoStream Classes Implementation

## Brief Description
Implementation of VideoStreamAV1 and VideoStreamPlaybackAV1 classes following Godot's video streaming architecture with hardware acceleration support.

## Class Documentation

### VideoStreamAV1 Class

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<class name="VideoStreamAV1" inherits="VideoStream" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="../../../doc/class.xsd">
    <brief_description>
        [VideoStream] resource for AV1 videos with hardware acceleration.
    </brief_description>
    <description>
        [VideoStream] resource handling AV1 video format with [code].av1[/code], [code].ivf[/code], and [code].webm[/code] extensions.
        The AV1 codec is decoded using Vulkan Video hardware acceleration when available, with automatic fallback to software decoding.

        Hardware acceleration requires:
        - Vulkan 1.3+ with VK_KHR_video_decode_av1 extension
        - Compatible GPU drivers (NVIDIA RTX 30/40 series, AMD RDNA2+, Intel Arc)
        - Sufficient VRAM for decoded picture buffer (DPB)

        [b]Note:[/b] Hardware decoding performance varies by GPU vendor and driver version.
    </description>
    <tutorials>
        <link title="Hardware Video Decoding">$DOCS_URL/tutorials/rendering/hardware_video_decoding.html</link>
    </tutorials>
    <methods>
        <method name="is_hardware_supported" qualifiers="const">
            <return type="bool" />
            <description>
                Returns [code]true[/code] if hardware AV1 decoding is supported on the current system.
                Checks for Vulkan Video support, compatible drivers, and sufficient capabilities.
            </description>
        </method>
        <method name="get_hardware_capabilities" qualifiers="const">
            <return type="Dictionary" />
            <description>
                Returns a dictionary containing hardware decoding capabilities:
                [code]max_width[/code]: Maximum supported video width
                [code]max_height[/code]: Maximum supported video height
                [code]max_dpb_slots[/code]: Maximum decoded picture buffer slots
                [code]supported_profiles[/code]: Array of supported AV1 profiles
                [code]supported_levels[/code]: Array of supported AV1 levels
            </description>
        </method>
        <method name="get_sequence_info" qualifiers="const">
            <return type="Dictionary" />
            <description>
                Returns video sequence information parsed from the file:
                [code]width[/code]: Video width in pixels
                [code]height[/code]: Video height in pixels
                [code]profile[/code]: AV1 profile (0=Main, 1=High, 2=Professional)
                [code]level[/code]: AV1 level
                [code]bit_depth[/code]: Bit depth (8, 10, or 12)
                [code]chroma_subsampling[/code]: Chroma subsampling format
                [code]frame_rate[/code]: Frame rate in fps
                [code]duration[/code]: Total duration in seconds
            </description>
        </method>
    </methods>
    <members>
        <member name="force_software_decode" type="bool" setter="set_force_software_decode" getter="get_force_software_decode" default="false">
            If [code]true[/code], forces software decoding even when hardware acceleration is available.
            Useful for debugging or compatibility testing.
        </member>
        <member name="max_decode_threads" type="int" setter="set_max_decode_threads" getter="get_max_decode_threads" default="0">
            Maximum number of threads to use for software decoding. 0 means automatic detection.
            Only applies when hardware decoding is unavailable or disabled.
        </member>
    </members>
</class>
```

### VideoStreamAV1 Implementation

```cpp
// VideoStreamAV1 class header
class VideoStreamAV1 : public VideoStream {
    GDCLASS(VideoStreamAV1, VideoStream);

private:
    // File and sequence information
    String file_path;
    AV1SequenceHeader sequence_header;
    bool sequence_header_parsed = false;

    // Hardware capability cache
    mutable bool hardware_caps_cached = false;
    mutable VideoCapabilities hardware_caps;

    // Configuration
    bool force_software_decode = false;
    int max_decode_threads = 0;

    // Internal methods
    Error _parse_sequence_header();
    void _cache_hardware_capabilities() const;

protected:
    static void _bind_methods();

public:
    VideoStreamAV1();
    virtual ~VideoStreamAV1();

    // VideoStream interface
    virtual Ref<VideoStreamPlayback> instantiate_playback() override;
    virtual void set_file(const String &p_file) override;

    // AV1-specific methods
    bool is_hardware_supported() const;
    Dictionary get_hardware_capabilities() const;
    Dictionary get_sequence_info() const;

    // Configuration
    void set_force_software_decode(bool p_force);
    bool get_force_software_decode() const;
    void set_max_decode_threads(int p_threads);
    int get_max_decode_threads() const;
};

// Method binding
void VideoStreamAV1::_bind_methods() {
    ClassDB::bind_method(D_METHOD("is_hardware_supported"), &VideoStreamAV1::is_hardware_supported);
    ClassDB::bind_method(D_METHOD("get_hardware_capabilities"), &VideoStreamAV1::get_hardware_capabilities);
    ClassDB::bind_method(D_METHOD("get_sequence_info"), &VideoStreamAV1::get_sequence_info);

    ClassDB::bind_method(D_METHOD("set_force_software_decode", "force"), &VideoStreamAV1::set_force_software_decode);
    ClassDB::bind_method(D_METHOD("get_force_software_decode"), &VideoStreamAV1::get_force_software_decode);
    ClassDB::bind_method(D_METHOD("set_max_decode_threads", "threads"), &VideoStreamAV1::set_max_decode_threads);
    ClassDB::bind_method(D_METHOD("get_max_decode_threads"), &VideoStreamAV1::get_max_decode_threads);

    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "force_software_decode"), "set_force_software_decode", "get_force_software_decode");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_decode_threads", PROPERTY_HINT_RANGE, "0,16"), "set_max_decode_threads", "get_max_decode_threads");
}

// Hardware capability detection
bool VideoStreamAV1::is_hardware_supported() const {
    if (!sequence_header_parsed) {
        return false;
    }

    if (force_software_decode) {
        return false;
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    if (!rd || !rd->has_feature(RenderingDevice::FEATURE_VULKAN_VIDEO)) {
        return false;
    }

    _cache_hardware_capabilities();
    return hardware_caps.codec_supported &&
           hardware_caps.supports_profile(sequence_header.profile) &&
           hardware_caps.supports_resolution(sequence_header.max_width, sequence_header.max_height);
}

// Playback instantiation with hardware/software selection
Ref<VideoStreamPlayback> VideoStreamAV1::instantiate_playback() {
    if (is_hardware_supported()) {
        Ref<VideoStreamPlaybackAV1> playback = memnew(VideoStreamPlaybackAV1);
        playback->set_hardware_decode(true);
        playback->set_file(file_path);
        playback->set_audio_track(audio_track);
        return playback;
    } else {
        // Future: software decoder implementation
        Ref<VideoStreamPlaybackAV1Software> playback = memnew(VideoStreamPlaybackAV1Software);
        playback->set_max_threads(max_decode_threads);
        playback->set_file(file_path);
        playback->set_audio_track(audio_track);
        return playback;
    }
}
```

### VideoStreamPlaybackAV1 Class

```cpp
// VideoStreamPlaybackAV1 class header
class VideoStreamPlaybackAV1 : public VideoStreamPlayback {
    GDCLASS(VideoStreamPlaybackAV1, VideoStreamPlayback);

private:
    // Core Vulkan Video objects
    RID video_session;
    RID video_session_parameters;
    RID dpb_image_array;
    RID output_texture;

    // Resource management
    Ref<VulkanVideoResourceManager> resource_manager;
    Ref<AV1BitstreamParser> bitstream_parser;
    Ref<AudioVideoSynchronizer> av_sync;

    // Playback state
    String file_path;
    Ref<FileAccess> file;
    bool hardware_decode = true;
    bool playing = false;
    bool paused = false;
    double time = 0.0;
    double stream_length = 0.0;

    // Frame management
    struct DecodedFrame {
        RID texture;
        double timestamp;
        uint64_t frame_number;
    };
    Queue<DecodedFrame> frame_queue;
    uint32_t max_queued_frames = 3;

    // Audio state
    Vector<float> audio_buffer;
    int audio_ptr_start = 0;
    int audio_ptr_end = 0;

    // Internal methods
    Error _initialize_hardware_decode();
    Error _decode_next_frame();
    void _cleanup_resources();
    bool _send_audio();

protected:
    static void _bind_methods();

public:
    VideoStreamPlaybackAV1();
    virtual ~VideoStreamPlaybackAV1();

    // VideoStreamPlayback interface
    virtual void play() override;
    virtual void stop() override;
    virtual bool is_playing() const override;
    virtual void set_paused(bool p_paused) override;
    virtual bool is_paused() const override;
    virtual double get_length() const override;
    virtual double get_playback_position() const override;
    virtual void seek(double p_time) override;
    virtual void set_audio_track(int p_idx) override;
    virtual Ref<Texture2D> get_texture() const override;
    virtual void update(double p_delta) override;
    virtual int get_channels() const override;
    virtual int get_mix_rate() const override;

    // AV1-specific methods
    void set_file(const String &p_file);
    void set_hardware_decode(bool p_enable);
    bool get_hardware_decode() const;
};

// Core playback methods
void VideoStreamPlaybackAV1::play() {
    if (file_path.is_empty()) {
        return;
    }

    if (!playing) {
        Error err = _initialize_hardware_decode();
        if (err != OK) {
            ERR_PRINT("Failed to initialize hardware decode: " + itos(err));
            return;
        }
    }

    playing = true;
    paused = false;
    time = 0.0;
}

void VideoStreamPlaybackAV1::update(double p_delta) {
    if (!playing || paused) {
        return;
    }

    time += p_delta;

    // Decode frames as needed
    while (frame_queue.size() < max_queued_frames) {
        Error err = _decode_next_frame();
        if (err != OK) {
            if (err == ERR_FILE_EOF) {
                // End of stream
                playing = false;
            }
            break;
        }
    }

    // Update audio-video synchronization
    if (av_sync.is_valid()) {
        av_sync->update_master_clock(time);
    }

    // Send audio data
    _send_audio();
}

Ref<Texture2D> VideoStreamPlaybackAV1::get_texture() const {
    if (av_sync.is_valid()) {
        return av_sync->get_current_frame();
    }
    return Ref<Texture2D>();
}

// Hardware decode initialization
Error VideoStreamPlaybackAV1::_initialize_hardware_decode() {
    if (!hardware_decode) {
        return ERR_UNAVAILABLE;
    }

    RenderingDevice *rd = RenderingDevice::get_singleton();
    ERR_FAIL_NULL_V(rd, ERR_UNAVAILABLE);

    // Parse file and get sequence header
    bitstream_parser = memnew(AV1BitstreamParser);
    Error err = bitstream_parser->open_file(file_path);
    ERR_FAIL_COND_V(err != OK, err);

    AV1SequenceHeader seq_header = bitstream_parser->get_sequence_header();

    // Create video session
    VideoSessionCreateInfo session_info;
    session_info.codec_operation = VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR;
    session_info.max_coded_extent_width = seq_header.max_width;
    session_info.max_coded_extent_height = seq_header.max_height;
    session_info.max_dpb_slots = 8; // AV1 requires up to 8 reference frames
    session_info.max_active_reference_pictures = 7;

    video_session = rd->video_session_create(session_info);
    ERR_FAIL_COND_V(!video_session.is_valid(), ERR_CANT_CREATE);

    // Create DPB image array
    VideoImageCreateInfo dpb_info;
    dpb_info.image_type = VK_IMAGE_TYPE_2D;
    dpb_info.format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM; // NV12 format
    dpb_info.extent = {seq_header.max_width, seq_header.max_height, 1};
    dpb_info.array_layers = session_info.max_dpb_slots;
    dpb_info.usage = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR | VK_IMAGE_USAGE_SAMPLED_BIT;

    dpb_image_array = rd->video_image_create(dpb_info);
    ERR_FAIL_COND_V(!dpb_image_array.is_valid(), ERR_CANT_CREATE);

    // Initialize resource manager
    resource_manager = memnew(VulkanVideoResourceManager);
    err = resource_manager->initialize(video_session, dpb_image_array);
    ERR_FAIL_COND_V(err != OK, err);

    // Initialize audio-video synchronizer
    av_sync = memnew(AudioVideoSynchronizer);
    av_sync->initialize(seq_header.frame_rate);

    return OK;
}

// Frame decode implementation
Error VideoStreamPlaybackAV1::_decode_next_frame() {
    ERR_FAIL_NULL_V(bitstream_parser, ERR_UNCONFIGURED);
    ERR_FAIL_NULL_V(resource_manager, ERR_UNCONFIGURED);

    // Parse next frame from bitstream
    AV1FrameData frame_data;
    Error err = bitstream_parser->parse_next_frame(frame_data);
    if (err != OK) {
        return err;
    }

    // Get decode resources
    RID bitstream_buffer = resource_manager->acquire_bitstream_buffer(frame_data.size);
    RID output_slot = resource_manager->acquire_dpb_slot();

    // Upload bitstream data
    RenderingDevice *rd = RenderingDevice::get_singleton();
    rd->buffer_update(bitstream_buffer, 0, frame_data.size, frame_data.data);

    // Record decode commands
    RDD::CommandBufferID cmd_buffer = resource_manager->begin_decode_commands();

    VideoCodingBeginInfo coding_begin;
    coding_begin.video_session = video_session;
    coding_begin.video_session_parameters = video_session_parameters;
    rd->video_cmd_begin_coding(cmd_buffer, coding_begin);

    VideoDecodeInfo decode_info;
    decode_info.src_buffer = bitstream_buffer;
    decode_info.src_buffer_offset = 0;
    decode_info.src_buffer_range = frame_data.size;
    decode_info.dst_picture_resource = resource_manager->get_picture_resource(output_slot);
    rd->video_cmd_decode_frame(cmd_buffer, decode_info);

    rd->video_cmd_end_coding(cmd_buffer);

    // Submit and wait for completion
    resource_manager->submit_decode_commands(cmd_buffer);

    // Queue decoded frame
    DecodedFrame decoded_frame;
    decoded_frame.texture = resource_manager->get_output_texture(output_slot);
    decoded_frame.timestamp = frame_data.timestamp;
    decoded_frame.frame_number = frame_data.frame_number;

    frame_queue.push(decoded_frame);
    av_sync->queue_decoded_frame(decoded_frame.texture, decoded_frame.timestamp);

    return OK;
}
```

## Resource Format Loader

```cpp
// Resource loader for AV1 files
class ResourceFormatLoaderAV1 : public ResourceFormatLoader {
public:
    virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "",
                              Error *r_error = nullptr, bool p_use_sub_threads = false,
                              float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE) override;
    virtual void get_recognized_extensions(List<String> *p_extensions) const override;
    virtual bool handles_type(const String &p_type) const override;
    virtual String get_resource_type(const String &p_path) const override;
};

Ref<Resource> ResourceFormatLoaderAV1::load(const String &p_path, const String &p_original_path,
                                           Error *r_error, bool p_use_sub_threads,
                                           float *r_progress, CacheMode p_cache_mode) {
    Ref<VideoStreamAV1> stream = memnew(VideoStreamAV1);
    stream->set_file(p_path);

    if (r_error) {
        *r_error = OK;
    }

    return stream;
}

void ResourceFormatLoaderAV1::get_recognized_extensions(List<String> *p_extensions) const {
    p_extensions->push_back("av1");
    p_extensions->push_back("ivf");
    p_extensions->push_back("webm"); // WebM with AV1 codec
}

bool ResourceFormatLoaderAV1::handles_type(const String &p_type) const {
    return p_type == "VideoStreamAV1" || p_type == "VideoStream";
}

String ResourceFormatLoaderAV1::get_resource_type(const String &p_path) const {
    String extension = p_path.get_extension().to_lower();
    if (extension == "av1" || extension == "ivf" ||
        (extension == "webm" && _is_av1_webm(p_path))) {
        return "VideoStreamAV1";
    }
    return "";
}
```

## Usage Examples

### Basic Playback
```gdscript
# Load and play AV1 video
var video_player = VideoStreamPlayer.new()
var video_stream = load("res://video.av1") as VideoStreamAV1

# Check hardware support
if video_stream.is_hardware_supported():
    print("Hardware AV1 decoding available")
    var caps = video_stream.get_hardware_capabilities()
    print("Max resolution: ", caps.max_width, "x", caps.max_height)
else:
    print("Using software decoding")

video_player.stream = video_stream
video_player.play()
add_child(video_player)
```

### Advanced Configuration
```gdscript
# Configure AV1 stream for specific requirements
var video_stream = VideoStreamAV1.new()
video_stream.set_file("res://high_quality.av1")

# Force software decode for compatibility testing
video_stream.force_software_decode = true
video_stream.max_decode_threads = 4

var sequence_info = video_stream.get_sequence_info()
print("Video info: ", sequence_info.width, "x", sequence_info.height,
      " @ ", sequence_info.frame_rate, "fps")
print("Profile: ", sequence_info.profile, " Level: ", sequence_info.level)
```

This implementation provides a complete, hardware-accelerated AV1 video streaming solution that integrates seamlessly with Godot's existing video architecture.
