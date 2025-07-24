[Google Gemini](https://gemini.google.com/app/01b67e81b550186c)

# In-Core Integration of Hardware-Accelerated AV1 Encoding and Decoding in Godot Engine via Vulkan Video

## Section 1: Foundational Architectures

A successful in-core implementation of hardware-accelerated AV1 video processing within the Godot Engine necessitates a profound understanding of two distinct yet interconnected systems: the engine's own rendering and video playback architecture, and the low-level Vulkan Video framework that provides access to the GPU's dedicated hardware. This section provides a comprehensive analysis of both systems, establishing the architectural context and constraints that will govern the design of a native, high-performance AV1 module.

### 1.1. Analysis of Godot's Rendering and Video Subsystems

Godot's internal architecture is meticulously designed for modularity, portability, and performance. Any new core functionality must respect and integrate with these established principles. A deep dive into its rendering and video subsystems reveals a clear path for integration.

#### The Server Architecture

The engine's architecture is fundamentally built upon a server-based model, which creates a robust separation between the high-level game logic, managed by the `SceneTree`, and the low-level implementation details of rendering, physics, and audio. At the heart of the rendering pipeline is the

`RenderingServer`, a singleton that acts as the sole interface for all rendering commands and resource management. Nodes within the

`SceneTree`, such as `MeshInstance3D` or `Sprite2D`, do not draw themselves directly; instead, they submit draw commands and resource definitions to the `RenderingServer`. This design is paramount for several reasons: it centralizes rendering logic, simplifies multi-threading by ensuring all low-level graphics API calls are funneled through a single, controlled point, and decouples the scene graph from the underlying graphics backend. For our AV1 module, this means all communication with the GPU must be routed through the `RenderingServer`'s API to maintain architectural integrity and thread safety.

#### The `RenderingDevice` Abstraction

In Godot 4, the engine's commitment to cross-platform compatibility was further solidified with the introduction of the `RenderingDevice` abstraction layer. The

`RenderingDevice` is a low-level, backend-agnostic interface that the `RenderingServer` uses to communicate with the active graphics API, be it Vulkan, Direct3D 12, or Metal. It exposes a common set of functions for creating and managing fundamental GPU resources like buffers, textures (images), shaders, and synchronization primitives. The source code for this critical abstraction can be found in `servers/rendering/rendering_device.h`.

This architectural choice is the primary enabler for a clean, maintainable, and truly "in-core" AV1 implementation. A naive approach might involve writing Vulkan-specific code directly within the new module. However, this would violate Godot's core design philosophy of portability. By exclusively using the

`RenderingDevice` API, our AV1 module can request the creation of GPU resources without any direct knowledge of the underlying backend. This ensures that, while our initial implementation will target Vulkan, the groundwork is laid for future support on other backends if they were to adopt similar video processing extensions. This transforms the task from merely adding a feature into a more robust effort of extending the engine's core rendering abstraction itself, a far more future-proof strategy.

#### Deconstructing the Existing Video Pipeline

Godot's current video playback functionality is exposed to the user through the `VideoStreamPlayer` node. This

`Control` node is responsible for the user-facing aspects of video playback, such as size, position, and playback controls (`play`, `stop`, `pause`). The actual video data is handled by a `VideoStream` resource, which is an abstract base class. For the engine's built-in video support, this resource is a `VideoStreamTheora` instance.

An analysis of the `VideoStreamTheora` implementation reveals a purely CPU-bound pipeline. It utilizes the `libtheora` library to decode Ogg Theora video frames on the CPU. During playback, the `VideoStreamPlayer`'s internal logic repeatedly calls the `get_video_texture()` method on the playback object, which returns a `Texture2D` containing the latest decoded frame. This process involves decoding a frame into a raw pixel buffer in system memory (RAM) and then uploading that data to the GPU's video memory (VRAM) as a new texture for every single frame.

This pipeline suffers from several critical limitations that motivate the need for a modern, hardware-accelerated solution:

- **Outdated Codec:** Ogg Theora is an old and inefficient codec compared to modern standards like AV1, resulting in larger file sizes for comparable quality.
- **CPU Bottleneck:** Decoding is performed entirely on the CPU. For high-resolution video (e.g., 1080p or 4K), this can consume significant CPU resources, leading to performance degradation, stuttering, and increased power consumption, especially on mobile devices.

- **Inefficient Data Path:** The constant transfer of decoded frames from system memory to VRAM consumes significant bus bandwidth, which could otherwise be used for other rendering tasks.
- **Limited Functionality:** The current implementation lacks essential features like seeking (`stream_position` has no effect), a long-standing issue noted by the community.

This existing structure, however, provides a valuable template. The separation of concerns between the `VideoStreamPlayer` (UI/control), `VideoStream` (resource), and a `VideoStreamPlayback` object (stateful playback instance) is a sound design.

Feature

`VideoStreamTheora` (Current)

`VideoStreamAV1` (Proposed)

**Execution Unit**

CPU (via `libtheora`)

GPU (via Vulkan Video)

**Codec**

Ogg Theora

AV1

**Performance**

Low (bottleneck on high-res)

High (hardware accelerated)

**Data Path**

GPU - CPU - GPU (Frame Upload)

GPU-only (Zero-Copy)

**Seeking**

Not Implemented

Feasible (via keyframe seeking)

**Power Consumption**

High on CPU

Low (dedicated hardware block)

#### The Path Forward

The goal is to architect a new set of classes, `VideoStreamAV1` and `VideoStreamPlaybackAV1`, that adhere to the existing `VideoStream` interface. The `VideoStreamPlayer` node will interact with this new implementation seamlessly, without requiring any changes to its own logic. The crucial difference will be internal: instead of calling `libtheora` on the CPU, `VideoStreamPlaybackAV1` will orchestrate a series of commands through the `RenderingDevice` to instruct the GPU to perform the decoding.

The complexity of the Vulkan Video API, with its stateful sessions and explicit resource management, necessitates that the `VideoStreamPlaybackAV1` class be a sophisticated state machine. It will encapsulate all the persistent Vulkan Video objects and manage the intricate lifecycle of the Decoded Picture Buffer (DPB) and reference frames. This design localizes the immense complexity of the low-level API within a single, well-defined class, presenting a clean, high-level interface (

`update`, `get_texture`) to the rest of the engine. This approach aligns perfectly with Godot's object-oriented design philosophy, ensuring the new feature is both powerful and well-integrated.

### 1.2. The Vulkan Video Framework for AV1

Vulkan Video is not a simple, high-level API. It is a comprehensive, low-level framework that extends the core Vulkan specification to provide fine-grained, explicit control over the entire hardware-accelerated video processing pipeline. It is designed for high-performance applications that require direct management of scheduling, synchronization, and memory.

#### Core Concepts

The framework introduces dedicated queue families for video operations, identified by the `VK_QUEUE_VIDEO_DECODE_BIT_KHR` and `VK_QUEUE_VIDEO_ENCODE_BIT_KHR` flags. These queues can operate asynchronously from the primary graphics and compute queues, enabling true parallelism where video decoding, 3D rendering, and compute tasks can all be in-flight on the GPU simultaneously.

#### Key Objects and State Management

The API is built around several key object types that manage the state of a video stream:

- **`VkVideoSessionKHR`**: This object represents the context for a single video stream. It is a stateful object that maintains persistent data required for decoding or encoding a sequence of pictures, most notably the state of the Decoded Picture Buffer (DPB).

- **`VkVideoSessionParametersKHR`**: This object is designed to store and manage codec-specific header information, such as the AV1 Sequence Header or H.264 Sequence Parameter Sets (SPS). By parsing this data once and storing it in a parameters object, the application avoids redundant processing on every frame and allows the driver to perform implementation-specific optimizations.

- **Decoded Picture Buffer (DPB) Management**: Modern video codecs rely heavily on inter-frame prediction, where previously decoded frames are used as references to decode subsequent frames. The DPB is the set of these reference pictures. In Vulkan Video, the application is fully responsible for managing the DPB. This involves allocating a set of `VkImage` resources to serve as DPB slots and explicitly telling the driver which slot to decode into and which slots to use as references for each frame.

#### AV1-Specific Extensions

The core, codec-agnostic framework is augmented by codec-specific extensions. For this implementation, two are critical:

- **`VK_KHR_video_decode_av1`**: This extension provides the necessary structures and enumerations for AV1 decoding.

- **`VK_KHR_video_encode_av1`**: This extension provides the corresponding definitions for AV1 encoding.

These extensions define Vulkan structures, such as `VkVideoDecodeAV1PictureInfoKHR` and `VkVideoEncodeAV1PictureInfoKHR`, whose members map directly to the syntax elements defined in the official AV1 bitstream specification. This design choice places the responsibility of parsing the bitstream on the application. Our module will need to read the AV1 file, extract the sequence and frame headers, populate these Vulkan structures, and pass them to the driver. The driver does not contain a full bitstream parser; it expects to receive the already-parsed syntax elements.

Vulkan Object/Command

Role in Pipeline

Relevant Source(s)

`VK_QUEUE_VIDEO_DECODE_BIT_KHR`

Queue family flag for decode operations

`VkVideoSessionKHR`

Manages persistent state for a video stream (e.g., DPB)

`VkVideoDecodeAV1ProfileInfoKHR`

Specifies the AV1 profile being used for capability checks

`vkCmdDecodeVideoKHR`

Records a video decode operation into a command buffer

`VkVideoEncodeAV1PictureInfoKHR`

Provides per-picture parameters for an AV1 encode operation

`vkCmdEncodeVideoKHR`

Records a video encode operation into a command buffer



#### The Command Structure

All video operations are recorded into a `VkCommandBuffer`, just like graphics or compute work. These operations must be enclosed within a "video coding scope," which is initiated by `vkCmdBeginVideoCodingKHR` and terminated by `vkCmdEndVideoCodingKHR`. Within this scope, one or more `vkCmdDecodeVideoKHR` or `vkCmdEncodeVideoKHR` commands are recorded to perform the actual work. This structure allows the driver to manage the state of the video session across multiple decode/encode operations within a single command buffer submission.

## Section 2: Implementation Strategy: A Native Engine Module

The architectural foundations of Godot and Vulkan Video dictate a clear implementation path. The optimal approach is to create a new, native C++ engine module that integrates deeply with the `RenderingDevice` abstraction. This section outlines the high-level design of this module, defining the class structures and the necessary extensions to Godot's core rendering API.

### 2.1. Rationale for a Core C++ Module vs. GDExtension

While Godot's GDExtension system is powerful for creating add-ons and integrating third-party libraries, a feature as performance-critical and deeply integrated as hardware-accelerated video processing warrants a native C++ module compiled directly into the engine.

- **Performance and Zero-Copy Pipelines:** The primary advantage of a core module is direct, unfettered access to the engine's internal `RenderingDevice` API. This is the only way to achieve a true zero-copy pipeline, a scenario where a video frame is decoded by the GPU into a

`VkImage` which is then used directly as a texture by the rendering pipeline, without ever being copied back to system memory. A GDExtension operates at a higher level of abstraction and would either be unable to access these low-level resources directly or would require complex and fragile workarounds, likely incurring a performance penalty that would negate the benefits of hardware acceleration.

- **Deep Engine Integration:** A native module can integrate more cleanly with other core engine systems. For instance, modifying Godot's Movie Maker to use our hardware encoder for real-time video capture is far more direct and robust from within the engine's source code. Similarly, exposing the decoded video frames as a standard texture resource that can be seamlessly used by the engine's material and shader systems is most effectively achieved at the core level.

- **Maintainability and Stability:** While GDExtensions offer modularity, they can also introduce versioning and dependency challenges. A fundamental capability like video playback benefits from being part of the core engine's continuous integration and testing pipeline. This ensures long-term stability and compatibility across engine versions. The Godot project has past experience with the maintenance burden of complex, dependency-heavy modules, such as the WebM module which was eventually removed in part due to these challenges. Given that compiling the engine from source is a well-documented and standard procedure for core contributors and advanced users, building the AV1 support directly into the engine is the most sustainable path.

### 2.2. Designing `VideoStreamAV1` and `VideoStreamPlaybackAV1`

Following the established pattern of the `VideoStreamTheora` implementation, our module will be centered around two primary classes.

- **`VideoStreamAV1` (Resource):**

  - This class will inherit from `VideoStream` and be registered as a new `Resource` type within the engine. Its primary role is to represent the AV1 video file asset.
  - **Responsibilities:**

    - Store the file path to the `.av1` or container file (e.g., `.mkv`, `.mp4`).
    - On load, it will be responsible for opening the file and using a lightweight demuxer/parser to extract essential stream-level metadata, most importantly the AV1 Sequence Header. This header data will be cached within the resource.
    - When playback is requested by a `VideoStreamPlayer`, it will instantiate and return a `VideoStreamPlaybackAV1` object, passing the file handle and cached header information to it.

- **`VideoStreamPlaybackAV1` (Playback State):**

  - This class will inherit from `VideoStreamPlayback`. It is the stateful workhorse of the implementation, managing the entire lifecycle of a Vulkan Video session for a single playback instance.
  - **Member Variables:**

    - It will hold Godot's abstract Resource IDs (`RID`s) for all GPU resources, which are created and managed via the `RenderingDevice`. These will include RIDs for the video session, the session parameters object, the `VkImage` array used for the DPB, the `VkBuffer` for the input bitstream, the final output texture, and the necessary synchronization primitives (semaphores).
    - It will also manage the state of the DPB, tracking which slots contain valid reference frames.

  - **Virtual Function Overrides:** It will provide concrete implementations for the key virtual functions inherited from its parent class:

    - `_play()`: This method will be called to begin playback. It will trigger the initialization of all necessary Vulkan video resources through the `RenderingDevice`.
    - `_stop()`: This method will release all GPU resources associated with the playback session.
    - `_update(delta)`: This is the main processing loop, called once per frame by the `VideoStreamPlayer`. It will be responsible for reading the next chunk of compressed data from the file, uploading it to the bitstream buffer, submitting a decode command to the video queue, and managing the queue of decoded frames for presentation.
    - `get_texture()`: This crucial method will return a `Texture2D` `RID` that points to the most recently decoded and ready-to-display video frame, which resides entirely in VRAM.
    - `get_playback_position()`: It will return the current presentation timestamp of the video, enabling synchronization with other engine events.

### 2.3. Extending Godot's `RenderingDevice` API

To maintain the engine's backend abstraction and keep our `VideoStreamPlaybackAV1` class free of Vulkan-specific code, we must extend the base `RenderingDevice` API. This involves adding new virtual functions to the `RenderingDevice` class in `servers/rendering/rendering_device.h` and then implementing them in the Vulkan backend, located in `drivers/vulkan/rendering_device_vulkan.cpp`.

This architectural decision has profound long-term benefits that extend beyond this initial AV1 implementation. The Vulkan Video API is intentionally designed with a layered approach: a codec-agnostic core framework (`VK_KHR_video_queue`) is augmented by codec-specific extensions. A forward-thinking design for our

`RenderingDevice` extension should mirror this. Instead of creating highly specific functions like `video_cmd_decode_av1_frame`, we should design more generic functions like `video_cmd_decode_frame`. This generic function would accept a structure containing a pointer to codec-specific parameter data.

This approach establishes a generic, extensible hardware video framework at the core of Godot's rendering abstraction. When the time comes to add support for other hardware-accelerated codecs like H.265 or VP9, the `RenderingDevice` API will not need to be changed again. Developers will simply need to implement new `VideoStream` classes (`VideoStreamH265`, etc.) that populate the generic structures with the appropriate H.265-specific data. This significantly reduces the effort required to support future codecs and ensures the engine's rendering layer remains clean and adaptable.

- **Proposed New `RenderingDevice` Functions:**

  - `RID video_session_create(video_profile)`: Creates a video session for a given profile.
  - `void video_session_destroy(RID session)`
  - `RID video_session_parameters_create(RID session)`: Creates a parameters object for a session.
  - `void video_session_parameters_destroy(RID params)`
  - `bool video_session_parameters_update(RID params, parameters_info)`: Adds new header info (e.g., PPS/SPS) to the parameters object.
  - `void video_session_bind_memory(RID session,...)`
  - `void video_cmd_begin_coding(CommandBufferID cmd_buffer, begin_info)`
  - `void video_cmd_decode_frame(CommandBufferID cmd_buffer, decode_info)`
  - `void video_cmd_encode_frame(CommandBufferID cmd_buffer, encode_info)`
  - `void video_cmd_end_coding(CommandBufferID cmd_buffer, end_info)`

This set of functions provides a complete, abstract interface for video processing, cleanly encapsulating all Vulkan-specific logic within the Vulkan driver where it belongs.

## Section 3: The In-Core AV1 Decoding Pipeline

This section provides a granular, step-by-step walkthrough of the decoding process, detailing the sequence of operations from hardware initialization to the final synchronization of a decoded frame for rendering within the Godot engine.

### 3.1. Initialization and Hardware Capability Verification

Before any decoding can occur, the implementation must rigorously verify that the user's hardware and drivers are capable of AV1 video decoding via Vulkan. This process is a non-negotiable first step to ensure stability and provide clear feedback in cases of incompatibility.

- **Querying for Extension and Queue Support:** The process begins within the `VideoStreamAV1` resource's initialization logic. It must query the active `RenderingDevice` to determine if the necessary Vulkan extensions are supported by the physical device. The critical extensions are `VK_KHR_video_queue`, `VK_KHR_video_decode_queue`, and the codec-specific `VK_KHR_video_decode_av1`. Concurrently, it must query the available queue families to find one that reports the

`VK_QUEUE_VIDEO_DECODE_BIT_KHR` capability flag. The presence of a dedicated video decode queue is the fundamental prerequisite for hardware-accelerated playback.

- **Profile and Capability Checks:** If the required extensions and queue family are present, the next step is to check for support of the specific AV1 profile being requested. This is accomplished by calling `vkGetPhysicalDeviceVideoCapabilitiesKHR`. The call is made with a `VkVideoProfileInfoKHR` structure that specifies the codec operation (`VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR`), chroma subsampling, and bit depth. Crucially, this structure must have a `VkVideoDecodeAV1ProfileInfoKHR` structure chained to its `pNext` member, which specifies the exact AV1 profile (e.g., `STD_VIDEO_AV1_PROFILE_MAIN`). The results of this query provide vital information, such as the maximum supported AV1 level and resolution, which can be used to validate if the video file is playable on the current hardware. If any of these checks fail, the

- `VideoStreamAV1` resource should be marked as invalid, preventing any attempt at playback and logging a descriptive error to the console.

### 3.2. Resource Management and the Decode Loop

Once capabilities are verified, the `VideoStreamPlaybackAV1` object takes over to manage the GPU resources and execute the frame-by-frame decode loop.

- **Session and Resource Creation:** Upon the `_play()` call, the playback object will use the newly extended `RenderingDevice` API to create the core Vulkan Video objects. This includes a `VkVideoSessionKHR` to manage the stream's state and a `VkVideoSessionParametersKHR`. The parameters object is immediately populated with the AV1 Sequence Header data that was parsed and cached by the `VideoStreamAV1` resource.

- **Memory Allocation:** Several key GPU resources are allocated:

  - **Decoded Picture Buffer (DPB):** A `VkImage` is created with the `VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR` flag. This image will be an array texture (`VkImageViewType` of `VK_IMAGE_VIEW_TYPE_2D_ARRAY`), with the number of layers determined by the codec's requirements for reference frames (typically `max_ref_frames + 1` to accommodate the current frame being decoded plus all potential references).

- - **Bitstream Buffer:** A `VkBuffer` is created with the `VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR` flag. This buffer will serve as the source for compressed AV1 frame data uploaded from the CPU.
  - **Output Textures:** One or more `VkImage` resources are created to hold the final, display-ready frames. These will have usage flags like `VK_IMAGE_USAGE_SAMPLED_BIT` and `VK_IMAGE_USAGE_TRANSFER_DST_BIT`, as they will be the destination of a color conversion pass and sampled by shaders.
- **The `_update()` Loop:** This method forms the core of the playback logic and is executed on each engine frame.

  1.  **Data Loading:** The next compressed AV1 frame (a temporal unit) is read from the video file into a buffer in system memory.
  2.  **Bitstream Upload:** The compressed data is copied from the system memory buffer to the GPU's bitstream `VkBuffer`.
  3.  **Command Recording:** A new command buffer is recorded for submission to the video decode queue.
  4.  **Begin Coding Scope:** The command recording starts with `vkCmdBeginVideoCodingKHR`. This command binds the video session and parameters, and crucially, it is passed an array of `VkVideoReferenceSlotInfoKHR` structures. This array maps the DPB's `VkImage` array layers to the logical DPB slot indices, informing the driver about the current state of available reference pictures.

- **Decode Command:** The `vkCmdDecodeVideoKHR` command is recorded. This command is the heart of the operation. It is provided with a pointer to the bitstream buffer, the destination slot in the DPB for the newly decoded picture, and information about which DPB slots should be used as references for this specific frame.

- 2.  **End Coding Scope:** The video operations are concluded with `vkCmdEndVideoCodingKHR`.
  3.  **Queue Submission:** The completed command buffer is submitted to the video decode queue. This submission is configured to signal a `VkSemaphore` upon completion, which is essential for synchronization.

### 3.3. Synchronization: From Video Queue to Graphics Queue

The most complex aspect of the pipeline is ensuring correct synchronization between the asynchronous video and graphics queues. The video queue produces a decoded frame in a `VkImage`, and the graphics queue must consume this frame as a texture for rendering. Failure to synchronize these operations will lead to race conditions, resulting in severe visual artifacts, validation errors, or GPU crashes.

- **Semaphore-Based Synchronization:** The solution lies in using Vulkan's GPU-GPU synchronization primitives, specifically `VkSemaphore`.

  1.  When the decode command buffer is submitted to the video queue via `vkQueueSubmit`, the submission info is configured to signal a specific semaphore (e.g., `VideoDecodeCompleteSemaphore`) once all commands in the buffer have finished executing.
  2.  The main rendering command buffer, which will be submitted to the graphics queue, is configured to _wait_ on this same `VideoDecodeCompleteSemaphore`. The wait is specified with a pipeline stage mask (e.g., `VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT`), ensuring that the graphics pipeline will not proceed to any stage that might sample the video texture until the semaphore is signaled.
  3.  This creates an explicit dependency on the GPU timeline: the graphics queue will physically stall until the video queue signals that the decode operation is complete, guaranteeing that the texture data is valid and ready to be read.

- **Image Layout Transitions:** `VkImage` resources in Vulkan have an explicit layout that defines how the image's memory is organized and what operations can be performed on it. A pipeline barrier (`vkCmdPipelineBarrier`) must be used to transition the DPB image from its initial layout to `VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR` before decoding, and then from that layout to `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` after decoding is complete so it can be sampled by a shader. This transition barrier must be recorded in the graphics command buffer _after_ the semaphore wait but _before_ any draw call that uses the video texture.

A simple, single-buffered implementation where the CPU submits a decode job and then the graphics queue waits on it would lead to significant performance issues. If the GPU is still processing frame N when the CPU is ready to submit frame N+1, the CPU would be forced to wait for the GPU to finish, creating a pipeline stall and causing visible stutter. A robust implementation must therefore use a multi-buffered (double- or triple-buffered) scheme for its resources. This involves creating a pool of bitstream buffers, output textures, and synchronization primitives (fences and semaphores). On frame N, resource set `N % pool_size` is used. This decouples the CPU from the GPU, allowing the CPU to record commands for frame N+1 while the GPU is still busy processing frame N. By the time the CPU loops back to using the first resource set, the GPU work associated with its previous use will have long since completed, thus preventing stalls and ensuring smooth, parallel execution.

## Section 4: The In-Core AV1 Encoding Pipeline

The encoding pipeline is conceptually the reverse of decoding: it takes uncompressed image data from the Godot renderer and uses the GPU's dedicated hardware to produce a compressed AV1 bitstream. A primary goal is to achieve this with minimal performance impact, enabling use cases like real-time, in-engine video capture.

### 4.1. Frame Capture for Encoding

The input to a hardware video encoder must be a `VkImage` in a format the encoder understands, typically a YCbCr (YUV) format.

- **Acquiring the Source Image:** The source frame can originate from various points within Godot's rendering pipeline:

  - **`SubViewport` Output:** A `SubViewport` node renders a scene to an off-screen texture. The underlying `VkImage` of this texture can be used as the encoder's input, allowing for the capture of specific game elements, secondary cameras, or UI scenes.
  - **Main Swapchain Image:** For capturing exactly what the player sees, a copy of the main window's swapchain image can be made using a transfer command (`vkCmdBlitImage` or `vkCmdCopyImage`). This provides the final, post-processed frame as input.

- **Mandatory Color Space Conversion:** Godot, like most game engines, renders in an RGB or RGBA color space. However, hardware video encoders operate natively on YCbCr color formats for compression efficiency. Therefore, a mandatory pre-processing step is required to convert the source RGB `VkImage` to a YCbCr `VkImage`. This conversion should be performed on the GPU using a dedicated compute shader or a simple graphics pipeline pass (drawing a full-screen quad). This keeps the entire pipeline on the GPU, avoiding a costly round-trip to the CPU for a software-based color conversion.

### 4.2. The Zero-Copy Encode Loop

The encoding loop leverages the asynchronous video encode queue to offload the computationally expensive compression work from the CPU and rendering pipeline.

- **Initialization:** The process begins by verifying hardware support for `VK_KHR_video_encode_av1` and querying for specific encoding capabilities, such as supported rate control modes and quality levels. An encoder-specific

- `VkVideoSessionKHR` and `VkVideoSessionParametersKHR` are created via the `RenderingDevice`.
- **The `_encode_frame()` Loop:** This function would be called for each frame to be encoded.

  1.  **Acquire and Convert Source:** The source `VkImage` is obtained from the renderer and the GPU-side color space conversion is executed. A pipeline barrier must be used to ensure the conversion is complete before the encoder accesses the resulting YCbCr image.
  2.  **Command Recording:** Commands are recorded for the video encode queue.
  3.  **Begin Coding Scope:** `vkCmdBeginVideoCodingKHR` is called, binding the encode session.
  4.  **Encode Command:** `vkCmdEncodeVideoKHR` is recorded. This command takes the YCbCr `VkImage` as its source input and a `VkBuffer` as the destination for the compressed bitstream output. Similar to decoding, this command also requires reference frames from a DPB to perform inter-frame prediction.

- 2.  **End Coding Scope:** `vkCmdEndVideoCodingKHR` concludes the video operations.
  3.  **Queue Submission:** The command buffer is submitted to the encode queue. The submission is paired with a `VkFence`, which will be signaled by the GPU upon completion.
  4.  **Bitstream Retrieval:** The CPU can then wait on this `VkFence`. Once signaled, it is safe to map the destination `VkBuffer`, read back the compressed AV1 data, and write it to a file or network stream.

A crucial aspect of an efficient encoding pipeline is the management of reference frames. The AV1 codec uses previously encoded frames to predict future ones. The `vkCmdEncodeVideoKHR` command can optionally output a "reconstructed picture," which is a `VkImage` representing what the frame looks like _after_ being compressed and then immediately decompressed by the hardware. This reconstructed picture, not the original pristine input frame, must be used as the reference for encoding the next frame. This ensures that the encoder and a future decoder will be working from the identical set of reference frames, preventing visual artifacts and drift. This creates a feedback loop entirely on the GPU—the output of one encode operation becomes the input for the next—which is the cornerstone of a high-performance, high-quality hardware encoding pipeline.

### 4.3. Integration with Godot's Movie Maker

Godot's built-in Movie Maker mode is a feature designed for capturing high-quality video footage by rendering frames slower than real-time and saving them individually. Its current implementation is slow, writing uncompressed PNG images or using an inefficient MJPEG AVI format. There have been several proposals to enhance this feature with hardware acceleration.

The AV1 encoding pipeline developed here provides a perfect, high-performance backend for this feature. The Movie Maker's core logic can be modified: instead of saving a rendered frame to a PNG file (which involves a slow GPU-to-CPU readback and CPU-based compression), it would instead pass the rendered frame's `VkImage` to our `_encode_frame()` function. This would leverage the zero-copy pipeline to encode frames directly on the GPU with minimal overhead, making it possible to capture high-resolution, high-quality compressed video in real-time or near real-time, a significant improvement over the current system.

## Section 5: Advanced Implementation Challenges and Solutions

Implementing a production-ready video processing pipeline involves solving several complex challenges that go beyond simple API calls. Robust memory management, precise audio-video synchronization, and navigating a fragmented hardware ecosystem are critical for stability and performance.

### 5.1. Optimized GPU Memory Management for Video

Video processing is memory-intensive, requiring large, contiguous allocations for resources like the DPB image array and bitstream buffers. A naive approach of calling `vkAllocateMemory` for each individual resource is unviable in a real-world application.

- **The Problem:** Vulkan drivers impose a limit on the total number of active memory allocations, typically 4096, as specified by `VkPhysicalDeviceLimits::maxMemoryAllocationCount`. Furthermore, frequent allocations and deallocations at the driver level are slow operations that can cause significant stuttering and lead to memory fragmentation, where enough total memory is free but no single contiguous block is large enough to satisfy a new request.

- **Solution: Sub-allocation:** The universally accepted best practice in Vulkan is to adopt a sub-allocation strategy. The application should allocate a few very large
- `VkDeviceMemory` blocks (e.g., 128MB or 256MB) at startup and then manage this memory manually. When a new resource (like a buffer or image) is needed, a chunk is partitioned from one of these large blocks. This approach keeps the number of driver-level allocations low and gives the application full control over memory layout, reducing fragmentation. Libraries like the Vulkan Memory Allocator (VMA) are designed for this purpose, but for a self-contained engine module, a simpler linear or pool allocator can be implemented.
- **Memory Types and Aliasing:** Careful selection of memory types is crucial for performance. Resources that are exclusively accessed by the GPU, such as the DPB images, should be allocated from `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` memory for the fastest access. Buffers used to upload data from the CPU to the GPU (staging buffers) should be allocated from memory that is

`VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`. For transient resources, such as an intermediate texture used for color space conversion that is only needed for a brief part of the frame, memory aliasing can be employed. This technique binds multiple different resources to the same piece of device memory at different times, significantly reducing the overall memory footprint.

### 5.2. Achieving Robust Audio-Video Synchronization

Perhaps the most perceptible failure in a video playback system is incorrect audio-video synchronization, commonly known as a "lip-sync" error. This issue arises from the fundamentally different and variable latencies of audio and video processing pipelines.

- **The "Lip-Sync" Problem:** Audio is typically decoded on the CPU with low and highly predictable latency. In contrast, hardware video decoding latency is variable. It depends on the type of frame being decoded (I-frames are self-contained and fast, while P- and B-frames depend on other frames and are more complex), the current load on the GPU's video engine, and other system factors. If an application simply plays the audio packet that corresponds to the video frame it has just submitted for decoding, the audio will almost invariably be heard before the corresponding image appears on screen, as video processing takes longer. Human perception is highly sensitive to audio leading video, making this a critical issue to solve.

- **Solution: Presentation Timestamp (PTS) Driven Playback:** The correct solution is to decouple the audio and video pipelines and synchronize them based on timestamps.

  1.  **Timestamp Extraction:** During the initial parsing of the video file (demuxing), both the compressed data for each packet and its associated Presentation Timestamp (PTS) must be extracted. The PTS is a metadata value that indicates the exact time at which a frame should be displayed.
  2.  **Audio as the Master Clock:** The audio playback system (Godot's `AudioServer` and `AudioStreamPlayer`) should be treated as the master clock. Its playback position provides a reliable, constantly advancing timeline.
  3.  **Video as the Slave with a Frame Buffer:** The video decoding pipeline should operate asynchronously, decoding frames ahead of when they are needed for display. The resulting decoded textures, along with their PTS values, are stored in a small queue (a buffer of ready frames).
  4.  **Synchronization Logic:** In the main game loop (`_process`), the application's logic compares the current playback time of the master audio clock to the PTS of the frames in the ready-frame queue. It then selects and displays the video frame whose PTS is closest to, but not later than, the current audio time. If the video decoding falls behind, frames are skipped. If it gets too far ahead, it waits. This master-slave clocking mechanism is the standard industry approach to ensure that video playback remains perfectly synchronized to the audio, providing a smooth and natural viewing experience.

### 5.3. Managing the Driver and Platform Ecosystem

While Vulkan is a cross-platform standard, the implementation of its video extensions is still maturing across different hardware vendors and operating systems. A robust module must be defensive and aware of this fragmented ecosystem.

#### The Fragmentation Reality

Hardware support for AV1 decode and encode is generally available on modern GPUs: NVIDIA RTX 30-series (Ampere) and newer, AMD RDNA 2 (RX 6000) and newer, and Intel Arc and Xe2 graphics. On Linux, support is provided through the open-source Mesa drivers, with RADV (for AMD) and ANV (for Intel) having rapidly improving AV1 support. However, support can depend on specific driver versions and firmware.

| Vendor | GPU Generation         | Windows Driver | Linux Driver (Mesa) | AV1 Decode | AV1 Encode        |
| ------ | ---------------------- | -------------- | ------------------- | ---------- | ----------------- |
| NVIDIA | RTX 40 Series (Ada)    | \= 535.xx      | Proprietary         | Yes        | Yes               |
|        | RTX 30 Series (Ampere) | \= 535.xx      | Proprietary         | Yes        | Yes               |
| AMD    | RDNA 3 (RX 7000)       | \= 24.8.1      | Mesa = 24.1 (RADV)  | Yes        | Yes (Mesa = 25.2) |
|        | RDNA 2 (RX 6000)       | \= 24.8.1      | Mesa = 24.1 (RADV)  | Yes        | Yes (Mesa = 25.2) |
| Intel  | Arc / Xe2 (Battlemage) | Supported      | Mesa = 25.2 (ANV)   | Yes        | Supported         |

#### Robust Fallbacks and Error Handling

The module _must not assume_ that Vulkan Video support is available. The capability verification process detailed in Section 3.1 is absolutely critical. If the required extensions, queue families, or profile support are not found on the user's system, the `VideoStreamAV1` resource must fail to load or play gracefully. It should provide a clear and informative error message in the Godot console. There should be no attempt to fall back to a software AV1 decoder within this module, as that would defeat the entire purpose of a hardware-accelerated implementation and reintroduce the performance problems of the existing Theora system. It is the responsibility of the application developer to provide an alternative video format (such as the built-in Theora) for systems that lack AV1 hardware support.

#### Testing and Reference Implementations

Given the potential for subtle differences in driver behavior, the implementation must be thoroughly tested on recent hardware and driver releases from NVIDIA, AMD, and Intel. The open-source NVIDIA Vulkan Video samples (`vk_video_samples`) serve as an invaluable public reference for a known-good implementation and can be used to debug and validate behavior.

## Section 6: Conclusion and Future Trajectory

The implementation of a native, in-core AV1 video encoder and decoder represents a significant advancement for the Godot Engine, replacing the outdated, CPU-bound Theora pipeline with a modern, high-performance, GPU-accelerated solution. By leveraging the power and explicit control of the Vulkan Video API, this module can deliver smooth, high-resolution video playback and real-time encoding capabilities with minimal impact on system resources.

### 6.1. Summary of the Implementation Path

The architectural blueprint outlined in this report provides a robust and maintainable path for this integration. The core decisions are:

1.  **Native C++ Module:** Building the functionality as a core engine module is essential for achieving the required performance, enabling zero-copy data paths, and ensuring tight integration with engine systems like the Movie Maker.
2.  **`RenderingDevice` Extension:** Extending the abstract `RenderingDevice` API, rather than writing backend-specific code in the module, is the key to a portable and future-proof design that aligns with Godot's core architecture.
3.  **Stateful Playback Objects:** Encapsulating the complexity of the Vulkan Video API within dedicated `VideoStreamPlaybackAV1` classes provides a clean interface to the rest of the engine and properly manages the stateful nature of video streams.
4.  **Asynchronous, Timestamp-Driven Pipeline:** A multi-buffered pipeline that uses presentation timestamps to synchronize video to a master audio clock is the definitive solution for achieving robust, stutter-free playback.

### 6.2. Recommendations for Future Development

With this foundational framework in place, several avenues for future enhancement become possible.

- **Expanding Codec Support:** The most significant benefit of the proposed `RenderingDevice` extension is its generic design. Adding support for other Vulkan-accelerated codecs, such as H.265, H.264, and VP9, would not require further modifications to the core `RenderingDevice` API. It would simply be a matter of implementing new `VideoStream` and `VideoStreamPlayback` classes that provide the correct codec-specific structures to the existing abstract functions. This dramatically lowers the barrier to entry for supporting a full suite of modern video formats.
- **Advanced Encoder Controls:** This report outlines a baseline encoder suitable for high-quality, real-time capture. Future work could expose more of the advanced features available in the `VK_KHR_video_encode_av1` extension. This includes providing users with fine-grained control over rate control modes (Constant Bitrate, Variable Bitrate), quality-versus-performance tuning levels, and the use of quantization maps to allocate more bits to specific regions of a frame. These features would elevate the module from a simple capture tool to a professional-grade in-engine encoder.

- **Upstreaming to Godot Core:** The ultimate objective of this work should be to refine, test, and contribute this module to the official Godot Engine repository. Integrating this functionality directly into the engine would provide all Godot users with a powerful, out-of-the-box solution for modern video playback and capture, solidifying Godot's position as a feature-complete, professional-grade game engine.
