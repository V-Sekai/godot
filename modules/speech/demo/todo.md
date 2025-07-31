# Implementation Order

1. **Voice Packet Codec** - Binary protocol utilities (foundation) ✅ **TOMBSTONED**
2. **VoiceChatDemoController** - Main demo controller (core functionality)
3. **MockNetworkLayer** - Network simulation for testing
4. **Test Framework Integration** - Main test file with scene tree tests
5. **Module Registration** - Register new classes
6. **Test Data Setup** - Create test audio samples and scenarios

## Key Implementation Details

### Phase 1: Foundation Classes ✅ **TOMBSTONED**

-   ✅ `voice_packet_codec.h` - Binary encoding/decoding utilities header implemented
-   [ ] `voice_packet_codec.cpp` - Implementation file (pending)
-   [ ] `demo_test_utils.h/cpp` - Common utilities for test setup

### Phase 2: Core Demo Controller

-   `voice_chat_demo_controller.h/cpp` - Port entry_point.gd functionality
-   Integration with existing VoiceChat API
-   MockAudioDevice integration for synthetic testing

### Phase 3: Network Simulation

-   `mock_network_layer.h/cpp` - Port network_layer.gd with simulation capabilities
-   Packet history tracking for validation
-   Multi-peer simulation without actual networking

### Phase 4: Scene Tree Tests

-   `test_voice_chat_demo.h` - Main test file following GLTF pattern
-   Scene creation and management
-   End-to-end pipeline validation

### Phase 5: Integration

-   Update `register_types.cpp` to include demo classes
-   Create test data directory structure
-   Update build system integration

This will give you a complete, testable C++ implementation of the original GDScript demo that can run automated tests in the scene tree context, validate the entire VOIP pipeline, and serve as executable documentation of the VoiceChat API.
