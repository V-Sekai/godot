# OpenTelemetry Module for Godot

This module provides OpenTelemetry (OTLP) observability support for Godot Engine, enabling distributed tracing, metrics, and logging capabilities with full HTTP export to OTLP collectors.

## Status

**✅ PRODUCTION READY**

**Functional Status:**

-   ✅ **Godot C++ Implementation: COMPLETE**
    -   Full OTLP HTTP/JSON export
    -   Traces, Metrics, and Logs working
    -   Multi-sink support for redundant export
    -   Automatic batching and flushing
-   ✅ **Conformance Tests: 100% Pass Rate (11/11)**
-   ✅ **Cloud Integration: Verified** (Logflare, localhost collectors)

**Recent Improvements:**

-   ✅ Refactored JSON serialization to use native Dictionary/Array structures
-   ✅ Fixed all numeric type serialization (timestamps, integers, counts)
-   ✅ Eliminated manual string building for maintainability
-   ✅ Current timestamps (no longer 1970 epoch issues)

## Features

### Core Capabilities

-   **Distributed Tracing** - Full span hierarchy with parent-child relationships
-   **Metrics Collection** - Counters, Gauges, and Histograms
-   **Structured Logging** - With severity levels and attributes
-   **Multi-Sink Export** - Send to multiple collectors simultaneously
-   **Batch Processing** - Configurable batching and flush intervals
-   **Type-Safe Serialization** - Automatic JSON type handling (no more quote bugs!)

### OTLP Protocol Support

-   ✅ OTLP/HTTP JSON format
-   ✅ Trace spans with events, attributes, status
-   ✅ Metric data points with aggregations
-   ✅ Log records with severity and body
-   ✅ Resource attributes (service.name, etc.)
-   ✅ Current Unix nanosecond timestamps

## Directory Structure

```
modules/open_telemetry/
├── open_telemetry.h           # C++ API header
├── open_telemetry.cpp         # C++ implementation (COMPLETE)
├── register_types.cpp         # Module registration
├── SCsub                      # Build configuration
├── README.md                  # This file
├── test_project/              # Godot test project
│   ├── project.godot          # Test project config
│   ├── run_all_tests.bat      # Automated test runner
│   ├── test_logflare.py       # Cloud integration tests
│   ├── test_scenes/           # 11 conformance tests
│   └── scripts/               # Test utilities
└── doc_classes/               # Documentation
```

## C++ API

### Initialization

```cpp
// Basic setup
String init_tracer_provider(String name, String host, Dictionary attributes);
String set_headers(Dictionary headers);

// Multi-sink configuration
String add_sink(String sink_name, String hostname, Dictionary headers);
String remove_sink(String sink_name);
String set_sink_enabled(String sink_name, bool enabled);
```

### Tracing

```cpp
// Span lifecycle
String start_span(String name, SpanKind kind=SPAN_KIND_INTERNAL,
                  Array links=[], Dictionary attributes={});
String start_span_with_parent(String name, String parent_uuid,
                                SpanKind kind=SPAN_KIND_INTERNAL);
void end_span(String span_uuid);

// Span operations
void set_attributes(String span_uuid, Dictionary attributes);
void add_event(String span_uuid, String event_name,
               Dictionary attributes={}, uint64_t timestamp=0);
void set_status(String span_uuid, int status_code, String description="");
void update_name(String span_uuid, String name);
void record_exception(String span_uuid, String message, Dictionary attributes={});
```

### Metrics

```cpp
// Metric instruments
String create_counter(String name, String unit="", String description="");
String create_gauge(String name, String unit="", String description="");
String create_histogram(String name, String unit="", String description="");

// Record values
void increment_counter(String counter_id, float value=1.0, Dictionary attributes={});
void set_gauge(String gauge_id, float value, Dictionary attributes={});
void record_histogram(String histogram_id, float value, Dictionary attributes={});
```

### Logging

```cpp
void log_message(String level, String message, Dictionary attributes={});
```

### Lifecycle

```cpp
void set_flush_interval(int interval_ms);  // Default: 5000ms
void set_batch_size(int size);             // Default: 10
void flush_all();                          // Force immediate flush
String shutdown();                         // Flush and cleanup
```

## GDScript Usage

### Basic Example

```gdscript
extends Node

var otel: OpenTelemetry

func _ready():
    # Initialize OpenTelemetry
    otel = OpenTelemetry.new()
    otel.init_tracer_provider(
        "my-godot-game",
        "http://localhost:4318",
        {
            "deployment.environment": "production",
            "service.version": "1.0.0"
        }
    )

    # Create a trace
    var span = otel.start_span("game_session")
    otel.set_attributes(span, {
        "player.id": "player123",
        "level": 5
    })

    # Add events
    otel.add_event(span, "level_started", {"level_name": "Forest"})

    # End span
    otel.set_status(span, OpenTelemetry.STATUS_OK)
    otel.end_span(span)

    # Record metrics
    var fps_gauge = otel.create_gauge("fps", "frames", "Frames per second")
    otel.set_gauge(fps_gauge, Engine.get_frames_per_second())

    # Log message
    otel.log_message("INFO", "Game started", {"player_count": 1})

func _exit_tree():
    otel.shutdown()
```

### Multi-Sink Configuration

```gdscript
# Send to multiple collectors
otel.add_sink("local", "http://localhost:4318", {})
otel.add_sink("cloud", "https://otel.logflare.app:443", {
    "X-API-Key": "your-api-key",
    "X-Source-ID": "your-source-id"
})

# Disable a sink temporarily
otel.set_sink_enabled("local", false)
```

### Span Hierarchy

```gdscript
# Parent span
var request_span = otel.start_span("http_request")

# Child span
var db_span = otel.start_span_with_parent("database_query", request_span)
otel.set_attributes(db_span, {"query": "SELECT * FROM users"})
otel.end_span(db_span)

otel.end_span(request_span)
```

## ProjectSettings Configuration

Add to `project.godot`:

```ini
[opentelemetry]

enabled=true
endpoint="http://localhost:4318"
service_name="my-godot-app"
environment="production"
```

## Conformance Tests

The module includes 11 comprehensive conformance tests that validate OTLP protocol compliance:

```bash
cd modules/open_telemetry/test_project
.\run_all_tests.bat
```

**Test Coverage:**

1. ✅ basic_span - Simple span creation
2. ✅ span_attributes - Attribute types (string, int, float, bool)
3. ✅ span_hierarchy - Parent-child relationships
4. ✅ span_events - Event recording with timestamps
5. ✅ span_status - Status codes (OK, ERROR, UNSET)
6. ✅ span_kinds - All span kinds (INTERNAL, SERVER, CLIENT, etc.)
7. ✅ span_links - Cross-trace links
8. ✅ counter_metric - Monotonic counter aggregation
9. ✅ gauge_metric - Current value metrics
10. ✅ histogram_metric - Distribution metrics
11. ✅ complete_scenario - Full integration test

**Results:** 100% pass rate, all data successfully exported to collectors.

## Cloud Integration

Tested and verified with:

-   ✅ **Logflare** (https://logflare.app) - Cloud observability platform
-   ✅ **Local OTLP Collector** (localhost:4318)
-   ✅ **OpenTelemetry Collector** - Standard distributions

### Example: Logflare Setup

```gdscript
otel.init_tracer_provider("my-game", "https://otel.logflare.app:443", {
    "deployment.environment": "production"
})

otel.set_headers({
    "X-API-Key": "YOUR_API_KEY",
    "X-Source-ID": "YOUR_SOURCE_ID"
})
```

## Build Instructions

### Prerequisites

-   Godot 4.x source code
-   SCons build system
-   C++17 compiler

### Build Commands

```bash
# Standard build
scons platform=windows target=editor

# Development build with debugging
scons platform=windows target=editor dev_build=yes

# Custom module path
scons platform=windows custom_modules=modules/open_telemetry
```

### Module Integration

1. Clone/copy this module to `godot/modules/open_telemetry/`
2. Run SCons build (module auto-detected)
3. Module classes available in editor: `OpenTelemetry`, `OTelSpan`, etc.

## Architecture

### JSON Serialization (v2.0 - Refactored)

The module now uses native Godot Dictionary/Array structures with `JSON::stringify()` for type-safe serialization:

```cpp
// Build OTLP structure as nested Dictionaries
Dictionary span;
span["traceId"] = trace_id;
span["spanId"] = span_id;
span["startTimeUnixNano"] = (int64_t)start_time;  // Native type!
span["attributes"] = _dict_to_otlp_attributes(attrs);

// Single stringify call
String json = JSON::stringify(root_dict);
```

**Benefits:**

-   ✅ Automatic type handling (no more quoted numbers!)
-   ✅ Maintainable and readable
-   ✅ Debuggable (inspect Dictionary before export)
-   ✅ Follows Godot best practices

### Data Flow

```
Game Code
    ↓
OpenTelemetry API (GDScript/C++)
    ↓
Internal Buffering (Dictionaries)
    ↓
Batch Processing (configurable)
    ↓
JSON Serialization (JSON::stringify)
    ↓
HTTP Export (HTTPClient)
    ↓
OTLP Collectors (multi-sink)
```

## Performance

-   **Batching:** Default 10 items or 5 seconds
-   **Async Export:** Background HTTP requests
-   **Memory:** Minimal overhead, Dictionary-based storage
-   **CPU:** Efficient JSON serialization using native APIs

## Troubleshooting

### Data Not Appearing in Collector

1. **Check timestamps** - Should be current (2025+, not 1970)
2. **Verify collector endpoint** - Ensure reachable
3. **Check headers** - API keys, source IDs if using cloud
4. **Force flush** - Call `otel.flush_all()` before shutdown
5. **Enable logging** - Check console output for HTTP errors

### Common Issues

**Issue:** HTTP 200 but no data

-   **Cause:** Numeric fields serialized as strings (pre-v2.0)
-   **Fix:** Rebuild with latest code (Dictionary serialization)

**Issue:** Old timestamps (1970 epoch)

-   **Cause:** Using milliseconds instead of nanoseconds
-   **Fix:** Fixed in current version (uses `Time.get_unix_time_from_system() * 1e9`)

## API Reference

See `doc_classes/` for detailed XML documentation compatible with Godot editor.

## References

-   [OpenTelemetry Specification](https://opentelemetry.io/docs/specs/otel/)
-   [OTLP Specification](https://opentelemetry.io/docs/specs/otlp/)
-   [Logflare Integration Guide](https://logflare.app/guides/otlp)

## Contributing

Contributions welcome! Areas for enhancement:

-   Resource detection (OS, runtime info)
-   Sampling strategies
-   Compression support
-   gRPC transport
-   Additional test coverage

## License

Same as Godot Engine (MIT)

## Version History

### v2.0.0 (Current)

-   ✅ Refactored JSON serialization to Dictionary-based approach
-   ✅ Fixed all numeric type bugs (timestamps, integers, counts)
-   ✅ 100% conformance test pass rate
-   ✅ Cloud integration verified (Logflare)
-   ✅ Production ready

### v1.0.0

-   Initial implementation with manual JSON string building
-   Basic OTLP HTTP export
-   Known issues with numeric serialization
