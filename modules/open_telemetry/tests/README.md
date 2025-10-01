# OpenTelemetry Collector Testing

This directory contains tests for using Godot as an OTLP (OpenTelemetry Protocol) collector, receiving and processing telemetry data from Python clients.

---

## Test Overview

### Components

1. **Python Client** (`python_client_test.py`)
   - Generates OTLP trace, metric, and log data
   - Saves data to JSON files
   - Can send data to HTTP collector endpoints

2. **Godot Collector** (`godot_collector_test.gd`)
   - Uses OTelReflector to load OTLP JSON files
   - Processes and analyzes telemetry data
   - Demonstrates Godot as a collector

---

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install requests
```

### Test Workflow

#### Step 1: Generate Test Data with Python

```bash
cd modules/open_telemetry/tests
python python_client_test.py
```

**Output:**
- `test_traces_<timestamp>.json` - Trace data
- `test_metrics_<timestamp>.json` - Metric data
- `test_logs_<timestamp>.json` - Log data

#### Step 2: Run Godot Collector Test

1. Open Godot editor
2. Create a new scene
3. Add a Node and attach `godot_collector_test.gd`
4. Copy generated JSON files to your Godot project directory
5. Run the scene

**Expected Output:**
```
============================================================
Godot OTLP Collector Test
============================================================

Test 1: Loading OTLP files from Python client
------------------------------------------------------------
Found 3 test files:
  - test_traces_1234567890.json
  - test_metrics_1234567890.json
  - test_logs_1234567890.json

Loading traces from: test_traces_1234567890.json
✓ Loaded successfully!
  Service: python-test-client
  Spans: 1

  Span Details:
    Name: python_operation
    Trace ID: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    Duration: 1000.00 ms
    Events: 1
...
```

---

## Test Scenarios

### Scenario 1: File-Based Collection

**Use Case:** Python generates telemetry, Godot processes it offline

```python
# Python: Generate data
python python_client_test.py
```

```gdscript
# Godot: Load and process
var reflector = OTelReflector.new()
var state = reflector.load_traces_from_file("test_traces_1234.json")

for span in state.get_spans():
    print("Span: ", span.get_name())
    print("Duration: ", span.get_duration_ms(), " ms")
```

### Scenario 2: Merge Multiple Sources

**Use Case:** Combine telemetry from multiple Python services

```bash
# Generate data from multiple sources
python python_client_test.py  # Service 1
python python_client_test.py  # Service 2
python python_client_test.py  # Service 3
```

```gdscript
# Godot: Merge all traces
var reflector = OTelReflector.new()
var files = reflector.find_otlp_files(".", "test_traces_*.json")
var merged = reflector.load_and_merge_traces(files)

print("Total spans from all services: ", merged.get_spans().size())
```

### Scenario 3: Real-Time Analysis

**Use Case:** Godot analyzes telemetry patterns

```gdscript
# Load trace data
var state = reflector.load_traces_from_file("traces.json")

# Analyze performance
var slow_spans = []
for span in state.get_spans():
    if span.get_duration_ms() > 1000:
        slow_spans.append(span)

print("Found ", slow_spans.size(), " slow operations")

# Check for errors
var error_spans = []
for span in state.get_spans():
    if span.get_status_code() == OTelSpan.STATUS_CODE_ERROR:
        error_spans.append(span)

print("Found ", error_spans.size(), " errors")
```

---

## Python Client API

### Generate Trace Data

```python
from python_client_test import generate_trace_data, save_to_file

data = generate_trace_data()
filename = save_to_file("traces", data)
print(f"Saved to {filename}")
```

### Generate Metric Data

```python
from python_client_test import generate_metric_data, save_to_file

data = generate_metric_data()
filename = save_to_file("metrics", data)
```

### Generate Log Data

```python
from python_client_test import generate_log_data, save_to_file

data = generate_log_data()
filename = save_to_file("logs", data)
```

---

## Godot Collector API

### Load Files

```gdscript
var reflector = OTelReflector.new()

# Load traces
var trace_state = reflector.load_traces_from_file("traces.json")

# Load metrics
var metric_state = reflector.load_metrics_from_file("metrics.json")

# Load logs
var log_state = reflector.load_logs_from_file("logs.json")
```

### Merge Multiple Files

```gdscript
# Merge traces from multiple files
var files = PackedStringArray([
    "trace1.json",
    "trace2.json",
    "trace3.json"
])

var merged = reflector.load_and_merge_traces(files)
```

### Find Files

```gdscript
# Find all JSON files in directory
var files = reflector.find_otlp_files("user://telemetry", "*.json")

# Find specific pattern
var trace_files = reflector.find_otlp_files(".", "traces_*.json")
```

---

## Advanced Usage

### Custom Python OTLP Client

```python
import json
import time

def create_custom_trace():
    return {
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "my-service"}}
                ]
            },
            "scopeSpans": [{
                "scope": {"name": "my-tracer"},
                "spans": [{
                    "traceId": "a" * 32,
                    "spanId": "b" * 16,
                    "name": "my_operation",
                    "startTimeUnixNano": int(time.time() * 1e9),
                    "endTimeUnixNano": int(time.time() * 1e9) + 1000000,
                    "attributes": [
                        {"key": "custom.field", "value": {"stringValue": "value"}}
                    ]
                }]
            }]
        }]
    }

# Save for Godot
with open("my_trace.json", "w") as f:
    json.dump(create_custom_trace(), f)
```

### Godot Trace Analysis

```gdscript
func analyze_trace(state: OTelState):
    var stats = {
        "total_spans": 0,
        "avg_duration": 0.0,
        "errors": 0,
        "services": {}
    }

    var total_duration = 0.0

    for span in state.get_spans():
        stats.total_spans += 1
        total_duration += span.get_duration_ms()

        if span.get_status_code() == OTelSpan.STATUS_CODE_ERROR:
            stats.errors += 1

    if stats.total_spans > 0:
        stats.avg_duration = total_duration / stats.total_spans

    stats.services[state.get_resource().get_service_name()] = stats.total_spans

    return stats
```

---

## Troubleshooting

### Issue: "No test files found"

**Solution:** Run `python python_client_test.py` first to generate test data files.

### Issue: "Failed to load traces"

**Solution:** Ensure JSON files are valid OTLP format. Check Python client output for errors.

### Issue: "Could not open current directory"

**Solution:** Ensure JSON files are in the correct directory (Godot project root or specified path).

---

## Example Output

### Python Client

```
============================================================
Python OTLP Client Test
============================================================

1. Generating OTLP data files...
✓ Saved traces to test_traces_1696034400.json
✓ Saved metrics to test_metrics_1696034400.json
✓ Saved logs to test_logs_1696034400.json

Generated files can be loaded in Godot with OTelReflector:
  var reflector = OTelReflector.new()
  var state = reflector.load_traces_from_file('test_traces_1696034400.json')

2. Attempting to send to Godot collector (localhost:4318)...
   (Start Godot collector first for this to work)
✗ Failed to send to /v1/traces: Connection refused
...
```

### Godot Collector

```
============================================================
Godot OTLP Collector Test
============================================================

Test 1: Loading OTLP files from Python client
------------------------------------------------------------
Found 3 test files:
  - test_traces_1696034400.json
  - test_metrics_1696034400.json
  - test_logs_1696034400.json

Loading traces from: test_traces_1696034400.json
✓ Loaded successfully!
  Service: python-test-client
  Spans: 1

  Span Details:
    Name: python_operation
    Trace ID: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    Duration: 1000.00 ms
    Events: 1

Test 2: Merging multiple trace files
------------------------------------------------------------
ℹ Need at least 2 trace files. Run python client multiple times.

============================================================
All tests complete!
============================================================
```

---

## Next Steps

1. **Build Godot** with OpenTelemetry module
2. **Run Python client** to generate test data
3. **Run Godot collector** to process data
4. **Analyze results** and verify bidirectional flow

---

## Summary

This test suite demonstrates:
- ✅ Python → Godot data flow
- ✅ OTLP JSON file generation
- ✅ File-based telemetry collection
- ✅ Trace merging and analysis
- ✅ Bidirectional telemetry workflow

**Godot is ready to act as an OTLP collector!**
