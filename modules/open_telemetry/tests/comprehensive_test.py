#!/usr/bin/env python3
"""
Comprehensive OpenTelemetry Test Suite - Using Official Python SDK
Replaces test_project GDScript tests with official OpenTelemetry SDK
"""

import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Link, SpanKind

def setup_tracing(service_name="python-comprehensive-test"):
    """Setup OpenTelemetry tracing"""
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0"
    })
    
    provider = TracerProvider(resource=resource)
    console_exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(console_exporter))
    
    trace.set_tracer_provider(provider)
    return trace.get_tracer("comprehensive-test-tracer", "1.0.0")

# Test 1: Basic Span
def test_basic_span(tracer):
    """Test basic span creation - matches basic_span.gd"""
    print("\n" + "="*60)
    print("Test 1: Basic Span")
    print("="*60)
    
    with tracer.start_as_current_span("test-basic-span") as span:
        time.sleep(0.01)
    
    print("✓ Basic span created")

# Test 2: Span with Attributes
def test_span_attributes(tracer):
    """Test span attributes - matches span_attributes.gd"""
    print("\n" + "="*60)
    print("Test 2: Span Attributes")
    print("="*60)
    
    with tracer.start_as_current_span("span-with-attributes") as span:
        span.set_attribute("string_attr", "test_value")
        span.set_attribute("int_attr", 42)
        span.set_attribute("float_attr", 3.14)
        span.set_attribute("bool_attr", True)
    
    print("✓ Span with attributes created")

# Test 3: Span with Events
def test_span_events(tracer):
    """Test span events - matches span_events.gd"""
    print("\n" + "="*60)
    print("Test 3: Span Events")
    print("="*60)
    
    with tracer.start_as_current_span("span-with-events") as span:
        time.sleep(0.1)
        span.add_event("event_1", {"event_attr": "event_value_1"})
        
        time.sleep(0.4)
        span.add_event("event_2", {"event_attr": "event_value_2"})
    
    print("✓ Span with events created")

# Test 4: Span Hierarchy (Parent-Child)
def test_span_hierarchy(tracer):
    """Test parent-child spans - matches span_hierarchy.gd"""
    print("\n" + "="*60)
    print("Test 4: Span Hierarchy")
    print("="*60)
    
    with tracer.start_as_current_span("parent-span") as parent:
        with tracer.start_as_current_span("child-span") as child:
            time.sleep(0.01)
    
    print("✓ Parent-child span hierarchy created")

# Test 5: Span Kinds
def test_span_kinds(tracer):
    """Test different span kinds - matches span_kinds.gd"""
    print("\n" + "="*60)
    print("Test 5: Span Kinds")
    print("="*60)
    
    kinds = [
        ("span-kind-internal", SpanKind.INTERNAL),
        ("span-kind-server", SpanKind.SERVER),
        ("span-kind-client", SpanKind.CLIENT),
        ("span-kind-producer", SpanKind.PRODUCER),
        ("span-kind-consumer", SpanKind.CONSUMER)
    ]
    
    for name, kind in kinds:
        with tracer.start_as_current_span(name, kind=kind):
            pass
    
    print(f"✓ Created {len(kinds)} spans with different kinds")

# Test 6: Span Status
def test_span_status(tracer):
    """Test span status codes - matches span_status.gd"""
    print("\n" + "="*60)
    print("Test 6: Span Status")
    print("="*60)
    
    # UNSET status (default)
    with tracer.start_as_current_span("span-status-unset"):
        pass
    
    # OK status
    with tracer.start_as_current_span("span-status-ok") as span:
        span.set_status(trace.Status(trace.StatusCode.OK))
    
    # ERROR status
    with tracer.start_as_current_span("span-status-error") as span:
        span.set_status(trace.Status(trace.StatusCode.ERROR, "Status: ERROR"))
    
    print("✓ Created 3 spans with different statuses")

# Test 7: Span Links
def test_span_links(tracer):
    """Test span links - matches span_links.gd"""
    print("\n" + "="*60)
    print("Test 7: Span Links")
    print("="*60)
    
    # Create a span to link to
    with tracer.start_as_current_span("linked-span") as linked_span:
        linked_context = linked_span.get_span_context()
    
    # Create span with link
    link = Link(linked_context, attributes={"link_type": "related"})
    with tracer.start_as_current_span("span-with-links", links=[link]):
        pass
    
    print("✓ Span with links created")

# Test 8: Complete Scenario
def test_complete_scenario(tracer):
    """Test complete scenario - matches complete_scenario.gd"""
    print("\n" + "="*60)
    print("Test 8: Complete Scenario")
    print("="*60)
    
    # Root span (HTTP request)
    with tracer.start_as_current_span("http_request", kind=SpanKind.SERVER) as root:
        root.set_attribute("http.method", "GET")
        root.set_attribute("http.url", "/api/users")
        root.set_attribute("http.status_code", 200)
        
        # Child span (Database query)
        with tracer.start_as_current_span("db_query", kind=SpanKind.CLIENT) as db:
            db.set_attribute("db.system", "postgresql")
            db.set_attribute("db.statement", "SELECT * FROM users")
            
            db.add_event("query_start")
            time.sleep(0.02)
            db.add_event("query_end")
    
    print("✓ Complete scenario created")

# Test 9: Exception Recording
def test_exception_recording(tracer):
    """Test exception recording in spans"""
    print("\n" + "="*60)
    print("Test 9: Exception Recording")
    print("="*60)
    
    with tracer.start_as_current_span("operation_with_exception") as span:
        try:
            raise ValueError("Test exception for telemetry")
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
    
    print("✓ Exception recorded in span")

# Test 10: Multiple Attributes
def test_multiple_attributes(tracer):
    """Test span with many attributes"""
    print("\n" + "="*60)
    print("Test 10: Multiple Attributes")
    print("="*60)
    
    with tracer.start_as_current_span("span-many-attributes") as span:
        # Add various types of attributes
        span.set_attributes({
            "user.id": "12345",
            "user.name": "test_user",
            "user.role": "admin",
            "request.method": "POST",
            "request.path": "/api/data",
            "response.status": 201,
            "response.time_ms": 45.3,
            "cache.hit": True,
            "retry.count": 0
        })
    
    print("✓ Span with multiple attributes created")

# Test 11: Long-Running Operation
def test_long_operation(tracer):
    """Test span for longer operation with multiple events"""
    print("\n" + "="*60)
    print("Test 11: Long-Running Operation")
    print("="*60)
    
    with tracer.start_as_current_span("long-operation") as span:
        span.add_event("operation_started")
        time.sleep(0.01)
        
        span.add_event("phase_1_complete", {"progress": 25})
        time.sleep(0.01)
        
        span.add_event("phase_2_complete", {"progress": 50})
        time.sleep(0.01)
        
        span.add_event("phase_3_complete", {"progress": 75})
        time.sleep(0.01)
        
        span.add_event("operation_finished", {"progress": 100})
    
    print("✓ Long-running operation with events created")

def main():
    print("="*60)
    print("OpenTelemetry Comprehensive Test Suite")
    print("Official OpenTelemetry Python SDK")
    print("="*60)
    
    # Setup tracer
    tracer = setup_tracing()
    
    # Run all tests
    tests = [
        test_basic_span,
        test_span_attributes,
        test_span_events,
        test_span_hierarchy,
        test_span_kinds,
        test_span_status,
        test_span_links,
        test_complete_scenario,
        test_exception_recording,
        test_multiple_attributes,
        test_long_operation
    ]
    
    for test_func in tests:
        try:
            test_func(tracer)
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Force flush
    print("\n" + "="*60)
    print("Flushing telemetry data...")
    print("="*60)
    
    provider = trace.get_tracer_provider()
    if hasattr(provider, 'force_flush'):
        provider.force_flush()
    
    print("\n" + "="*60)
    print(f"All {len(tests)} tests complete!")
    print("="*60)
    
    print("\nNOTE: This test suite uses the official OpenTelemetry Python SDK.")
    print("Telemetry data is exported to console (visible above).")
    print("\nTo capture in Godot:")
    print("1. Configure Python to export to OTLP HTTP endpoint")
    print("2. Set up Godot as OTLP collector/receiver")
    print("3. Or use OTLP Collector to save to files")
    print("4. Load files with OTelReflector in Godot")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print("\n" + "="*60)
        print("ERROR: Missing OpenTelemetry dependencies")
        print("="*60)
        print("\nPlease install the official OpenTelemetry Python SDK:")
        print("  pip install opentelemetry-api")
        print("  pip install opentelemetry-sdk")
        print(f"\nError: {e}")
        exit(1)
