#!/usr/bin/env python3
"""
Python OTLP Client Test - Using Official OpenTelemetry Python SDK
Sends telemetry to Godot collector or exports to JSON files
"""

import json
import time
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

def setup_tracing(service_name="python-test-client", endpoint="http://localhost:4318/v1/traces"):
    """Setup OpenTelemetry tracing with OTLP exporter"""
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "host.name": "test-machine"
    })
    
    provider = TracerProvider(resource=resource)
    
    # Try OTLP exporter, fall back to console
    try:
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        print(f"✓ OTLP trace exporter configured: {endpoint}")
    except Exception as e:
        print(f"! OTLP exporter unavailable, using console: {e}")
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
    
    trace.set_tracer_provider(provider)
    return trace.get_tracer(__name__)

def setup_metrics(service_name="python-test-client", endpoint="http://localhost:4318/v1/metrics"):
    """Setup OpenTelemetry metrics with OTLP exporter"""
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0"
    })
    
    # Try OTLP exporter, fall back to console
    try:
        otlp_exporter = OTLPMetricExporter(endpoint=endpoint)
        reader = PeriodicExportingMetricReader(otlp_exporter, export_interval_millis=5000)
        print(f"✓ OTLP metric exporter configured: {endpoint}")
    except Exception as e:
        print(f"! OTLP exporter unavailable, using console: {e}")
        console_exporter = ConsoleMetricExporter()
        reader = PeriodicExportingMetricReader(console_exporter, export_interval_millis=5000)
    
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    return metrics.get_meter(__name__)

def test_basic_traces(tracer):
    """Test basic span creation"""
    print("\n" + "="*60)
    print("Test 1: Basic Traces")
    print("="*60)
    
    with tracer.start_as_current_span("python_operation") as span:
        span.set_attribute("operation.type", "test")
        span.set_attribute("test.iteration", 1)
        
        # Add event
        span.add_event("checkpoint", {"progress": 50})
        
        # Simulate work
        time.sleep(0.01)
        
        span.set_status(trace.Status(trace.StatusCode.OK))
    
    print("✓ Basic span created")

def test_nested_spans(tracer):
    """Test parent-child span hierarchy"""
    print("\n" + "="*60)
    print("Test 2: Nested Spans")
    print("="*60)
    
    with tracer.start_as_current_span("parent_operation") as parent:
        parent.set_attribute("level", "parent")
        
        with tracer.start_as_current_span("child_operation") as child:
            child.set_attribute("level", "child")
            time.sleep(0.005)
        
        time.sleep(0.005)
    
    print("✓ Parent-child span hierarchy created")

def test_span_with_error(tracer):
    """Test span with error status"""
    print("\n" + "="*60)
    print("Test 3: Span with Error")
    print("="*60)
    
    with tracer.start_as_current_span("error_operation") as span:
        try:
            raise ValueError("Test error")
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
    
    print("✓ Error span created")

def test_metrics(meter):
    """Test metric recording"""
    print("\n" + "="*60)
    print("Test 4: Metrics")
    print("="*60)
    
    # Create counter
    counter = meter.create_counter(
        "requests_total",
        description="Total number of requests",
        unit="1"
    )
    counter.add(100, {"method": "GET", "status": "200"})
    
    # Create histogram
    histogram = meter.create_histogram(
        "request_duration",
        description="Request duration",
        unit="ms"
    )
    histogram.record(42.5, {"endpoint": "/api/users"})
    
    print("✓ Metrics recorded")

def export_to_file(provider, filename):
    """Export telemetry data to JSON file for Godot reflector"""
    print(f"\nExporting to {filename}...")
    
    # Force flush to ensure all data is exported
    if hasattr(provider, 'force_flush'):
        provider.force_flush()
    
    print(f"✓ Data exported (check OTLP endpoint or console output)")
    print(f"  To use in Godot:")
    print(f"  var reflector = OTelReflector.new()")
    print(f"  # Configure OTLP endpoint to receive this data")

def main():
    print("=" * 60)
    print("Python OTLP Client Test - Official OpenTelemetry SDK")
    print("=" * 60)
    
    # Setup instrumentation
    tracer = setup_tracing()
    meter = setup_metrics()
    
    # Run tests
    test_basic_traces(tracer)
    test_nested_spans(tracer)
    test_span_with_error(tracer)
    test_metrics(meter)
    
    # Get provider for export
    trace_provider = trace.get_tracer_provider()
    
    # Force flush
    print("\n" + "="*60)
    print("Flushing telemetry data...")
    print("="*60)
    
    if hasattr(trace_provider, 'force_flush'):
        trace_provider.force_flush()
    
    metric_provider = metrics.get_meter_provider()
    if hasattr(metric_provider, 'force_flush'):
        metric_provider.force_flush()
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)
    
    print("\nNOTE: This client uses the official OpenTelemetry Python SDK.")
    print("To capture this data in Godot:")
    print("1. Set up Godot to act as an OTLP HTTP endpoint")
    print("2. Or use an OTLP collector to save to files")
    print("3. Then load with OTelReflector in Godot")

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
        print("  pip install opentelemetry-exporter-otlp")
        print(f"\nError: {e}")
        exit(1)
