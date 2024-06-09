# Godot Engine Open Telemetry

```gdscript
extends Node3D

var otel: OpenTelemetry = Opentelemetry.new()

func _ready() -> void:
	var error = otel.init_tracer_provider("godot", "localhost:4317", Engine.get_version_info())
	print(error)

func _process(_delta) -> void:
	var parent_span_id = otel.start_span("test-_ready")
	var span_id = otel.start_span_with_parent("test-child", parent_span_id)
	otel.add_event(span_id, "test-event")
	otel.set_attributes(span_id, {"test-key": "test-value"})
	otel.record_error(span_id, str(get_stack()))
	otel.end_span(span_id)
	otel.end_span(parent_span_id)

func _exit_tree() -> void:
	otel.shutdown()
```
