def can_build(env, platform):
    return True


def configure(env):
    pass


def get_doc_classes():
    return [
        "OpenTelemetrySpan",
        "OpenTelemetryResource",
        "OpenTelemetryScope",
        "OpenTelemetryMetric",
        "OpenTelemetryLog",
        "OpenTelemetryDocument",
        "OpenTelemetryState",
        "OpenTelemetryExporterFile",
        "OpenTelemetryReflector",
        "OpenTelemetryTracer",
        "OpenTelemetryTracerProvider",
        "OpenTelemetry",
    ]


def get_doc_path():
    return "doc_classes"
