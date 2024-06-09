package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"errors"
	"log"
	"sync"
	"time"
	"fmt"
	"encoding/json"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"github.com/google/uuid"
)

var (
	tracerProvider *sdktrace.TracerProvider
	tracer         trace.Tracer
	mu             sync.Mutex
    spans    = make(map[uuid.UUID]trace.Span)
    contexts = make(map[uuid.UUID]context.Context)
	nextID         int64
)

//export InitTracerProvider
func InitTracerProvider(name *C.char, host *C.char, jsonString *C.char) *C.char {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var data map[string]interface{}
	err := json.Unmarshal([]byte(C.GoString(jsonString)), &data)
	if err != nil {
		return C.CString(fmt.Sprintf("Failed to decode JSON: %v", err))
	}

	attrs := make([]attribute.KeyValue, 0, len(data)+2)

	attrs = append(attrs, attribute.String("service.name", C.GoString(name)))
	attrs = append(attrs, attribute.String("library.language", "go"))

	for k, v := range data {
		strVal, ok := v.(string)
		if !ok {
			strVal = fmt.Sprintf("%v", v)
		}
		exists := false
		for _, attr := range attrs {
			if string(attr.Key) == k {
				exists = true
				break
			}
		}
		if !exists {
			attrs = append(attrs, attribute.String(k, strVal))
		}
	}

	resources, err := resource.New(
		ctx,
		resource.WithAttributes(attrs...),
	)
	if err != nil {
		return C.CString(fmt.Sprintf("Could not set resources: %v", err))
	}

	conn, err := grpc.DialContext(ctx, C.GoString(host), grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
	if err != nil {
		return C.CString(fmt.Sprintf("Failed to create gRPC connection to collector: %v", err))
	}
	traceExporter, err := otlptracegrpc.New(ctx, otlptracegrpc.WithGRPCConn(conn))

	batchSpanProcessor := sdktrace.NewBatchSpanProcessor(traceExporter)

	tracerProvider = sdktrace.NewTracerProvider(
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
		sdktrace.WithResource(resources),
		sdktrace.WithSpanProcessor(batchSpanProcessor),
	)

	otel.SetTracerProvider(tracerProvider)
	tracer = tracerProvider.Tracer(C.GoString(name))
	return nil
}

//export StartSpan
func StartSpan(name *C.char) *C.char {
    mu.Lock()
    defer mu.Unlock()

    if tracer == nil {
        return C.CString("00000000-0000-0000-0000-000000000000")
    }

    ctx := context.Background()
    _, span := tracer.Start(ctx, C.GoString(name))

    id := uuid.New()
    spans[id] = span

    return C.CString(id.String())
}

//export StartSpanWithParent
func StartSpanWithParent(name *C.char, parentID *C.char) *C.char {
    parentUUID, err := uuid.Parse(C.GoString(parentID))
    if err != nil {
        return C.CString("00000000-0000-0000-0000-000000000000")
    }

    mu.Lock()
    parentSpan, ok := spans[parentUUID]
    mu.Unlock()

    if !ok {   
        return C.CString("00000000-0000-0000-0000-000000000000")
    }

    ctx := trace.ContextWithSpan(context.Background(), parentSpan)
    _, span := tracer.Start(ctx, C.GoString(name))

    mu.Lock()
    id := uuid.New()
    spans[id] = span
    mu.Unlock()

    return C.CString(id.String())
}

//export AddEvent
func AddEvent(id *C.char, name *C.char) {
    uuidID, err := uuid.Parse(C.GoString(id))
    if err != nil || uuidID == uuid.Nil {
        return
    }

    mu.Lock()
    span := spans[uuidID]
    mu.Unlock()
    span.AddEvent(C.GoString(name))
}

//export SetAttributes
func SetAttributes(id *C.char, jsonStr *C.char) {
	uuidID, err := uuid.Parse(C.GoString(id))
	if err != nil || uuidID == uuid.Nil {
		return
	}

	mu.Lock()
	span := spans[uuidID]
	mu.Unlock()

	var data map[string]interface{}
	err = json.Unmarshal([]byte(C.GoString(jsonStr)), &data)
	if err != nil {
		log.Printf("Invalid JSON: %s", C.GoString(jsonStr))
		return
	}

	for k, v := range data {
		strVal, ok := v.(string)
		if !ok {
			strVal = fmt.Sprintf("%v", v)
		}
		attribute := attribute.String(k, strVal)
		span.SetAttributes(attribute)
	}
}

//export RecordError
func RecordError(id *C.char, err *C.char) {
    uuidID, parseErr := uuid.Parse(C.GoString(id))
    if parseErr != nil || uuidID == uuid.Nil {        
        return
    }

    mu.Lock()
    span := spans[uuidID]
    mu.Unlock()
    errGo := C.GoString(err)
    span.RecordError(errors.New(errGo))
    span.SetStatus(codes.Error, errGo)
}

//export EndSpan
func EndSpan(id *C.char) {
    uuidID, err := uuid.Parse(C.GoString(id))
    if err != nil || uuidID == uuid.Nil {
        return
    }

    mu.Lock()
    span := spans[uuidID]
    delete(spans, uuidID)
    mu.Unlock()
    span.End()
}

//export Shutdown
func Shutdown() *C.char {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := tracerProvider.Shutdown(ctx)
	if err != nil {
		return C.CString(err.Error())
	}

	return nil
}

//export DeleteContext
func DeleteContext(id *C.char) {
    uuidID, err := uuid.Parse(C.GoString(id))
    if err != nil {
        return
    }

    mu.Lock()
    delete(contexts, uuidID)
    mu.Unlock()
}

func main() {
	InitTracerProvider(C.CString("godot"), C.CString("localhost:4317"), C.CString("{}"))

	parentSpanID := StartSpan(C.CString("parent-function"))

	childSpanID := StartSpanWithParent(C.CString("child-function"), parentSpanID)

	AddEvent(childSpanID, C.CString("test-event"))

	SetAttributes(childSpanID, C.CString("{\"test-key\": \"test-value\"}"))

	RecordError(childSpanID, C.CString("test-error"))

	EndSpan(childSpanID)

	EndSpan(parentSpanID)

	err := Shutdown()
	if err != nil {
		log.Printf("Failed to shutdown TracerProvider: %s", C.GoString(err))
	}
}
