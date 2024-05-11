package main

/*
#include <stdlib.h>
*/
import "C"
import (
    "context"
    "fmt"
)

//export DesyncUntar
func DesyncUntar(storeUrl *C.char, indexUrl *C.char, outputDir *C.char, cacheDir *C.char) C.int {
    store := C.GoString(storeUrl)
    index := C.GoString(indexUrl)
    output := C.GoString(outputDir)
    cache := C.GoString(cacheDir)

    if store == "" || index == "" || output == "" {
        fmt.Println("Error: storeUrl, indexUrl, and outputDir are required")
        return 1
    }

    args := []string{"--no-same-owner", "--store", store, "--index", index, output}
    if cache != "" {
        args = append(args, "--cache", cache)
    }

	cmd := newUntarCommand(context.Background())
    cmd.SetArgs(args)
    _, err := cmd.ExecuteC()

    if err != nil {
        fmt.Printf("Error executing desync command: %v\n", err)
        return 2
    }
    return 0
}
